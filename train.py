import os
# os.environ['CUDA_IS_VISIABLE'] = '2'
import shutil
import debugpy
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import math
from tqdm import tqdm
from diffusion import create_diffusion
import torchvision
from SDiT import transformer_snn
# from SDiT_noASM import transformer_snn
from torch.cuda.amp import autocast as autocast
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger

        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
            
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def broadcast_string(value, src, max_length=256):
    buffer = torch.zeros(max_length, dtype=torch.uint8,device="cuda")
    if dist.get_rank() == src:
        buffer[:len(value)] = torch.ByteTensor(list(value.encode()))
    dist.broadcast(buffer, src)
    return buffer.cpu().numpy().tobytes().decode().rstrip('\x00')

def main(args):
    """
    Trains a new SDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    checkpoint_dir = None
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f'{args}')
        logger.info(f"Experiment directory created at {experiment_dir}")
        
    else:
        logger = create_logger(None)

    # Create model:
    checkpoint_dir = broadcast_string(checkpoint_dir,0)
    # print(checkpoint_dir)
    model = transformer_snn(input_size=args.image_size,in_channels=args.in_channels,patch_size=args.patch_size,num_classes=args.num_classes,embed_dim=args.dim,num_heads=args.heads,local_heads=args.local_heads,depths=args.depth,T=args.T,enable_amp=args.amp)
    ckpt = None
    if args.load_ckpt:
        ckpt = torch.load(args.ckpt_dir)
        model.load_state_dict(ckpt['model'])

    # Note that parameter initialization is done within the SDiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    if args.load_ckpt:
        ema.load_state_dict(ckpt['ema'])
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    sample_diffusion = create_diffusion(str(args.num_sampling_steps))
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, )
    # if args.load_ckpt:
    #     opt.load_state_dict(ckpt['opt'])
    scaler = torch.cuda.amp.GradScaler()

    # Setup data:
    SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
    imagenet_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    MNIST_transform  = torchvision.transforms.Compose([torchvision.transforms.Resize(32),torchvision.transforms.ToTensor(),SetRange])
    FMNIST_transform  = torchvision.transforms.Compose([torchvision.transforms.Resize(32),torchvision.transforms.ToTensor(),SetRange])
    CIFAR10_transform  = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.Resize(32),torchvision.transforms.ToTensor(),SetRange])
    TinyImageNet_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    celeba_transform =  torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.CenterCrop(148),
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor(),
        SetRange])
    celebaHQ_transform =  torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        SetRange])
    FFHQ_transform =  torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize((128,128)),        
        torchvision.transforms.ToTensor(),
        SetRange])
    if args.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(root=args.data_path,train=True,transform=MNIST_transform,download=False)
    elif args.dataset == "fmnist":
        dataset = torchvision.datasets.FashionMNIST(root=args.data_path,train=True,transform=FMNIST_transform,download=False)
    elif args.dataset == "cifar10":
        dataset = torchvision.datasets.ImageFolder(root="/HDD/data/cifar10", transform=CIFAR10_transform)
    elif args.dataset == "TinyImageNet":
        dataset = torchvision.datasets.ImageFolder(root=args.data_path, transform=TinyImageNet_transform)
    elif args.dataset == "imagenet":
        dataset = ImageFolder(args.data_path, transform=imagenet_transform)
    elif args.dataset == "celeba":
        dataset = torchvision.datasets.ImageFolder(root='/HDD/dataset/celeba/',transform=celeba_transform)
    elif args.dataset == "celeba-hq":
        dataset = torchvision.datasets.ImageFolder(root='/HDD/dataset/CelebA-128/',transform=celebaHQ_transform)
    elif args.dataset == "ffhq":
        dataset = torchvision.datasets.ImageFolder(root='/HDD/dataset/FFHQ_256/images256x256/',transform=FFHQ_transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    begin_epoch = 0 if not args.load_ckpt else ckpt['epoch']
    del(ckpt)
    ckpt = None
    torch.cuda.empty_cache()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(begin_epoch,begin_epoch + args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            opt.zero_grad()
            x = x.to(device)
            y = y.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = None
            with autocast(args.amp):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            if not args.amp:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    save_ckpt(checkpoint_dir,train_steps,rank,model,ema,opt,args,epoch,logger)
                dist.barrier()
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                sample_step(ema,sample_diffusion,checkpoint_path,args.global_batch_size,device,rank,args)
                dist.barrier()

    # scheduler.step()
    save_ckpt(checkpoint_dir,train_steps,rank,model,ema,opt,args,epoch,logger)
    dist.barrier()
    logger.info("Done!")
    cleanup()


def save_ckpt(checkpoint_dir,train_steps,rank,model,ema,opt,args,epoch,logger):
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "args": args
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

def sample_step(model,diffusion,checkpoint_dir,num_samples,device,rank,args):

    model.eval() 
    using_cfg = args.cfg_scale > 1.0
    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = checkpoint_dir.split('/')[-1]
    a = os.path.dirname(os.path.dirname(checkpoint_dir))
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = os.path.join(a,"samples",folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = int(args.global_batch_size / dist.get_world_size())
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)
    dist.barrier()
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, model.input_size, model.input_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            if args.dataset == "mnist" or args.dataset == "fmnist":
                Image.fromarray(sample[:,:,0],mode="L").save(f"{sample_folder_dir}/{index:06d}.png")
            else:
                Image.fromarray(sample,mode="RGB").save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,default="MNIST")
    parser.add_argument("--data-path", type=str,default="/HDD/dataset/")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--model", type=str, default="SDiT")

    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--lr',type=float,default=1e-4)

    parser.add_argument("--image-size", type=int,  default=32)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument('--dim',type=int,default=512)
    parser.add_argument('--heads',type=int,default=16)
    parser.add_argument('--local-heads',type=int,default=4)
    parser.add_argument('--T',type=int,default=4)
    parser.add_argument('--depth',type=int,default=1)
    parser.add_argument("--num-classes", type=int, default=10)


    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100)

    parser.add_argument('--load-ckpt',action='store_true')
    parser.add_argument("--ckpt-dir", type=str, default=None)

    args = parser.parse_args()
    main(args)
