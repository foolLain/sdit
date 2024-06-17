torchrun --nproc_per_node 4  train.py \
--dataset cifar10 \
--results-dir results/cifar10 \
--image-size 32 \
--in-channels 3 \
--patch-size 2 \
--dim 512 \
--heads 12 \
--local-heads 6 \
--T 4 \
--depth 6 \
--num-classes 1 \
--epochs 1200 \
--cfg-scale 1.0 \
--global-batch-size 64 \
--lr 1e-4 \
--ckpt-every 50_000 \
--amp 

