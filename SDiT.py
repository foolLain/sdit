import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode
from timm.models.layers import  trunc_normal_
from torch.nn.init import kaiming_normal_
import transformers
from typing import Optional, Tuple
import math
from spikingjelly.activation_based.functional import reset_net
import numpy as np

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    T, bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, :, None,:]
        .expand(T, bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(T, bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MLP(nn.Module):
    def __init__(self, dim,  enable_amp=False):
        super().__init__()
        self.mlp1_lif = LIFNode(decay_input=False,v_reset=0.,detach_reset=True, backend='cupy',step_mode='m')
        self.mlp1 = nn.Linear(dim,dim,bias=False)
        # self.ln1 = nn.LayerNorm(dim)

        self.mlp2_lif = LIFNode(decay_input=False,v_reset=0.,detach_reset=True, backend='cupy',step_mode='m')
        self.mlp2 = nn.Linear(dim,dim,bias=False)
        # self.ln2 = nn.LayerNorm(dim)

        self.mlp3_lif = LIFNode(decay_input=False,v_reset=0.,detach_reset=True, backend='cupy',step_mode='m')
        self.mlp3 = nn.Linear(dim,dim,bias=False)

        self.enable_amp = enable_amp

    def inner_forward(self, x):
        T, B, C, N = x.shape
        # x = x.permute(0,1,3,2)
        y1 = self.mlp1_lif(x)
        y1 = self.mlp1(y1)
        y2 = self.mlp2_lif(x)
        y2 = self.mlp2(y2)
        x = self.mlp3(y1*self.mlp3_lif(y2))
        return x

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            return self.inner_forward(x)

class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, local_heads=2,enable_amp=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim 
        self.num_heads = num_heads
        self.local_heads = local_heads
        self.n_rep = num_heads//local_heads
        self.mq_dim = self.dim // num_heads
        self.mkv_dim = self.mq_dim*self.local_heads

        self.wo = nn.Linear(self.dim, self.dim,bias=False)

        self.proj_lif = LIFNode(decay_input=False,v_reset=0.,detach_reset=True, backend='cupy',step_mode='m')
        
        self.q_w = nn.Linear(self.dim, self.dim, bias=False)
        self.q_lif = LIFNode(decay_input=False,v_reset=0.,detach_reset=True, backend='cupy',step_mode='m')

        self.k_w = nn.Linear(self.dim,self.mkv_dim, bias=False)
        self.k_lif = LIFNode(decay_input=False,v_reset=0.,detach_reset=True, backend='cupy',step_mode='m')

        self.v_w = nn.Linear(self.dim,self.mkv_dim, bias=False)
        self.v_lif = LIFNode(decay_input=False,v_reset=0.,detach_reset=True, backend='cupy',step_mode='m')

        self.enable_amp = enable_amp

    def inner_forward(self, x):

        T, B, C, N= x.shape

        x = self.proj_lif(x)
        x_for_qkv = x

        q_w_out = self.q_w(x_for_qkv)
        k_w_out = self.k_w(x_for_qkv)
        v_w_out = self.v_w(x_for_qkv)
        
        q_w_out = self.q_lif(q_w_out) #T,B,C,N
        q = q_w_out.reshape(T,B,C,self.num_heads, -1).permute(0,1,3,2,4) # T,B,HEADS,C,mq_dim

        k_w_out = self.k_lif(k_w_out)
        k = k_w_out.reshape(T,B,C,self.local_heads, -1)# T,B,Local_HEADS,C,mq_dim
        k = repeat_kv(k,self.n_rep).permute(0, 1, 3, 2, 4) #T,B,C,HEADS,mq_dim  -> T,B,HEADS,C,mq_dim

        v_w_out = self.v_lif(v_w_out)
        v = v_w_out.reshape(T,B,C,self.local_heads, -1)# T,B,Local_HEADS,C,mq_dim
        v = repeat_kv(v,self.n_rep).permute(0, 1, 3, 2, 4) #T,B,C,HEADS,mq_dim  -> T,B,HEADS,C,mq_dim


        attn = (q @ k.transpose(-2, -1)) # T,B,num_heads,N,C//num_heads * # T,B,num_heads,C//num_heads,N
        scores = attn * 0.1
        output = torch.matmul(scores, v)  # (T,bs, n_local_heads, seqlen, head_dim)
        x = output
        x = x.transpose(3, 4).reshape(T,B,self.dim,-1).contiguous().transpose(2, 3)
        x = self.wo(x)

        return x

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            return self.inner_forward(x)

class blk(nn.Module):
    def __init__(self,T,dim,num_heads,local_heads,enable_amp=False) :
        super().__init__()
        self.att = SpikingSelfAttention(dim=dim,
                                       num_heads=num_heads,local_heads=local_heads,
                                       enable_amp=enable_amp,)
        self.mlp = MLP(dim=dim,  enable_amp=enable_amp)
        self.att_norm = RMSNorm(dim, eps=1e-5)
        self.mlp_norm = RMSNorm(dim, eps=1e-5)
        self.theta_norm = RMSNorm(dim, eps=1e-5)
        self.enable_amp = enable_amp

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        self.re_token = nn.Parameter(torch.ones((T,1,dim)))
    
    def forward(self,x,c):
        with torch.cuda.amp.autocast(enabled=self.enable_amp): 
            T,B,C,D = x.shape   
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)    
            x = torch.concat([x,self.re_token[:,None,:,:].expand((T,B,1,D))],dim=2)     
            
            x = x + gate_msa.unsqueeze(1) * self.att(modulate(self.att_norm(x),shift_msa, scale_msa))    
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.mlp_norm(x), shift_mlp, scale_mlp))        

            x,theta = torch.split(x,dim=2,split_size_or_sections=[C,1])
            x = x + self.theta_norm(x*theta)
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x,c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=28, embed_dim=256, norm_layer=True):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = RMSNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        T,B,C,H,W = x.shape
        x = x.flatten(0,1)
        x = self.projection(x).flatten(2).transpose(1, 2) 
        _,N,D  = x.shape
        embedding = self.norm(x).reshape(T,B,N,D)
        return embedding

class transformer_snn(nn.Module):
    def __init__(self, input_size=28,patch_size=4,in_channels=3,
                  num_classes=10,
                 embed_dim=384, num_heads=8,local_heads=2, 
                 depths=2, T=4,  learn_sigma=True,enable_amp=False
                 ):
        super().__init__()
        assert (input_size * input_size) // (patch_size ** 2) 
        self.input_size= input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.ctx_len = int(input_size * input_size / patch_size ** 2)
        self.depths = depths
        self.T = T
        self.layers = nn.ModuleList()
        for j in range(0, depths):
            self.layers.append(blk(T=self.T,dim=embed_dim,local_heads=local_heads,
                  num_heads=num_heads, enable_amp=enable_amp,))
            
        self.x_embedder = PatchEmbed(in_channels=in_channels, patch_size=patch_size,embed_dim=embed_dim)
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels)
        self.final_conv = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1) 
        self.pos_embed = nn.Parameter(torch.zeros(self.T,1, self.ctx_len, embed_dim), requires_grad=False)

        self._init_weights()

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _init_weights(self,):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.ctx_len ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.projection.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.projection.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)        
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)


    def forward(self, x, t):
        # B,C,H,W
        x = x[None,:,:,:,:].repeat(self.T,1,1,1,1) # T,B,N,D
        # x = self.x_embedder(x)
        x = self.x_embedder(x)  + self.pos_embed  # T,B,N,D
        t = self.t_embedder(t)                   # (B, D)
        c = t

        for block in self.layers:
            x = block(x,c)                      # (T, B, N, D)

        x = x.mean(0)
        x = self.final_layer(x,c)                # (B, N, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, H, W)
        x = self.final_conv(x)
        reset_net(self)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

if __name__ == "__main__":
    model = transformer_snn().cuda()
    x = torch.randn((4,3,28,28)).cuda()
    t = torch.randint(0, 1000, (x.shape[0],), device='cuda')
    y = torch.randint(0,10,(x.shape[0],), device='cuda')
    output = model(x,t)
    print(output.shape)
