import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 时间步嵌入
# =========================================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.freq_emb_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        """
        t: [B]
        return: [B, hidden_size]
        """
        half = self.freq_emb_size // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.freq_emb_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return self.mlp(emb)


# =========================================================
# 条件池化
# cond_seq: [B, Tau, cond_emb] -> [B, cond_emb]
# =========================================================
class ConditionPool(nn.Module):
    def __init__(self, cond_emb=256, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_emb, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, cond_seq):
        """
        cond_seq: [B, Tau, cond_emb]
        return:   [B, out_dim]
        """
        cond_global = cond_seq.mean(dim=1)
        return self.proj(cond_global)


# =========================================================
# FiLM 调制
# =========================================================
class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_channels * 2)
        )

    def forward(self, x, cond_vec):
        """
        x:        [B, C, L]
        cond_vec: [B, cond_dim]
        """
        scale, shift = self.to_scale_shift(cond_vec).chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1.0 + scale) + shift


# =========================================================
# 残差块
# =========================================================
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.film1 = FiLM(cond_dim, in_ch)
        self.film2 = FiLM(cond_dim, out_ch)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, cond_vec):
        h = self.norm1(x)
        h = self.film1(h, cond_vec)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.film2(h, cond_vec)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# =========================================================
# 自注意力块
# =========================================================
class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, L]
        """
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.view(B, self.num_heads, self.head_dim, L).transpose(2, 3)  # [B,H,L,D]
        k = k.view(B, self.num_heads, self.head_dim, L).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, L).transpose(2, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B,H,L,D]
        out = out.transpose(2, 3).contiguous().view(B, C, L)
        out = self.proj(out)

        return x + out


# =========================================================
# 下采样 / 上采样
# =========================================================
class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# =========================================================
# UNet 主体
# 输入输出都使用 token 形式：
# zt_tokens: [B, num_latent_tokens, token_dim]
# eps_hat:   [B, num_latent_tokens, token_dim]
# =========================================================
class UNet1DLatent(nn.Module):
    def __init__(
        self,
        token_dim=32,
        num_latent_tokens=32,
        cond_emb=256,
        base_channels=128,
        channel_mults=(1, 2, 4),
        num_heads=4,
        dropout=0.0,
        max_cond_len=8,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_latent_tokens = num_latent_tokens
        self.cond_emb = cond_emb
        self.base_channels = base_channels
        self.max_cond_len = max_cond_len

        # 输入: [B, T, D] -> [B, D, T]
        self.in_proj = nn.Conv1d(token_dim, base_channels, kernel_size=3, padding=1)

        # 时间 / 条件嵌入
        cond_dim_total = base_channels * 4
        self.t_embedder = TimestepEmbedder(cond_dim_total)
        self.cond_pool = ConditionPool(cond_emb=cond_emb, out_dim=cond_dim_total)

        # CFG 学习型空条件
        self.null_cond_token = nn.Parameter(
            torch.randn(1, max_cond_len, cond_emb) * 0.02
        )

        # encoder
        ch1 = base_channels * channel_mults[0]
        ch2 = base_channels * channel_mults[1]
        ch3 = base_channels * channel_mults[2]

        self.enc1 = ResBlock1D(base_channels, ch1, cond_dim_total, dropout=dropout)
        self.attn1 = SelfAttention1D(ch1, num_heads=num_heads)
        self.down1 = Downsample1D(ch1)

        self.enc2 = ResBlock1D(ch1, ch2, cond_dim_total, dropout=dropout)
        self.attn2 = SelfAttention1D(ch2, num_heads=num_heads)
        self.down2 = Downsample1D(ch2)

        self.enc3 = ResBlock1D(ch2, ch3, cond_dim_total, dropout=dropout)
        self.attn3 = SelfAttention1D(ch3, num_heads=num_heads)

        # middle
        self.mid1 = ResBlock1D(ch3, ch3, cond_dim_total, dropout=dropout)
        self.mid_attn = SelfAttention1D(ch3, num_heads=num_heads)
        self.mid2 = ResBlock1D(ch3, ch3, cond_dim_total, dropout=dropout)

        # decoder
        self.up2 = Upsample1D(ch3)
        self.dec2 = ResBlock1D(ch3 + ch2, ch2, cond_dim_total, dropout=dropout)
        self.dec2_attn = SelfAttention1D(ch2, num_heads=num_heads)

        self.up1 = Upsample1D(ch2)
        self.dec1 = ResBlock1D(ch2 + ch1, ch1, cond_dim_total, dropout=dropout)
        self.dec1_attn = SelfAttention1D(ch1, num_heads=num_heads)

        self.out_norm = nn.GroupNorm(8, ch1)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv1d(ch1, token_dim, kernel_size=3, padding=1)

    # -----------------------------------------------------
    # 条件处理
    # -----------------------------------------------------
    def get_null_cond(self, batch_size, cond_len, device):
        if cond_len > self.max_cond_len:
            extra_len = cond_len - self.max_cond_len
            extra = self.null_cond_token[:, -1:, :].repeat(1, extra_len, 1)
            null_cond = torch.cat([self.null_cond_token, extra], dim=1)
        else:
            null_cond = self.null_cond_token[:, :cond_len, :]
        return null_cond.repeat(batch_size, 1, 1).to(device)

    def build_cond_vector(self, t, cond_seq=None, force_uncond=False):
        """
        t:        [B]
        cond_seq: [B, Tau, cond_emb] or None
        return:   [B, cond_dim_total]
        """
        t_vec = self.t_embedder(t)

        if cond_seq is None or force_uncond:
            cond_len = 1 if cond_seq is None else cond_seq.size(1)
            cond_seq = self.get_null_cond(t.size(0), cond_len, t.device)

        c_vec = self.cond_pool(cond_seq)
        cond_vec = t_vec + c_vec
        return cond_vec

    # -----------------------------------------------------
    # 前向
    # -----------------------------------------------------
    def forward(self, zt_tokens, t, cond_seq=None, return_flat=False, force_uncond=False):
        """
        zt_tokens: [B, 32, 32]
        t:         [B]
        cond_seq:  [B, Tau, cond_emb] or None
        return:    [B, 32, 32]
        """
        x = zt_tokens.transpose(1, 2).contiguous()   # [B, D, T]
        cond_vec = self.build_cond_vector(t, cond_seq, force_uncond=force_uncond)

        x0 = self.in_proj(x)

        e1 = self.enc1(x0, cond_vec)
        e1 = self.attn1(e1)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, cond_vec)
        e2 = self.attn2(e2)
        d2 = self.down2(e2)

        e3 = self.enc3(d2, cond_vec)
        e3 = self.attn3(e3)

        m = self.mid1(e3, cond_vec)
        m = self.mid_attn(m)
        m = self.mid2(m, cond_vec)

        u2 = self.up2(m)
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.interpolate(u2, size=e2.shape[-1], mode="nearest")
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, cond_vec)
        u2 = self.dec2_attn(u2)

        u1 = self.up1(u2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.interpolate(u1, size=e1.shape[-1], mode="nearest")
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, cond_vec)
        u1 = self.dec1_attn(u1)

        out = self.out_norm(u1)
        out = self.out_act(out)
        out = self.out_proj(out)

        out = out.transpose(1, 2).contiguous()  # [B, T, D]

        if return_flat:
            return out.reshape(out.size(0), -1)
        return out

    @torch.no_grad()
    def forward_with_cfg(self, zt_tokens, t, cond_seq, cfg_scale=2.0, return_flat=False):
        eps_cond = self.forward(
            zt_tokens=zt_tokens,
            t=t,
            cond_seq=cond_seq,
            return_flat=False,
            force_uncond=False
        )
        eps_uncond = self.forward(
            zt_tokens=zt_tokens,
            t=t,
            cond_seq=cond_seq,
            return_flat=False,
            force_uncond=True
        )
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        if return_flat:
            return eps.reshape(eps.size(0), -1)
        return eps