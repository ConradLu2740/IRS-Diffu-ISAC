# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointVAE(nn.Module):
    def __init__(self, num_points=2048, z_dim=256, hidden_dim=None, token_dim=None, num_latent_tokens=None):
        super().__init__()
        self.num_points = num_points
        self.use_token_mode = token_dim is not None and num_latent_tokens is not None

        if self.use_token_mode:
            self.token_dim = token_dim
            self.num_latent_tokens = num_latent_tokens
            self.z_dim = num_latent_tokens * token_dim
            _hidden = hidden_dim or 512

            self.encoder = nn.Sequential(
                nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Conv1d(256, _hidden, 1), nn.BatchNorm1d(_hidden), nn.ReLU(),
            )

            self.fc_mu = nn.Linear(_hidden, self.z_dim)
            self.fc_logvar = nn.Linear(_hidden, self.z_dim)

            self.decoder = nn.Sequential(
                nn.Linear(self.z_dim, _hidden), nn.BatchNorm1d(_hidden), nn.ReLU(),
                nn.Linear(_hidden, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                nn.Linear(2048, num_points * 3),
            )
        else:
            self.z_dim = z_dim

            self.encoder = nn.Sequential(
                nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Conv1d(512, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
            )

            self.fc_mu = nn.Linear(1024, z_dim)
            self.fc_logvar = nn.Linear(1024, z_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                nn.Linear(2048, num_points * 3),
            )

    def encode(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = torch.max(x, 2, keepdim=True)[0].squeeze(2)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(-20, 20)

        if self.use_token_mode:
            mu = mu.view(-1, self.num_latent_tokens, self.token_dim)
            logvar = logvar.view(-1, self.num_latent_tokens, self.token_dim)

        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        if self.use_token_mode:
            z_flat = z.view(z.size(0), -1)
            out = self.decoder(z_flat)
        else:
            out = self.decoder(z)
        pc = out.view(-1, self.num_points, 3)
        return pc, None

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        reconstructed_x, _ = self.decode(z)
        return reconstructed_x, mu, logvar, z

class AdvancedCondEncoder(nn.Module):
    def __init__(self, seq_len=8, input_size=114, hidden_size=128, out_emb=256):
        super().__init__()
        self.seq_len = seq_len
        

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=512, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, out_emb)
        )

    def forward(self, physical_signals):

        lstm_out, _ = self.lstm(physical_signals) # [B, 8, hidden_size * 2]
        

        proj_out = self.feature_proj(lstm_out)    # [B, 8, hidden_size]
        

        memory = self.transformer_encoder(proj_out) # [B, 8, hidden_size]
        
        c_seq = self.final_proj(memory)           # [B, 8, 256]
        
        return c_seq


# =========================================================================


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_emb_size = frequency_embedding_size

    def forward(self, t):
        half = self.freq_emb_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_emb_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)

class DiTBlock_CrossAttn(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) # 自注意力 (关注形状内部结构)

        self.norm_cross = nn.LayerNorm(hidden_size)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) 

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb, c_seq):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out1, _ = self.attn1(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out1
        
        x_norm_c = self.norm_cross(x)
        attn_out2, _ = self.attn2(query=x_norm_c, key=c_seq, value=c_seq)
        x = x + attn_out2 
        
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb):
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class LatentDiT1D_CrossAttn(nn.Module):

    def __init__(self, z_dim=256, cond_emb=256, hidden_size=256, depth=4, num_heads=8):
        super().__init__()
        self.z_dim = z_dim
        self.num_tokens = 16
        self.token_dim = z_dim // self.num_tokens
        
        self.x_embed = nn.Linear(self.token_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, hidden_size) * 0.02)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_proj = nn.Linear(cond_emb, hidden_size)
        
        self.blocks = nn.ModuleList([
            DiTBlock_CrossAttn(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, self.token_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, zt, t, cond_seq):
        B = zt.size(0)
        x = zt.view(B, self.num_tokens, self.token_dim)
        x = self.x_embed(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        c_seq = self.cond_proj(cond_seq)
        for block in self.blocks:
            x = block(x, t_emb, c_seq)
            
        x = self.final_layer(x, t_emb)
        return x.view(B, self.z_dim)


class LatentDiT_Token_CrossAttn(nn.Module):

    def __init__(self, token_dim=32, num_latent_tokens=32, cond_emb=256, hidden_size=256, depth=6, num_heads=8, max_cond_len=8):
        super().__init__()
        self.token_dim = token_dim
        self.num_latent_tokens = num_latent_tokens

        self.x_embed = nn.Linear(token_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_latent_tokens, hidden_size) * 0.02)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_proj = nn.Linear(cond_emb, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock_CrossAttn(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, token_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, zt, t, cond_seq=None):
        x = self.x_embed(zt) + self.pos_embed
        t_emb = self.t_embedder(t)
        if cond_seq is not None:
            c_seq = self.cond_proj(cond_seq)
        else:
            c_seq = torch.zeros(zt.size(0), 1, self.x_embed.out_features, device=zt.device)
        for block in self.blocks:
            x = block(x, t_emb, c_seq)
        x = self.final_layer(x, t_emb)
        return x

    @torch.no_grad()
    def forward_with_cfg(self, zt_tokens, t, cond_seq, cfg_scale=2.0):
        eps_cond = self.forward(zt_tokens, t, cond_seq=cond_seq)
        null_cond = torch.zeros_like(cond_seq)
        eps_uncond = self.forward(zt_tokens, t, cond_seq=null_cond)
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        return eps