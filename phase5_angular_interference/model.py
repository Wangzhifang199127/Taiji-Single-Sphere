import torch
import torch.nn as nn
import math

class ParallelV24Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 前置 MLP：宏观定相
        self.mlp_theta = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, 1, bias=False)
        )
        # 微观解耦频率
        self.omega = nn.Parameter(torch.ones(d_model))
        # Mamba 演化参数
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model) * 2 + 1))
        self.proj_B = nn.Linear(d_model, d_model, bias=False)
        # 后置微雕
        self.mlp_align = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_emb, return_states=False):
        B, L, D = x_emb.shape
        
        # 1. V20 力场界定 (纯方向内积)
        R_in = torch.norm(x_emb, p=2, dim=-1, keepdim=True) + 1e-5
        Z_hat = x_emb / R_in
        S = torch.bmm(Z_hat, Z_hat.transpose(1, 2))
        causal_mask = torch.tril(torch.ones(L, L, device=x_emb.device), diagonal=-1).bool()
        S = S.masked_fill(~causal_mask.unsqueeze(0), 0.0)
        
        W_macro, W_micro = S, S * (S ** 2)
        def calc_force(W):
            W_sum = W.sum(dim=-1, keepdim=True) + 1e-5
            return torch.bmm(W, x_emb) - W_sum * x_emb
            
        F_total = calc_force(W_macro) + calc_force(W_micro)
        evolution_scale = 1.0 / R_in
        Z_evolution = x_emb + (F_total * evolution_scale)
        R_out = torch.norm(Z_evolution, p=2, dim=-1, keepdim=True) + 1e-8
        V_obs = Z_evolution * (R_in / R_out)
        
        # 2. 前置 MLP 定相
        theta_input = torch.cat([x_emb, V_obs], dim=-1) 
        theta_global = math.pi * torch.tanh(self.mlp_theta(theta_input))
        theta_d = theta_global * self.omega.unsqueeze(0).unsqueeze(0)
        
        # 3. Mamba 演化 (复数旋转)
        cos_t, sin_t = torch.cos(theta_d), torch.sin(theta_d)
        A = torch.exp(-torch.exp(self.A_log))
        B_t = self.proj_B(x_emb)
        
        h_real, h_imag = torch.zeros_like(x_emb), torch.zeros_like(x_emb)
        curr_real, curr_imag = torch.zeros(B, D, device=x_emb.device), torch.zeros(B, D, device=x_emb.device)
        
        for t in range(L):
            rot_r = curr_real * cos_t[:, t, :] - curr_imag * sin_t[:, t, :]
            rot_i = curr_real * sin_t[:, t, :] + curr_imag * cos_t[:, t, :]
            curr_real = A * rot_r + B_t[:, t, :] * V_obs[:, t, :]
            curr_imag = A * rot_i + B_t[:, t, :] * V_obs[:, t, :] * 1.5
            h_real[:, t, :] = curr_real
            h_imag[:, t, :] = curr_imag
            
        # 4. 后置微雕
        Z_final = h_real + self.mlp_align(h_real)
        
        if return_states:
            return Z_final, theta_global, S, h_real, h_imag
        return Z_final

class V24_Model(nn.Module):
    def __init__(self, vocab_size, d_model, is_text=True, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model) if is_text else nn.Linear(1, d_model)
        self.block = ParallelV24Block(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, return_states=False):
        x_emb = self.embed(x)
        if return_states:
            Z_final, theta, S, h_r, h_i = self.block(x_emb, return_states=True)
            return self.classifier(Z_final[:, -1, :]), theta, S, h_r, h_i
        Z_final = self.block(x_emb)
        return self.classifier(Z_final[:, -1, :])

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, is_text=True, num_classes=2, num_layers=1, nhead=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model) if is_text else nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, L = x.shape[0], x.shape[1]
        x_emb = self.embed(x)
        if len(x_emb.shape) == 2: x_emb = x_emb.unsqueeze(1) # 适配1D时序输入
        x_emb = x_emb + self.pos_emb[:, :L, :]
        causal_mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        out = self.transformer(x_emb, mask=causal_mask)
        return self.classifier(out[:, -1, :])