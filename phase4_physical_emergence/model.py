import torch
import torch.nn as nn
import torch.nn.functional as F

class TaijiLayerV20(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 强制关闭 Bias，确保无外力时 F(0)=0 的绝对死寂
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model, bias=False)
        )

    def forward(self, Z):
        B, N, D = Z.shape
        R_in = torch.norm(Z, p=2, dim=-1, keepdim=True)
        
        # 【核心】没有 LayerNorm，直接提取纯方向
        Z_hat = F.normalize(Z, p=2, dim=-1)
        S = torch.bmm(Z_hat, Z_hat.transpose(1, 2))
        
        # 因果掩码与自身剔除
        causal_mask = torch.tril(torch.ones(N, N, device=Z.device), diagonal=0).bool()
        S = S.masked_fill(~causal_mask.unsqueeze(0).expand(B, -1, -1), 0.0)
        diag_mask = torch.eye(N, device=Z.device).bool().unsqueeze(0).expand(B, -1, -1)
        S = S.masked_fill(diag_mask, 0.0)
        
        W_macro = S
        W_micro = S * (S ** 2) 
        
        def calc_force(W):
            W_sum = W.sum(dim=-1, keepdim=True)
            return torch.bmm(W, Z) - W_sum * Z
        
        F_macro = calc_force(W_macro)
        F_micro = calc_force(W_micro)
        F_total = F_macro + F_micro
        
        # 能量自适应与防爆约束
        evolution_scale = 1.0 / (R_in + 1e-5) 
        Z_evolution = Z + (F_total * evolution_scale)
        R_out = torch.norm(Z_evolution, p=2, dim=-1, keepdim=True)
        Z_evolution = Z_evolution * (R_in / (R_out + 1e-8))
        
        Z_final = Z_evolution + self.mlp(Z_evolution)
        
        # 返回中间态 Z_evolution 用于死寂测试，S 矩阵和力用于诊断
        return Z_final, Z_evolution, S, F_macro, F_micro

class TaijiModelV20(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_classes=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.layers = nn.ModuleList([TaijiLayerV20(d_model) for _ in range(num_layers)])
        
        self.num_classes = num_classes
        if num_classes is not None:
            self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, return_states=False):
        B, N = x.shape
        Z = self.token_emb(x) + self.pos_emb[:, :N, :]
        
        last_S, last_Fm, last_Fmi = None, None, None
        for layer in self.layers:
            Z, _, S, Fm, Fmi = layer(Z)
            last_S, last_Fm, last_Fmi = S, Fm, Fmi
            
        if self.num_classes is not None:
            logits = self.head(Z[:, -1, :])
            if return_states: return logits, Z, last_S, last_Fm, last_Fmi
            return logits
        else:
            # 纯物理涌现
            Z_final_norm = F.normalize(Z, p=2, dim=-1)
            vocab_emb = F.normalize(self.token_emb.weight, p=2, dim=-1)
            logits = torch.matmul(Z_final_norm, vocab_emb.T) * 10.0 
            if return_states: return logits, Z, last_S, last_Fm, last_Fmi
            return logits