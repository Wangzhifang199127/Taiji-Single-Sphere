import torch
import torch.nn as nn
import torch.nn.functional as F

class TaijiLayerV16(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 【核心机制 1】强制关闭 Bias，确保无外界干涉时 F(0)=0
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)
        )

    def forward(self, Z):
        B, N, D = Z.shape
        Z_out = torch.zeros_like(Z)
        
        C = Z[:, 0, :].clone() 
        M = torch.norm(C, p=2, dim=-1, keepdim=True) 
        Z_out[:, 0, :] = C
        
        R_aggs, R_fields, C_norms = [], [], []
        
        for t in range(1, N):
            Z_t = Z[:, t, :]
            R_t = torch.norm(Z_t, p=2, dim=-1, keepdim=True)
            
            # 计算整体质心（包含自身，用于推进历史）
            C_overall = (M * C + R_t * Z_t) / (M + R_t + 1e-8)
            
            # 严格因果测距：当前词只看纯历史质心
            Z_hat = Z_t / (R_t + 1e-8)
            C_hist_hat = C / (torch.norm(C, p=2, dim=-1, keepdim=True) + 1e-8)
            
            S_t = torch.sum(Z_hat * C_hist_hat, dim=-1, keepdim=True)
            R_agg = S_t * torch.norm(C, p=2, dim=-1, keepdim=True)
            R_field = torch.tanh(R_agg)
            
            Input_to_MLP = Z_hat * R_field
            Delta_z = self.mlp(Input_to_MLP)
            
            # 【核心机制 2】应力-应变门控：形变严格受制于物理干涉力
            Z_out_t = Z_t + Delta_z * torch.abs(R_field)
            Z_out[:, t, :] = Z_out_t
            
            # 推进历史
            C = C_overall.clone()
            M = M + R_t
            
            R_aggs.append(R_agg.unsqueeze(-1))
            R_fields.append(R_field.unsqueeze(-1))
            C_norms.append(torch.norm(C_overall, p=2, dim=-1, keepdim=True).unsqueeze(-1))
            
        R_aggs = torch.cat([torch.zeros(B, 1, 1, device=Z.device)] + R_aggs, dim=1)
        R_fields = torch.cat([torch.zeros(B, 1, 1, device=Z.device)] + R_fields, dim=1)
        C_norms = torch.cat([torch.zeros(B, 1, 1, device=Z.device)] + C_norms, dim=1)
        
        return Z_out, C, R_aggs, R_fields, C_norms

class TaijiClassifierV16(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.layers = nn.ModuleList([TaijiLayerV16(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, 2)
        
    def forward(self, x, return_states=False):
        B, N = x.shape
        Z = self.token_emb(x) + self.pos_emb[:, :N, :]
        
        last_R_aggs, last_R_fields, last_C_norms = None, None, None
        C = None
        for layer in self.layers:
            Z, C, R_aggs, R_fields, C_norms = layer(Z)
            last_R_aggs, last_R_fields, last_C_norms = R_aggs, R_fields, C_norms
            
        logits = self.head(C)
        
        if return_states:
            return logits, Z, C, last_R_aggs, last_R_fields, last_C_norms
        return logits