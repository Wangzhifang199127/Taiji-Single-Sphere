import torch
import torch.nn as nn
import torch.nn.functional as F

class TaijiUnifiedLayer(nn.Module):
    def __init__(self, d_model, eta_macro=0.1, eta_micro=0.5):
        super().__init__()
        self.eta_macro = eta_macro
        self.eta_micro = eta_micro
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Z):
        B, N, D = Z.shape
        Z_norm = self.norm(Z)
        Z_hat = F.normalize(Z_norm, p=2, dim=-1)
        S = torch.bmm(Z_hat, Z_hat.transpose(1, 2)) 
        
        causal_mask = torch.tril(torch.ones(N, N, device=Z.device), diagonal=0).bool()
        S = S.masked_fill(~causal_mask.unsqueeze(0).expand(B, -1, -1), 0.0)
        diag_mask = torch.eye(N, device=Z.device).bool().unsqueeze(0).expand(B, -1, -1)
        S = S.masked_fill(diag_mask, 0.0)
        
        W_macro = S
        W_macro_sum = W_macro.sum(dim=-1, keepdim=True)
        F_macro = torch.bmm(W_macro, Z) - W_macro_sum * Z
        
        W_micro = S * (S ** 2) 
        W_micro_sum = W_micro.sum(dim=-1, keepdim=True)
        F_micro = torch.bmm(W_micro, Z) - W_micro_sum * Z
        
        P_inertia = Z + self.mlp(Z_norm)
        Z_out = P_inertia + self.eta_macro * F_macro + self.eta_micro * F_micro
        
        R_in = torch.norm(Z, p=2, dim=-1, keepdim=True)
        Z_out = Z_out / (torch.norm(Z_out, p=2, dim=-1, keepdim=True) / R_in + 1e-8)
        
        return Z_out, S, F_macro, F_micro

class TaijiUnifiedModel(nn.Module):
    # 增加 num_classes 参数
    def __init__(self, vocab_size, num_classes=None, d_model=128, num_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.layers = nn.ModuleList([TaijiUnifiedLayer(d_model) for _ in range(num_layers)])
        
        # 如果传入了类别数，输出就是类别数；否则输出词表大小（用于语言模型）
        out_dim = num_classes if num_classes is not None else vocab_size
        self.head = nn.Linear(d_model, out_dim)
        
    def forward(self, x, return_states=False):
        B, N = x.shape
        Z = self.token_emb(x) + self.pos_emb[:, :N, :]
        
        last_S, last_Fm, last_Fmi = None, None, None
        for layer in self.layers:
            Z, S, Fm, Fmi = layer(Z)
            last_S, last_Fm, last_Fmi = S, Fm, Fmi
            
        logits = self.head(Z)
        
        if return_states:
            return logits, Z, last_S, last_Fm, last_Fmi
        return logits