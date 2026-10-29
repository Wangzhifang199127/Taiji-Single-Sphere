import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TaijiAblationLayer(nn.Module):
    def __init__(self, d_model, use_micro=True):
        super().__init__()
        self.use_micro = use_micro
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Z):
        B, N, D = Z.shape
        R_in = torch.norm(Z, p=2, dim=-1, keepdim=True)
        
        Z_norm = self.norm(Z)
        Z_hat = F.normalize(Z_norm, p=2, dim=-1)
        S = torch.bmm(Z_hat, Z_hat.transpose(1, 2))
        
        causal_mask = torch.tril(torch.ones(N, N, device=Z.device), diagonal=0).bool()
        S = S.masked_fill(~causal_mask.unsqueeze(0).expand(B, -1, -1), 0.0)
        diag_mask = torch.eye(N, device=Z.device).bool().unsqueeze(0).expand(B, -1, -1)
        S = S.masked_fill(diag_mask, 0.0)
        
        def calc_force(W):
            W_sum = W.sum(dim=-1, keepdim=True)
            return torch.bmm(W, Z) - W_sum * Z
        
        F_macro = calc_force(S)
        
        if self.use_micro:
            W_micro = S * (S ** 2) 
            F_micro = calc_force(W_micro)
        else:
            F_micro = 0.0
            
        F_total = F_macro + F_micro
        
        Z_evolution = Z + F_total * (1.0 / (R_in + 1e-5))
        R_out = torch.norm(Z_evolution, p=2, dim=-1, keepdim=True)
        Z_evolution = Z_evolution * (R_in / (R_out + 1e-8))
        
        Z_final = Z_evolution + self.mlp(Z_evolution)
        return Z_final

class TaijiAblationModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, use_micro=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.layer = TaijiAblationLayer(d_model, use_micro)

    def forward(self, x):
        B, N = x.shape
        Z = self.token_emb(x) + self.pos_emb[:, :N, :]
        Z = self.layer(Z)
        
        Z_final_norm = F.normalize(Z, p=2, dim=-1)
        vocab_emb = F.normalize(self.token_emb.weight, p=2, dim=-1)
        logits = torch.matmul(Z_final_norm, vocab_emb.T) * 10.0 
        return logits

def run_hard_ablation():
    vocab_size = 50
    seq_len = 40      # 【升级】：长序列
    batch_size = 32
    epochs = 150
    
    model_macro = TaijiAblationModel(vocab_size, use_micro=False).to(device)
    model_unified = TaijiAblationModel(vocab_size, use_micro=True).to(device)
    
    opt_macro = optim.Adam(model_macro.parameters(), lr=0.005)
    opt_unified = optim.Adam(model_unified.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    def get_batch():
        # 【升级】：生成带随机噪音的序列，增加宏观场的干扰
        start = torch.randint(1, 5, (batch_size, 1)).to(device)
        x = start + torch.arange(seq_len).to(device)
        # 注入随机噪音
        noise_mask = torch.rand_like(x.float()) > 0.7
        x = torch.where(noise_mask, torch.randint(1, vocab_size, x.shape).to(device), x)
        
        y = torch.roll(x, shifts=-1, dims=1)
        y[:, -1] = (y[:, -1] % (vocab_size - 1)) + 1
        return x, y

    loss_macro_hist = []
    loss_unified_hist = []
    
    print(">>> 开始高难度消融实验 (长序列 + 强噪音)...")
    for epoch in range(epochs):
        x, y = get_batch()
        
        model_macro.train()
        opt_macro.zero_grad()
        logits_macro = model_macro(x)
        loss_macro = criterion(logits_macro.view(-1, vocab_size), y.view(-1))
        loss_macro.backward()
        opt_macro.step()
        loss_macro_hist.append(loss_macro.item())
        
        model_unified.train()
        opt_unified.zero_grad()
        logits_unified = model_unified(x)
        loss_unified = criterion(logits_unified.view(-1, vocab_size), y.view(-1))
        loss_unified.backward()
        opt_unified.step()
        loss_unified_hist.append(loss_unified.item())
        
        if (epoch+1) % 30 == 0:
            print(f"Epoch {epoch+1:03d} | Macro-Only: {loss_macro.item():.4f} | Unified: {loss_unified.item():.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_macro_hist, label='Macro-Only (1st-order)', color='orange', linestyle='--')
    plt.plot(loss_unified_hist, label='Unified (1st + 3rd order)', color='red', linewidth=2)
    plt.title('Hard Ablation: Long Sequence with Noise (Copy Task)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_hard_ablation()
