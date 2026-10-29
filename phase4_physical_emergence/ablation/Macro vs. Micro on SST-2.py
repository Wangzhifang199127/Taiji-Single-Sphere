import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import random
import os

# 固定种子
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 消融模型定义 (复用之前的宏微观开关)
# ==========================================
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

class TaijiAblationClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, use_micro=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.layer = TaijiAblationLayer(d_model, use_micro)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        B, N = x.shape
        Z = self.token_emb(x) + self.pos_emb[:, :N, :]
        Z = self.layer(Z)
        # 分类取最后一个词的状态
        return self.head(Z[:, -1, :])

# ==========================================
# 运行对比
# ==========================================
def run_sst2_ablation():
    print(">>> 加载 SST-2 数据集...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=32), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(dataset["train"].shuffle(seed=42).select(range(5000)), batch_size=64)
    val_loader = DataLoader(dataset["validation"], batch_size=64)
    
    model_macro = TaijiAblationClassifier(tokenizer.vocab_size, use_micro=False).to(device)
    model_unified = TaijiAblationClassifier(tokenizer.vocab_size, use_micro=True).to(device)
    
    opt_macro = optim.Adam(model_macro.parameters(), lr=0.002)
    opt_unified = optim.Adam(model_unified.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    loss_macro_hist, loss_unified_hist = [], []
    acc_macro_hist, acc_unified_hist = [], []
    
    print(">>> 开始训练对比...")
    for epoch in range(5):
        # 训练
        model_macro.train(); model_unified.train()
        total_loss_m, total_loss_u = 0, 0
        
        for batch in train_loader:
            x, y = batch['input_ids'].to(device), batch['label'].to(device)
            
            # 宏观
            opt_macro.zero_grad()
            loss_m = criterion(model_macro(x), y)
            loss_m.backward(); opt_macro.step()
            total_loss_m += loss_m.item()
            
            # 宏微统一
            opt_unified.zero_grad()
            loss_u = criterion(model_unified(x), y)
            loss_u.backward(); opt_unified.step()
            total_loss_u += loss_u.item()
            
        avg_loss_m = total_loss_m / len(train_loader)
        avg_loss_u = total_loss_u / len(train_loader)
        loss_macro_hist.append(avg_loss_m)
        loss_unified_hist.append(avg_loss_u)
        
        # 评估准确率
        def evaluate(model):
            model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch['input_ids'].to(device), batch['label'].to(device)
                    preds = torch.argmax(model(x), dim=-1)
                    correct += (preds == y).sum().item(); total += y.size(0)
            return (correct/total)*100
            
        acc_m = evaluate(model_macro)
        acc_u = evaluate(model_unified)
        acc_macro_hist.append(acc_m)
        acc_unified_hist.append(acc_u)
        
        print(f"Epoch {epoch+1:02d} | Macro Loss: {avg_loss_m:.4f}, Acc: {acc_m:.2f}% | Unified Loss: {avg_loss_u:.4f}, Acc: {acc_u:.2f}%")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(loss_macro_hist, label='Macro-Only', color='orange', marker='o')
    ax1.plot(loss_unified_hist, label='Unified (V20)', color='red', marker='o')
    ax1.set_title("SST-2 Training Loss Ablation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    
    ax2.plot(acc_macro_hist, label='Macro-Only', color='orange', marker='s')
    ax2.plot(acc_unified_hist, label='Unified (V20)', color='red', marker='s')
    ax2.set_title("SST-2 Validation Accuracy Ablation")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sst2_ablation()