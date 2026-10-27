import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.decomposition import PCA
from model import TaijiUnifiedModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("../assets", exist_ok=True)

# ==========================================
# 实验 1：Copy Task
# ==========================================
def run_copy_task():
    print("\n>>> [实验 1] 启动太极 V19：规律性 Copy 任务验证...")
    vocab_size = 20
    d_model = 64
    seq_len = 10
    batch_size = 32
    epochs = 100
    
    # 不需要 num_classes，默认输出 vocab_size
    model = TaijiUnifiedModel(vocab_size, d_model=d_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    def get_batch():
        start = torch.randint(1, 5, (batch_size, 1)).to(device)
        x = start + torch.arange(seq_len).to(device)
        y = torch.roll(x, shifts=-1, dims=1)
        y[:, -1] = (y[:, -1] % (vocab_size - 1)) + 1
        return x, y

    loss_history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x, y = get_batch()
        logits, _, _, _, _ = model(x, return_states=True)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    print(f">>> Copy Task 最终 Loss: {loss_history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)
        _, _, S, Fm, Fmi = model(test_x, return_states=True)
        S_np = S[0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_history, color='blue')
    ax1.set_title("V19 Copy Task: Loss")
    im = ax2.imshow(S_np, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title("Interaction Matrix S (Pos/Neg)")
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig("../assets/v19_copy_diagnostics.png")
    plt.show()

# ==========================================
# 实验 2：SST-2 情感分类
# ==========================================
def run_sst2_task():
    print("\n>>> [实验 2] 启动太极 V19：真实情感分类 (SST-2)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=32), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(dataset["train"].shuffle(seed=42).select(range(5000)), batch_size=64)
    val_loader = DataLoader(dataset["validation"], batch_size=64)
    
    # 【修复核心】输入维度是 tokenizer.vocab_size (30522)，输出维度是类别数 2
    model = TaijiUnifiedModel(tokenizer.vocab_size, num_classes=2, d_model=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 5
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch['input_ids'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits, _, _, _, _ = model(x, return_states=True)
            last_logits = logits[:, -1, :] 
            loss = criterion(last_logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")
        
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch['input_ids'].to(device), batch['label'].to(device)
            logits, _, _, _, _ = model(x, return_states=True)
            preds = torch.argmax(logits[:, -1, :], dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = (correct/total) * 100
    print(f">>> 验证集准确率: {acc:.2f}%")

    # ==========================================
    # 生成五维物理诊断图
    # ==========================================
    with torch.no_grad():
        sample_x = next(iter(val_loader))['input_ids'][0:1].to(device)
        logits, Z_out, S, F_macro, F_micro = model(sample_x, return_states=True)
        
        valid_len = (sample_x[0] != 0).sum().item()
        S_np = S[0, :valid_len, :valid_len].cpu().numpy()
        Fm_norm = torch.norm(F_macro[0, :valid_len], p=2, dim=-1).cpu().numpy()
        Fmi_norm = torch.norm(F_micro[0, :valid_len], p=2, dim=-1).cpu().numpy()
        
        Z_init = model.token_emb(sample_x[:, :valid_len]) + model.pos_emb[:, :valid_len, :]
        Z_in_norm = Z_init[0] / (torch.norm(Z_init[0], p=2, dim=-1, keepdim=True) + 1e-8)
        Z_out_norm = Z_out[0, :valid_len] / (torch.norm(Z_out[0, :valid_len], p=2, dim=-1, keepdim=True) + 1e-8)
        
        pca = PCA(n_components=2)
        Z_2d_init = pca.fit_transform(Z_in_norm.cpu().numpy())
        Z_2d_out = pca.transform(Z_out_norm.cpu().numpy())

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Taiji V19: Macro-Micro Unified Field (SST-2)', fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(231)
    ax1.plot(loss_history, color='blue', marker='o')
    ax1.set_title("1. Training Loss")
    
    ax2 = fig.add_subplot(232)
    im = ax2.imshow(S_np, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title("2. Interaction Matrix S")
    plt.colorbar(im, ax=ax2)
    
    ax3 = fig.add_subplot(233)
    ax3.plot(range(valid_len), Fm_norm, label='Macro', color='green', marker='s')
    ax3.plot(range(valid_len), Fmi_norm, label='Micro', color='red', marker='^')
    ax3.set_title("3. Force Magnitudes")
    ax3.legend()
    
    ax4 = fig.add_subplot(234)
    R_out = torch.norm(Z_out[0, :valid_len], p=2, dim=-1).cpu().numpy()
    ax4.bar(range(valid_len), R_out, color='purple')
    ax4.set_title("4. Token Energy (R)")
    
    ax5 = fig.add_subplot(235)
    for i in range(valid_len):
        ax5.annotate("", xy=Z_2d_out[i], xytext=Z_2d_init[i], arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
    ax5.scatter(Z_2d_init[:, 0], Z_2d_init[:, 1], c='blue', label='Initial', alpha=0.3)
    ax5.scatter(Z_2d_out[:, 0], Z_2d_out[:, 1], c='red', label='Final', marker='X')
    ax5.set_title("5. Phase Evolution Trajectory")
    ax5.legend()

    plt.tight_layout()
    plt.savefig("../assets/v19_sst2_diagnostics.png")
    plt.show()

if __name__ == "__main__":
    run_copy_task()
    run_sst2_task()