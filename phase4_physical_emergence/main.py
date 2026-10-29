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
from model import TaijiModelV20

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("assets", exist_ok=True)

# ==========================================
# 实验 1：V20 正交孤岛测试 (0.000000)
# ==========================================
def test_v20_nothing():
    print("\n>>> [实验 1] V20 正交孤岛绝对死寂测试...")
    d_model = 64
    model = TaijiModelV20(100, d_model=d_model, num_classes=2).to(device)
    model.eval()
    layer = model.layers[0]
    
    Z = torch.zeros(1, 10, d_model).to(device)
    base_dir = F.normalize(torch.randn(1, 1, d_model, device=device), p=2, dim=-1)
    for i in range(9):
        Z[:, i, :] = base_dir + torch.randn(1, d_model, device=device) * 0.1
        Z[:, i, :] = F.normalize(Z[:, i, :], p=2, dim=-1) * 1.0 
        
    with torch.no_grad():
        Z_hist = Z[0, :9, :] 
        U, S, Vh = torch.linalg.svd(Z_hist, full_matrices=True)
        orthogonal_dir = Vh[9, :].unsqueeze(0).unsqueeze(0)
        Z[:, 9, :] = F.normalize(orthogonal_dir, p=2, dim=-1) * 1.0 
        
        Z_final, Z_evolution, S, F_macro, F_micro = layer(Z)
        steps = torch.norm(Z_evolution[0] - Z[0], p=2, dim=-1).cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['gray' if steps[i] < 1e-6 else 'red' for i in range(10)]
    ax.bar(range(10), steps, color=colors)
    ax.set_title("V20 State Transition Step (External Force Only)\nToken 9 is Orthogonal to ALL History")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("State Transition Step")
    
    for i, v in enumerate(steps):
        if i == 9:
            ax.text(i, v + 0.05, f"{v:.6f}", ha='center', color='gray', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig("assets/v20_nothing_test.png", dpi=300)
    plt.show()
    print(f">>> 连贯词位移: {steps[4]:.4f}")
    print(f">>> 孤岛词位移: {steps[9]:.6f} (V20 绝对死寂验证)")

# ==========================================
# 实验 2：V20 真实情感分类 (SST-2)
# ==========================================
def run_sst2_experiment():
    print("\n>>> [实验 2] V20 真实情感分类 (SST-2)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=32), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    # 为保证快速验证，只取前 5000 条训练数据
    train_loader = DataLoader(dataset["train"].shuffle(seed=42).select(range(5000)), batch_size=64)
    val_loader = DataLoader(dataset["validation"], batch_size=64)
    
    model = TaijiModelV20(tokenizer.vocab_size, d_model=128, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch['input_ids'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d} | V20 Loss: {avg_loss:.4f}")
        
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch['input_ids'].to(device), batch['label'].to(device)
            preds = torch.argmax(model(x), dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = (correct/total) * 100
    print(f">>> SST-2 Accuracy: {acc:.2f}%")
    
    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o', color='blue')
    plt.title("V20: SST-2 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.savefig("assets/v20_sst2_loss.png", dpi=300)
    plt.show()

# ==========================================
# 实验 3：V20 纯物理涌现 Copy Task
# ==========================================
def run_v20_copy():
    print("\n>>> [实验 3] V20 纯物理涌现 Copy Task...")
    vocab_size, d_model, seq_len, batch_size, epochs = 20, 64, 10, 32, 100
    model = TaijiModelV20(vocab_size, d_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    def get_batch():
        start = torch.randint(1, 5, (batch_size, 1)).to(device)
        x = start + torch.arange(seq_len).to(device)
        y = torch.roll(x, shifts=-1, dims=1)
        y[:, -1] = (y[:, -1] % (vocab_size - 1)) + 1
        return x, y

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x, y = get_batch()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | V20 Copy Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # 按照顺序运行三个实验
    test_v20_nothing()
    run_sst2_experiment()
    run_v20_copy()