import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.decomposition import PCA
from model import TaijiClassifierV16, TaijiLayerV16

# ==========================================
# 规范：全局随机种子固定
# ==========================================
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
# 实验 1：SST-2 情感分类与五维诊断
# ==========================================
def run_v16_sst2():
    print("\n>>> [实验 1] 启动太极 V16 真实情感分类...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=32), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(dataset["train"].shuffle(seed=42).select(range(5000)), batch_size=64)
    val_loader = DataLoader(dataset["validation"], batch_size=64)
    
    d_model, num_layers, epochs = 128, 2, 5
    model = TaijiClassifierV16(tokenizer.vocab_size, d_model, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch['input_ids'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
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
            preds = torch.argmax(model(x), dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f">>> 验证集准确率: {(correct/total)*100:.2f}%")
    
    # 五维全景诊断
    with torch.no_grad():
        sample_x = next(iter(val_loader))['input_ids'][0:1].to(device)
        logits, Z_out, C, R_aggs, R_fields, C_norms = model(sample_x, return_states=True)
        
        valid_len = (sample_x[0] != 0).sum().item()
        R_agg_np = R_aggs[0, :valid_len, 0].cpu().numpy()
        R_field_np = R_fields[0, :valid_len, 0].cpu().numpy()
        C_norms_np = C_norms[0, :valid_len, 0].cpu().numpy()
        
        Z_init = model.token_emb(sample_x[:, :valid_len]) + model.pos_emb[:, :valid_len, :]
        Z_in_norm = Z_init[0] / (torch.norm(Z_init[0], p=2, dim=-1, keepdim=True) + 1e-8)
        Z_out_norm = Z_out[0, :valid_len] / (torch.norm(Z_out[0, :valid_len], p=2, dim=-1, keepdim=True) + 1e-8)
        cos_sim = torch.sum(Z_in_norm * Z_out_norm, dim=-1).cpu().numpy()
        
        pca = PCA(n_components=2)
        Z_2d_init = pca.fit_transform(Z_in_norm.cpu().numpy())
        Z_2d_out = pca.transform(Z_out_norm.cpu().numpy())

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Taiji V16: Pure Physical Center of Mass Integration (SST-2)', fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(231)
    ax1.plot(loss_history, color='blue', linewidth=2, marker='o')
    ax1.set_title("1. Training Loss"); ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(232)
    colors = ['gray' if abs(v) < 0.05 else ('red' if v < 0 else 'green') for v in R_agg_np]
    ax2.bar(range(valid_len), R_agg_np, color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title("2. Heartbeat (Overall Interference)")
    
    ax3 = fig.add_subplot(233)
    ax3.plot(range(valid_len), R_field_np, marker='o', color='purple', linewidth=2)
    ax3.axhline(0, color='black', linestyle='--')
    ax3.set_title("3. Modulation Field")
    
    ax4 = fig.add_subplot(234)
    ax4.plot(range(valid_len), C_norms_np, marker='s', color='orange', linewidth=2)
    ax4.set_title("4. Center of Mass Norm (Anti-Explosion)")
    
    ax5 = fig.add_subplot(235)
    for i in range(valid_len):
        ax5.annotate("", xy=Z_2d_out[i], xytext=Z_2d_init[i], arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
    ax5.scatter(Z_2d_init[:, 0], Z_2d_init[:, 1], c='blue', label='Initial', alpha=0.3)
    ax5.scatter(Z_2d_out[:, 0], Z_2d_out[:, 1], c='red', label='Final', marker='X')
    
    C_2d = pca.transform((C[0] / (torch.norm(C[0], p=2) + 1e-8)).unsqueeze(0).cpu().numpy())
    ax5.scatter(C_2d[0, 0], C_2d[0, 1], c='gold', marker='*', s=300, label='Final Center', zorder=5)
    ax5.set_title("5. Phase Evolution"); ax5.legend()

    plt.tight_layout()
    plt.savefig("../assets/v16_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 实验 2：正交孤岛死寂测试 (纯正数学正交)
# ==========================================
def run_orthogonal_island():
    print("\n>>> [实验 2] 正在运行“正交孤岛”死寂测试...")
    d_model = 64
    model = TaijiLayerV16(d_model).to(device)
    model.eval()
    
    Z = torch.zeros(1, 10, d_model).to(device)
    
    # 1. 构造连贯语境
    base_dir = torch.nn.functional.normalize(torch.randn(1, 1, d_model, device=device), p=2, dim=-1)
    for i in range(9):
        Z[:, i, :] = base_dir + torch.randn(1, d_model, device=device) * 0.1
        Z[:, i, :] = torch.nn.functional.normalize(Z[:, i, :], p=2, dim=-1) * 1.0 
        
    # 2. 算出前 9 个词的真实质心
    M_temp = torch.norm(Z[:, 0, :], p=2, dim=-1, keepdim=True)
    C_temp = Z[:, 0, :].clone()
    for i in range(1, 9):
        R_t = torch.norm(Z[:, i, :], p=2, dim=-1, keepdim=True)
        C_temp = (M_temp * C_temp + R_t * Z[:, i, :]) / (M_temp + R_t + 1e-8)
        M_temp = M_temp + R_t
        
    # 3. 【核心修复】彻底修正正交化逻辑
    random_vec = torch.randn(1, d_model, device=device)
    # 投影计算必须使用单位向量
    C_temp_hat = torch.nn.functional.normalize(C_temp, p=2, dim=-1)
    projection = torch.sum(random_vec * C_temp_hat, dim=-1, keepdim=True)
    orthogonal_dir = random_vec - projection * C_temp_hat
    Z[:, 9, :] = torch.nn.functional.normalize(orthogonal_dir, p=2, dim=-1) * 1.0 
    
    with torch.no_grad():
        Z_out, C, R_aggs, R_fields, C_norms = model(Z)
        
    fields_np = R_fields[0, :, 0].cpu().numpy()
    diff_10 = torch.norm(Z_out[0, 9, :] - Z[0, 9, :], p=2).item()
    diff_5 = torch.norm(Z_out[0, 4, :] - Z[0, 4, :], p=2).item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['gray' if abs(v) < 0.05 else ('red' if v < 0 else 'green') for v in fields_np]
    ax1.bar(range(10), fields_np, color=colors, edgecolor='black')
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_title("Modulation Field (R_field)\n(Token 9 is Orthogonal Island)")
    ax1.set_xticks(range(10)); ax1.set_xticklabels([f"T{i}" for i in range(9)] + ["T9"])
    
    ax2.bar(['Coherent (T4)', 'Island (T9)'], [diff_5, diff_10], color=['blue', 'gray'])
    ax2.set_title("State Transition Step Size\n(Near-zero means Absolute Silence)")
    for i, v in enumerate([diff_5, diff_10]):
        ax2.text(i, v + 0.002, f"{v:.6f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("../assets/orthogonal_island.png", dpi=300)
    plt.show()
    print(f">>> 连贯词转移步长: {diff_5:.4f}")
    print(f">>> 孤岛词转移步长: {diff_10:.6f} (物理死寂验证)")

if __name__ == "__main__":
    run_v16_sst2()
    run_orthogonal_island()