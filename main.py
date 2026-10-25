import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import time

from model import TaijiClassifier, TaijiGenerator, StandardTransformerBaseline

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 实验 1：太极 (分类) vs Transformer 基线
# ==========================================
def run_taiji_classification():
    print(">>> 开始训练太极分类引擎 (SST-2)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=32), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(dataset["train"].shuffle().select(range(5000)), batch_size=64)
    val_loader = DataLoader(dataset["validation"], batch_size=64)
    
    model_tf = StandardTransformerBaseline(tokenizer.vocab_size).to(device)
    model_taiji = TaijiClassifier(tokenizer.vocab_size).to(device)
    
    opt_tf = optim.Adam(model_tf.parameters(), lr=0.001)
    opt_taiji = optim.Adam(model_taiji.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history_tf, history_taiji = [], []
    time_tf, time_taiji = [], []
    
    for epoch in range(5):
        model_tf.train(); model_taiji.train()
        loss_tf_sum, loss_taiji_sum = 0, 0
        t0 = time.time()
        
        for batch in train_loader:
            x, y = batch['input_ids'].to(device), batch['label'].to(device)
            opt_tf.zero_grad(); loss_tf = criterion(model_tf(x), y); loss_tf.backward(); opt_tf.step()
            loss_tf_sum += loss_tf.item()
            
            opt_taiji.zero_grad(); loss_taiji = criterion(model_taiji(x), y); loss_taiji.backward(); opt_taiji.step()
            loss_taiji_sum += loss_taiji.item()
            
        time_tf.append(time.time() - t0)
        history_tf.append(loss_tf_sum/len(train_loader))
        history_taiji.append(loss_taiji_sum/len(train_loader))
        
    # 最终准确率
    def evaluate(model):
        model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['input_ids'].to(device), batch['label'].to(device)
                correct += (torch.argmax(model(x), dim=-1) == y).sum().item(); total += y.size(0)
        return (correct/total) * 100

    acc_tf, acc_taiji = evaluate(model_tf), evaluate(model_taiji)

    # 画图 
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle('太极单球架构 vs Transformer 基线 (SST-2)', fontsize=18, fontweight='bold')
    
    axs[0].plot(history_tf, label='Transformer', marker='x', color='gray', linestyle='--')
    axs[0].plot(history_taiji, label='Taiji', marker='o', color='red')
    axs[0].set_title('Training Loss'); axs[0].legend(); axs[0].grid(alpha=0.3)
    
    bars = axs[1].bar(['Transformer', 'Taiji'], [acc_tf, acc_taiji], color=['gray', 'red'])
    axs[1].set_title('Validation Accuracy (%)'); axs[1].set_ylim(0, 100)
    for bar in bars: axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f"{bar.get_height():.1f}%", ha='center')
    
    axs[2].plot(time_tf, label='Transformer', marker='x', color='gray', linestyle='--')
    axs[2].plot(time_taiji, label='Taiji', marker='o', color='red')
    axs[2].set_title('Time per Epoch (s)'); axs[2].legend(); axs[2].grid(alpha=0.3)
    
    model_taiji.eval()
    with torch.no_grad():
        sample_x = next(iter(val_loader))['input_ids'][0:1].to(device)
        _, Z_final = model_taiji(sample_x, return_states=True)
        R_plot = torch.norm(Z_final, p=2, dim=-1)[0].cpu().numpy()
    axs[3].bar(range(len(R_plot)), R_plot, color='purple', alpha=0.7)
    axs[3].set_title('Taiji Token Energy (R) Landscape')
    
    plt.tight_layout(); plt.show()

# ==========================================
# 实验 2：太极 (生成) 自回归对话与四维体检图
# ==========================================

def run_taiji_classification():
    print(">>> 开始训练太极分类引擎 (SST-2, 多次随机种子验证)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=32), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(dataset["train"].shuffle().select(range(5000)), batch_size=64)
    val_loader = DataLoader(dataset["validation"], batch_size=64)
    
    seeds = [42, 100, 2024]
    acc_tf_list, acc_taiji_list = [], []
    
    history_tf, history_taiji, time_tf, time_taiji = [], [], [], []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"--- 正在测试 Seed: {seed} ---")
        torch.manual_seed(seed); np.random.seed(seed)
        
        model_tf = StandardTransformerBaseline(tokenizer.vocab_size).to(device)
        model_taiji = TaijiClassifier(tokenizer.vocab_size).to(device)
        
        opt_tf = optim.Adam(model_tf.parameters(), lr=0.001)
        opt_taiji = optim.Adam(model_taiji.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        if seed_idx == 0:
            history_tf, history_taiji, time_tf, time_taiji = [], [], [], []
            
        for epoch in range(5):
            model_tf.train(); model_taiji.train()
            loss_tf_sum, loss_taiji_sum = 0, 0
            t0 = time.time()
            
            for batch in train_loader:
                x, y = batch['input_ids'].to(device), batch['label'].to(device)
                opt_tf.zero_grad(); loss_tf = criterion(model_tf(x), y); loss_tf.backward(); opt_tf.step(); loss_tf_sum += loss_tf.item()
                opt_taiji.zero_grad(); loss_taiji = criterion(model_taiji(x), y); loss_taiji.backward(); opt_taiji.step(); loss_taiji_sum += loss_taiji.item()
                
            if seed_idx == 0: 
                time_tf.append(time.time() - t0)
                history_tf.append(loss_tf_sum/len(train_loader))
                history_taiji.append(loss_taiji_sum/len(train_loader))
                
        def evaluate(model):
            model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch['input_ids'].to(device), batch['label'].to(device)
                    correct += (torch.argmax(model(x), dim=-1) == y).sum().item(); total += y.size(0)
            return (correct/total) * 100

        acc_tf_list.append(evaluate(model_tf))
        acc_taiji_list.append(evaluate(model_taiji))

    mean_tf, std_tf = np.mean(acc_tf_list), np.std(acc_tf_list)
    mean_taiji, std_taiji = np.mean(acc_taiji_list), np.std(acc_taiji_list)
    
    print(f"\n>>> 结果汇总: TF {mean_tf:.1f}% ± {std_tf:.1f}% | Taiji {mean_taiji:.1f}% ± {std_taiji:.1f}%")

    os.makedirs("assets", exist_ok=True)

    # ============ 图1：带误差棒的柱状图 ============
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    bars = ax_bar.bar(['Transformer', 'Taiji (TSA)'], [mean_tf, mean_taiji], 
                      yerr=[std_tf, std_taiji], capsize=10, color=['gray', 'red'], alpha=0.8)
    ax_bar.set_title('Validation Accuracy (Averaged over 3 seeds)')
    ax_bar.set_ylabel('Accuracy (%)'); ax_bar.set_ylim(0, 100)
    for i, bar in enumerate(bars):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f"{bar.get_height():.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("assets/sst2_accuracy_errorbar.png", dpi=300)
    plt.close()

    # ============ 图2：3合1 全景图 (完美居中) ============
    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) # 改为 1行3列
    fig.suptitle('Taiji Architecture vs Transformer Baseline (SST-2)', fontsize=18, fontweight='bold')
    
    # 1. Loss
    axs[0].plot(history_tf, label='Transformer', marker='x', color='gray', linestyle='--')
    axs[0].plot(history_taiji, label='Taiji', marker='o', color='red')
    axs[0].set_title('Training Loss (Seed 42)'); axs[0].legend(); axs[0].grid(alpha=0.3)
    
    # 2. Time
    axs[1].plot(time_tf, label='Transformer', marker='x', color='gray', linestyle='--')
    axs[1].plot(time_taiji, label='Taiji', marker='o', color='red')
    axs[1].set_title('Time per Epoch (s)'); axs[1].legend(); axs[1].grid(alpha=0.3)
    
    # 3. Energy Landscape
    model_taiji.eval()
    with torch.no_grad():
        sample_x = next(iter(val_loader))['input_ids'][0:1].to(device)
        _, Z_final = model_taiji(sample_x, return_states=True)
        R_plot = torch.norm(Z_final, p=2, dim=-1)[0].cpu().numpy()
    axs[2].bar(range(len(R_plot)), R_plot, color='purple', alpha=0.7)
    axs[2].set_title('Taiji Token Energy (R) Landscape')
    
    plt.tight_layout()
    plt.savefig("assets/sst2_comparison.png", dpi=300)
    plt.close()
    print(">>> 两张图表已保存至 assets/ 文件夹。")

if __name__ == "__main__":
    run_taiji_classification()
    run_taiji_generation()