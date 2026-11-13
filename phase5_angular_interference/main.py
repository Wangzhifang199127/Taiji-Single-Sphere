import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
from model import V24_Model, TransformerBaseline

# ==========================================
# 0. 规范：固定所有随机种子
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
os.makedirs("assets", exist_ok=True)
print(f"🔥 计算设备: {device}")

# ==========================================
# 1. 数据集定义
# ==========================================
# 数据集 A：真实开源情感 SST-2
def get_sst2_data():
    print("  [数据] 正在加载 SST-2 数据集...")
    try:
        dataset = load_dataset("nyu-mll/glue", "sst2")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        def process_split(split_name, max_samples):
            data = dataset[split_name].select(range(min(max_samples, len(dataset[split_name]))))
            texts = [example["sentence"] for example in data]
            labels = [example["label"] for example in data]
            encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
            return torch.utils.data.TensorDataset(encodings["input_ids"], torch.tensor(labels, dtype=torch.long))

        train_set = process_split("train", 5000)
        val_set = process_split("validation", 1000)
        return DataLoader(train_set, batch_size=32, shuffle=True), DataLoader(val_set, batch_size=32), tokenizer.vocab_size
    except Exception as e:
        print(f"  [数据] 网络异常({e})，使用离线兜底语料...")
        corpus = [("The movie is great but the ending is terrible.", 0), ("I love this, however, it broke.", 0), 
                  ("Best purchase ever, I am very happy.", 1), ("Highly recommended, fantastic quality.", 1)] * 500
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for text, _ in corpus:
            for w in text.lower().replace(",", "").replace(".", "").split():
                if w not in vocab: vocab[w] = len(vocab)
        X, Y = [], []
        for text, label in corpus:
            ids = [vocab.get(w, 1) for w in text.lower().replace(",", "").replace(".", "").split()]
            X.append((ids + [0]*32)[:32])
            Y.append(label)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
        return DataLoader(dataset, batch_size=32, shuffle=True), DataLoader(dataset, batch_size=32), len(vocab)

# 数据集 B：残缺波形恢复 (有规律)
class MissingWaveDataset(Dataset):
    def __init__(self, size=2000, seq_len=32, missing_rate=0.3):
        self.size, self.seq_len, self.missing_rate = size, seq_len, missing_rate
    def __len__(self): return self.size
    def __getitem__(self, idx):
        phase = np.random.uniform(-np.pi, np.pi)
        t = np.linspace(0, 4*np.pi, self.seq_len)
        wave = np.sin(1.5 * t + phase)
        label = 1 if phase > 0 else 0
        mask = np.random.rand(self.seq_len) < self.missing_rate
        wave_obscured = wave.copy()
        wave_obscured[mask] = 0.0
        return torch.tensor(wave_obscured, dtype=torch.float32).unsqueeze(-1), torch.tensor(label, dtype=torch.long)

# 数据集 C：随机延迟拷贝 (无规律)
class CopyDataset(Dataset):
    def __init__(self, size=2000, seq_len=15, vocab_size=10):
        self.size, self.seq_len, self.vocab_size = size, seq_len, vocab_size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        # 随机生成 2 到 9 之间的数字
        x = torch.randint(2, self.vocab_size, (self.seq_len,))
        target = x[0].clone() # 目标是记住第一个词
        x[0] = 1 # 特殊标记开始
        return x, target - 2 

# ==========================================
# 2. 训练与诊断绘图
# ==========================================
def train_and_eval(model, train_loader, val_loader, name, epochs=5):
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)
    print(f"  [{name}] Final Acc: {100.*correct/total:.2f}%")

def plot_diagnostics(task_name, model, sample_x, is_text=True):
    model.eval()
    with torch.no_grad():
        _, theta, S, h_real, h_imag = model(sample_x, return_states=True)
    valid_len = sample_x.shape[1]
    if is_text and len(sample_x[0].shape) > 0:
        valid_len = (sample_x[0] != 0).sum().item() or sample_x.shape[1]
        
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'V24 Diagnostics: {task_name}', fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(231); ax1.plot(range(valid_len), theta[0, :valid_len, 0].cpu(), color='red'); ax1.set_title("Global Phase (Theta)")
    ax2 = fig.add_subplot(232); ax2.bar(range(len(model.block.omega)), model.block.omega.detach().cpu(), color='purple'); ax2.set_title("Micro-frequencies (Omega)")
    ax3 = fig.add_subplot(233); im = ax3.imshow(S[0, :valid_len, :valid_len].cpu(), cmap='RdBu', aspect='auto', vmin=-1, vmax=1); ax3.set_title("V20 Force Matrix"); fig.colorbar(im, ax=ax3)
    
    ax4 = fig.add_subplot(234)
    ax4.plot(range(valid_len), h_real[0, :valid_len, 0].cpu(), label='Real Dim 0', color='blue')
    ax4.plot(range(valid_len), h_imag[0, :valid_len, 0].cpu(), label='Imag Dim 0', color='cyan', linestyle='--')
    ax4.set_title("Wave Interference"); ax4.legend()
    
    ax5 = fig.add_subplot(235); im2 = ax5.imshow(h_real[0, :valid_len].cpu().T, cmap='viridis', aspect='auto'); ax5.set_title("Real Feature Energy"); fig.colorbar(im2, ax=ax5)
    ax6 = fig.add_subplot(236); im3 = ax6.imshow(h_imag[0, :valid_len].cpu().T, cmap='magma', aspect='auto'); ax6.set_title("Imaginary Potential"); fig.colorbar(im3, ax=ax6)
    
    plt.tight_layout(); plt.savefig(f"assets/{task_name.replace(' ', '_')}_diag.png", dpi=200); plt.show()

# ==========================================
# 3. 核心实验执行
# ==========================================
if __name__ == "__main__":
    d_model = 64
    
    # ------------------------------------------
    # 任务 1: SST-2 文本情感 (验证语义因果规律)
    # ------------------------------------------
    print("\n>>> [任务 1] 真实语言：SST-2 情感因果分析 (单层对单层)")
    train_loader, val_loader, vocab_size = get_sst2_data()
    tf_sst = TransformerBaseline(vocab_size, d_model, is_text=True, num_classes=2, num_layers=1).to(device)
    v24_sst = V24_Model(vocab_size, d_model, is_text=True, num_classes=2).to(device)
    
    train_and_eval(tf_sst, train_loader, val_loader, "Transformer-SST2")
    train_and_eval(v24_sst, train_loader, val_loader, "V24-SST2")
    
    sample_x, _ = next(iter(val_loader))
    plot_diagnostics("SST2_Task", v24_sst, sample_x[0:1].to(device), is_text=True)

    # ------------------------------------------
    # 任务 2: 残缺波形恢复 (验证物理周期规律)
    # ------------------------------------------
    print("\n>>> [任务 2] 物理规律：残缺波形恢复 (30% 数据缺失)")
    wave_train = DataLoader(MissingWaveDataset(2000, 32, 0.3), batch_size=32, shuffle=True)
    wave_val = DataLoader(MissingWaveDataset(500, 32, 0.3), batch_size=32)
    
    tf_wave = TransformerBaseline(1, d_model, is_text=False, num_classes=2, num_layers=1).to(device)
    v24_wave = V24_Model(1, d_model, is_text=False, num_classes=2).to(device)
    
    train_and_eval(tf_wave, wave_train, wave_val, "Transformer-Wave")
    train_and_eval(v24_wave, wave_train, wave_val, "V24-Wave")
    
    sample_wave, _ = next(iter(wave_val))
    plot_diagnostics("Missing_Wave_Task", v24_wave, sample_wave[0:1].to(device), is_text=False)

    # ------------------------------------------
    # 任务 3: 随机延迟拷贝 (验证随机无规律不可学习)
    # ------------------------------------------
    print("\n>>> [任务 3] 纯随机乱码：延迟拷贝 (无规律，验证物理引擎的局限性)")
    copy_train = DataLoader(CopyDataset(2000, 15, 10), batch_size=32, shuffle=True)
    copy_val = DataLoader(CopyDataset(500, 15, 10), batch_size=32)
    
    tf_copy = TransformerBaseline(10, d_model, is_text=True, num_classes=8, num_layers=1).to(device)
    v24_copy = V24_Model(10, d_model, is_text=True, num_classes=8).to(device)
    
    # 我们多给几轮，看它能不能强行背下来
    train_and_eval(tf_copy, copy_train, copy_val, "Transformer-Copy", epochs=10)
    train_and_eval(v24_copy, copy_train, copy_val, "V24-Copy", epochs=10)
    
    sample_copy, _ = next(iter(copy_val))
    plot_diagnostics("Random_Copy_Task", v24_copy, sample_copy[0:1].to(device), is_text=True)

    print("\n✅ 三大任务实验执行完毕！图表已保存至 assets/ 目录。")