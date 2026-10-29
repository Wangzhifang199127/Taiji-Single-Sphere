import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.decomposition import PCA
from model import TaijiModelV20

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

def plot_diagnostics(S, F_macro, F_micro, Z_init, Z_out, valid_len, title, save_path):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    S_np = S[0, :valid_len, :valid_len].cpu().numpy()
    Fm_norm = torch.norm(F_macro[0, :valid_len], p=2, dim=-1).cpu().numpy()
    Fmi_norm = torch.norm(F_micro[0, :valid_len], p=2, dim=-1).cpu().numpy()
    
    Z_in_norm = Z_init[0] / (torch.norm(Z_init[0], p=2, dim=-1, keepdim=True) + 1e-8)
    Z_out_norm = Z_out[0, :valid_len] / (torch.norm(Z_out[0, :valid_len], p=2, dim=-1, keepdim=True) + 1e-8)
    
    pca = PCA(n_components=2)
    Z_2d_init = pca.fit_transform(Z_in_norm.cpu().numpy())
    Z_2d_out = pca.transform(Z_out_norm.cpu().numpy())

    ax1 = fig.add_subplot(231); ax1.plot(Fm_norm, label='Macro', color='green'); ax1.plot(Fmi_norm, label='Micro', color='red'); ax1.set_title("Force Magnitudes"); ax1.legend()
    ax2 = fig.add_subplot(232); im = ax2.imshow(S_np, cmap='RdBu', aspect='auto', vmin=-1, vmax=1); ax2.set_title("Interaction Matrix S")
    ax3 = fig.add_subplot(233); ax3.bar(range(valid_len), torch.norm(Z_out[0, :valid_len], p=2, dim=-1).cpu().numpy(), color='purple'); ax3.set_title("Token Energy")
    
    ax4 = fig.add_subplot(234)
    for i in range(valid_len):
        ax4.annotate("", xy=Z_2d_out[i], xytext=Z_2d_init[i], arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
    ax4.scatter(Z_2d_init[:, 0], Z_2d_init[:, 1], c='blue', label='Initial', alpha=0.3)
    ax4.scatter(Z_2d_out[:, 0], Z_2d_out[:, 1], c='red', label='Final', marker='X')
    ax4.set_title("Phase Evolution Trajectory"); ax4.legend()
    
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.show()

def run_sst2_experiment():
    print("\n>>> [实验 1] V20 真实情感分类...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=32), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(dataset["train"].shuffle(seed=42).select(range(5000)), batch_size=64)
    val_loader = DataLoader(dataset["validation"], batch_size=64)
    
    model = TaijiModelV20(tokenizer.vocab_size, d_model=128, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    for epoch in range(5):
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
    print(f">>> SST-2 Accuracy: {(correct/total)*100:.2f}%")

    with torch.no_grad():
        sample_x = next(iter(val_loader))['input_ids'][0:1].to(device)
        logits, Z_out, S, Fm, Fmi = model(sample_x, return_states=True)
        valid_len = (sample_x[0] != 0).sum().item()
        Z_init = model.token_emb(sample_x[:, :valid_len]) + model.pos_emb[:, :valid_len, :]
        
        plot_diagnostics(S, Fm, Fmi, Z_init, Z_out, valid_len, 
                         'V20 (Pure Physics): SST-2 Diagnostics', "../assets/v20_sst2_diagnostics.png")

def run_copy_experiment():
    print("\n>>> [实验 2] V20 纯物理涌现 Copy Task...")
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

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)
        _, Z_out, S, Fm, Fmi = model(test_x, return_states=True)
        Z_init = model.token_emb(test_x) + model.pos_emb[:, :seq_len, :]
        plot_diagnostics(S, Fm, Fmi, Z_init, Z_out, seq_len, 
                         'V20 (Pure Emergence): Copy Task Diagnostics', "../assets/v20_copy_diagnostics.png")

if __name__ == "__main__":
    run_sst2_experiment()
    run_copy_experiment()