import torch
import torch.nn as nn
import torch.nn.functional as F

class TaijiLayer(nn.Module):
    """
    太极单球核心层：实现相位测距与能量聚合
    """
    def __init__(self, d_model, temperature=0.5):
        super().__init__()
        self.tau = temperature
        # 非线性塑形：打破线性演化，负责能量撑大后的空间扭曲
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, Z, causal_mask=None):
        # 1. 提取纯方向 (阴) 与 能量半径 (阳)
        R = torch.norm(Z, p=2, dim=-1, keepdim=True)
        Z_hat = Z / (R + 1e-8)
        
        # 2. 空间测距 (相位差即异同)
        S = torch.bmm(Z_hat, Z_hat.transpose(1, 2))
        
        # 3. 因果截断 (若为双向分类任务，则不传 mask)
        if causal_mask is not None:
            S = S.masked_fill(causal_mask, float('-inf'))
            
        # 4. 引力份额分配
        W = F.softmax(S / self.tau, dim=-1)
        
        # 5. 能量聚合 (相对包含)
        Delta_R = torch.bmm(W, R) 
        
        # 6. 能量注入与非线性扭曲
        Input_to_MLP = Z_hat * (R + self.alpha * Delta_R)
        Delta_Z = self.mlp(Input_to_MLP)
        
        # 7. 状态转移 (残差累加)
        return Z + Delta_Z

class TaijiClassifier(nn.Module):
    """太极架构：双向情感分类任务 (对应之前跑出的 70.2% 准确率的图)"""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_classes=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.layers = nn.ModuleList([TaijiLayer(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x, return_states=False):
        # 分类任务不加因果掩码（全句可见）
        Z = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]
        for layer in self.layers:
            Z = layer(Z, causal_mask=None)
        logits = self.head(Z.mean(dim=1))
        if return_states: return logits, Z
        return logits

class TaijiGenerator(nn.Module):
    """太极架构：自回归生成任务 (对应之前跑出的 V6 四维体检图)"""
    def __init__(self, vocab_size, d_model=128, num_layers=3, max_len=100):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList([TaijiLayer(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, return_states=False):
        B, N = x.shape
        # 生成任务严格开启因果视界
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
        
        Z = self.token_emb(x) + self.pos_emb[:, :N, :]
        for layer in self.layers:
            Z = layer(Z, causal_mask=causal_mask)
        logits = self.head(Z)
        if return_states: return logits, Z
        return logits

class StandardTransformerBaseline(nn.Module):
    """对照组：标准 Transformer (用于反衬太极架构)"""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        Z = self.emb(x) + self.pos_emb[:, :x.size(1), :]
        return self.head(self.transformer(Z).mean(dim=1))