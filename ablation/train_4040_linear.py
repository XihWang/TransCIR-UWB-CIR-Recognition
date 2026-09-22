import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time

# ================= 配置 =================
WORK_DIR = './model_output'
DATA_PATH = os.path.join(WORK_DIR, 'radar_data_processed.pkl')
SAVE_DIR = WORK_DIR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TransUNet 参数配置
BATCH_SIZE = 512  
EPOCHS = 600
LR = 0.0005 

# ================= 标签编码器 =================
class LabelEncoder:
    def __init__(self):
        self.vec_to_id = {}
        self.id_to_vec = {}
        self.num_classes = 0

    def fit(self, y_data):
        unique_labels = set()
        for row in y_data:
            t = tuple(np.round(row).astype(int))
            unique_labels.add(t)
        
        sorted_labels = sorted(list(unique_labels))
        for idx, label in enumerate(sorted_labels):
            label_list = [int(x) for x in label]
            self.vec_to_id[str(tuple(label_list))] = idx
            self.id_to_vec[idx] = label_list
            
        self.num_classes = len(sorted_labels)
        print(f"Label Encoder fitted. Found {self.num_classes} unique classes.")

    def transform(self, y_data):
        ids = []
        for row in y_data:
            label_list = [int(x) for x in np.round(row)]
            ids.append(self.vec_to_id[str(tuple(label_list))])
        return np.array(ids)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'vec_to_id': self.vec_to_id, 'id_to_vec': self.id_to_vec}, f)
        print(f"Label mapping saved to {path}")

# ================= 【Encoder 消融实验】 纯 Linear / ViT-style =================
class RadarTransUNet1D(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=2):
        super(RadarTransUNet1D, self).__init__()
        
        # --- 1. 替换 Encoder 为纯 Point-wise Linear 映射 ---
        # 完全抛弃 CNN 感受野，每一个 Bin 各自为战，独立从 4 维升至 64 维
        self.linear_enc = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # --- 2. Bottleneck (Transformer 保持原生 40 Bins) ---
        self.pos_encoder = nn.Parameter(torch.randn(1, 40, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. 分类头 (不需要Decoder) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 40, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x 进来是 [B, 40, 4]
        # 注意：这里不需要像 CNN 那样 permute 成 [B, 4, 40]，因为 Linear 直接处理最后一个维度
        
        # === Linear 独立编码阶段 ===
        e_out = self.linear_enc(x)  # 输出大小: [B, 40, 64]
        
        # === Transformer 全局交互 ===
        t_out = self.transformer(e_out + self.pos_encoder) # 输入输出均为: [B, 40, 64]
        t_out = t_out.permute(0, 2, 1) # 转成 [B, 64, 40] 准备统一展平，保持跟其他模型特征对齐
        
        # === 最终分类 ===
        return self.classifier(t_out)

# ================= 主流程 =================
print("Loading data...")
with open(DATA_PATH, 'rb') as f: 
    data = pickle.load(f)
X_train_np = data['X_train']
y_train_np = data['y_train']

encoder = LabelEncoder()
encoder.fit(y_train_np)
encoder.save(os.path.join(SAVE_DIR, 'label_encoder.json'))

print("Encoding labels...")
y_train_ids = encoder.transform(y_train_np)

print("Moving data to Device...")
X_train = torch.FloatTensor(X_train_np).to(DEVICE)
y_train = torch.LongTensor(y_train_ids).to(DEVICE)

model = RadarTransUNet1D(num_classes=encoder.num_classes).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
criterion = nn.CrossEntropyLoss()

print(f"Starting Training Model (Linear/ViT-style Encoder) on {DEVICE}...")
model.train()
start_time = time.time()
num_samples = len(X_train)

for epoch in range(EPOCHS):
    indices = torch.randperm(num_samples, device=DEVICE)
    total_loss = 0
    correct = 0
    
    for i in range(0, num_samples, BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]
        X_batch = X_train[batch_idx]  
        y_batch = y_train[batch_idx]
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_idx)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_batch).sum().item()
        
    avg_loss = total_loss / num_samples
    acc = 100 * correct / num_samples
    scheduler.step(avg_loss)
    
    if (epoch+1) % 10 == 0:
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {current_lr:.6f} | Time: {elapsed:.0f}s")

# 更改模型保存名字
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'transunet_model_linear.pth'))
print("TransUNet Model (Linear Encoder) Saved.")