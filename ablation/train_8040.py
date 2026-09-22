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

# ================= 【创新实验】早期亚像素插值 (80-40 架构) =================
class RadarTransUNet1D(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=2):
        super(RadarTransUNet1D, self).__init__()
        
        # --- 1. 前置扩充 (Early Expansion: 40 -> 80) ---
        # 在输入端使用反卷积，将 4 通道提升到 32 通道，长度拉伸两倍进行空间插值
        self.up_early = nn.ConvTranspose1d(4, 32, kernel_size=2, stride=2) 
        self.enc1 = nn.Sequential(nn.Conv1d(32, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ReLU())
        
        # --- 2. Bottleneck (Transformer 长度变为 80) ---
        self.pos_encoder = nn.Parameter(torch.randn(1, 80, d_model) * 0.1) # 长度匹配 80
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. 后置压缩 (Late Compression: 80 -> 40) ---
        self.pool_late = nn.MaxPool1d(2) # 80 压回 40 过滤多余的插值幻觉
        # 通道压缩到 16，为全连接层减负
        self.dec_late = nn.Sequential(nn.Conv1d(d_model, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU())
        
        # --- 4. 分类头 (基于 40 Bins) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 40, 128), # 16通道 * 40 bins
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, 4, 40]
        
        # === 早期亚像素插值 ===
        x_up = self.up_early(x) # [B, 32, 80]
        e1 = self.enc1(x_up)    # [B, d_model, 80]
        
        # === Transformer 高分辨率交互 (80 Bins) ===
        t_in = e1.permute(0, 2, 1) # [B, 80, d_model]
        t_out = self.transformer(t_in + self.pos_encoder) # [B, 80, d_model]
        t_out = t_out.permute(0, 2, 1) # [B, d_model, 80]
        
        # === 回归真实物理尺度压缩 ===
        p_out = self.pool_late(t_out)  # [B, d_model, 40]
        d_out = self.dec_late(p_out)   # [B, 16, 40]
        
        # === 最终分类 ===
        return self.classifier(d_out)

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

print(f"Starting Training Model (80-40 Early Interpolation) on {DEVICE}...")
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
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'transunet_model_8040.pth'))
print("TransUNet Model (80-40) Saved.")