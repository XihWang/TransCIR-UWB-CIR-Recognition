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

# ================= 【全新终极架构】1D TransUNet =================
class RadarTransUNet1D(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=2):
        super(RadarTransUNet1D, self).__init__()
        
        # --- 1. Encoder (下采样提取特征，保留特征用于跳跃连接) ---
        self.enc1 = nn.Sequential(nn.Conv1d(4, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        self.pool1 = nn.MaxPool1d(2) # 40 -> 20
        
        self.enc2 = nn.Sequential(nn.Conv1d(32, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ReLU())
        self.pool2 = nn.MaxPool1d(2) # 20 -> 10
        
        # --- 2. Bottleneck (Transformer 全局感受野解决镜像模糊) ---
        self.pos_encoder = nn.Parameter(torch.randn(1, 10, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. Decoder (上采样 + 跳跃连接缝合高分辨率位置信息) ---
        self.up1 = nn.ConvTranspose1d(d_model, 32, kernel_size=2, stride=2) # 10 -> 20
        # 跳跃连接拼接通道：d_model(64来自enc2) + 32(来自up1) = 96
        self.dec1 = nn.Sequential(nn.Conv1d(d_model + 32, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        
        self.up2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2) # 20 -> 40
        # 跳跃连接拼接通道：32(来自enc1) + 16(来自up2) = 48
        self.dec2 = nn.Sequential(nn.Conv1d(32 + 16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU())
        
        # --- 4. 分类头 (不再求平均，直接展平保留目标所在的精确Bin坐标) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 40, 128), # 16通道 * 40 bins
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape进来是: [Batch, 40, 4] -> 给CNN需转为 [Batch, 4, 40]
        x = x.permute(0, 2, 1)
        
        # === 编码阶段 ===
        e1 = self.enc1(x)       # [B, 32, 40] (原汁原味高分辨率，留给dec2)
        p1 = self.pool1(e1)     # [B, 32, 20]
        e2 = self.enc2(p1)      # [B, 64, 20] (中等分辨率，留给dec1)
        p2 = self.pool2(e2)     # [B, 64, 10]
        
        # === Transformer 瓶颈阶段 ===
        t_in = p2.permute(0, 2, 1) # 转为 [B, 10, 64]
        t_out = self.transformer(t_in + self.pos_encoder) # [B, 10, 64]
        t_out = t_out.permute(0, 2, 1) # 转回 [B, 64, 10]
        
        # === 解码阶段 (结合跳跃连接) ===
        d1 = self.up1(t_out)               # 放大到 [B, 32, 20]
        c1 = torch.cat([e2, d1], dim=1)    # 任意门拼接: 64 + 32 = 96
        d1_out = self.dec1(c1)             # 融合输出 [B, 32, 20]
        
        d2 = self.up2(d1_out)              # 放大到 [B, 16, 40]
        c2 = torch.cat([e1, d2], dim=1)    # 任意门拼接: 32 + 16 = 48
        d2_out = self.dec2(c2)             # 融合输出 [B, 16, 40]
        
        # === 最终分类 ===
        return self.classifier(d2_out)

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

# 实例化 TransUNet 模型
model = RadarTransUNet1D(num_classes=encoder.num_classes).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
criterion = nn.CrossEntropyLoss()

print(f"Starting Training TransUNet Model on {DEVICE}...")
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
        
        # 梯度裁剪防爆
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
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'transunet_model.pth'))
print("TransUNet Model Saved.")