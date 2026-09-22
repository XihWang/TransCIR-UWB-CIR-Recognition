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

# LSTM-Transformer 参数配置
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

# ================= 【新架构】LSTM + Transformer (无压缩) =================
class RadarLSTMTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=2):
        super(RadarLSTMTransformer, self).__init__()
        
        # --- 1. Encoder (使用双向 LSTM 替代 CNN) ---
        # 输入特征数为 4 (v0, v1, amp, phase)
        # 双向 LSTM: 隐藏层设为 d_model // 2，这样正反向拼接后输出维度正好是 d_model (64)
        self.lstm = nn.LSTM(
            input_size=4, 
            hidden_size=d_model // 2, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # --- 2. Bottleneck (Transformer 长度直接为 40) ---
        self.pos_encoder = nn.Parameter(torch.randn(1, 40, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. 没有 Decoder，直接分类头 ---
        # 此时特征张量大小为 [B, 40, d_model]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 40, 128), # 64通道 * 40 bins
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [B, 40, 4] -> LSTM 原生支持这种格式，无需像 CNN 那样 permute
        
        # === LSTM 编码阶段 ===
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [B, 40, 64]
        
        # === Transformer 瓶颈阶段 ===
        t_out = self.transformer(lstm_out + self.pos_encoder) # [B, 40, 64]
        
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

model = RadarLSTMTransformer(num_classes=encoder.num_classes).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
criterion = nn.CrossEntropyLoss()

print(f"Starting Training Model (LSTM+Transformer 40-40) on {DEVICE}...")
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
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'lstm_transformer_model_4040.pth'))
print("LSTM-Transformer Model (40-40) Saved.")