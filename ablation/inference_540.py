import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import json

# ================= 配置 =================
WORK_DIR = './model_output'
DATA_PATH = os.path.join(WORK_DIR, 'radar_data_processed.pkl')
# 【修改点 1】修改输出结果的文件名
OUTPUT_FILE = os.path.join(WORK_DIR, 'inference_results_540.pkl')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 【消融实验】同步 Bottleneck=5 结构 =================
class RadarTransUNet1D(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=2):
        super(RadarTransUNet1D, self).__init__()
        
        self.enc1 = nn.Sequential(nn.Conv1d(4, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        self.pool1 = nn.MaxPool1d(2) 
        
        self.enc2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.pool2 = nn.MaxPool1d(2) 
        
        self.enc3 = nn.Sequential(nn.Conv1d(64, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ReLU())
        self.pool3 = nn.MaxPool1d(2) 
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 5, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.up0 = nn.ConvTranspose1d(d_model, 64, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv1d(d_model + 64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2) 
        self.dec1 = nn.Sequential(nn.Conv1d(64 + 32, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        
        self.up2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2) 
        self.dec2 = nn.Sequential(nn.Conv1d(32 + 16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 40, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        e1 = self.enc1(x)       
        p1 = self.pool1(e1)     
        e2 = self.enc2(p1)      
        p2 = self.pool2(e2)     
        e3 = self.enc3(p2)      
        p3 = self.pool3(e3)     
        
        t_in = p3.permute(0, 2, 1)
        t_out = self.transformer(t_in + self.pos_encoder) 
        t_out = t_out.permute(0, 2, 1) 
        
        d0 = self.up0(t_out)               
        c0 = torch.cat([e3, d0], dim=1)    
        d0_out = self.dec0(c0)             
        
        d1 = self.up1(d0_out)              
        c1 = torch.cat([e2, d1], dim=1)    
        d1_out = self.dec1(c1)             
        
        d2 = self.up2(d1_out)              
        c2 = torch.cat([e1, d2], dim=1)    
        d2_out = self.dec2(c2)             
        
        return self.classifier(d2_out)

class LabelDecoder:
    def __init__(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.id_to_vec = data['id_to_vec']
            self.num_classes = len(self.id_to_vec)
    
    def decode(self, class_ids):
        vecs = []
        for cid in class_ids:
            vecs.append(self.id_to_vec[str(cid)])
        return np.array(vecs)

print("Loading label mapping...")
decoder = LabelDecoder(os.path.join(WORK_DIR, 'label_encoder.json'))

print("Loading TransUNet model (5-40)...")
model = RadarTransUNet1D(num_classes=decoder.num_classes).to(DEVICE)
# 【修改点 2】注意加载的是 5-40 专属的权重名
model.load_state_dict(torch.load(os.path.join(WORK_DIR, 'transunet_model_540.pth'), map_location=DEVICE))
model.eval()

print("Loading test data...")
with open(DATA_PATH, 'rb') as f: 
    data = pickle.load(f)
X_test = torch.FloatTensor(data['X_test']).to(DEVICE)
y_test = data['y_test']

print("Running Inference...")
all_preds_ids = []
BATCH_SIZE = 1024

with torch.no_grad():
    for i in range(0, len(X_test), BATCH_SIZE):
        batch_X = X_test[i:i+BATCH_SIZE]
        logits = model(batch_X)
        preds = torch.argmax(logits, dim=1)
        all_preds_ids.append(preds.cpu().numpy())

y_pred_ids = np.concatenate(all_preds_ids)
y_pred_vec = decoder.decode(y_pred_ids)

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'y_test': y_test, 'y_pred': y_pred_vec, 'y_pred_ids': y_pred_ids}, f)

print(f"Done. Saved to {OUTPUT_FILE}")