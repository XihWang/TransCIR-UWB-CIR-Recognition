import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import json

# ================= 配置 =================
WORK_DIR = './model_output'
DATA_PATH = os.path.join(WORK_DIR, 'radar_data_processed.pkl')
OUTPUT_FILE = os.path.join(WORK_DIR, 'inference_results_linear.pkl')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 【Encoder 消融实验】 同步 Linear 结构 =================
class RadarTransUNet1D(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=2):
        super(RadarTransUNet1D, self).__init__()
        
        self.linear_enc = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 40, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 40, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        e_out = self.linear_enc(x)  
        
        t_out = self.transformer(e_out + self.pos_encoder) 
        t_out = t_out.permute(0, 2, 1) 
        
        return self.classifier(t_out)

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

print("Loading TransUNet model (Linear)...")
model = RadarTransUNet1D(num_classes=decoder.num_classes).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(WORK_DIR, 'transunet_model_linear.pth'), map_location=DEVICE))
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