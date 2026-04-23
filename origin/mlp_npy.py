import os
import torch
import pandas as pd
import numpy as np
from torch import nn

# =================== 配置参数 ===================
input_csv_path = "./csvdata/exchange_rate.csv"                 # 输入 CSV
output_npy_path = "./mlp_npy_output/csv_exchange_rate.npy"   # 保存 mlp_output
batch_size = 4096      # 前向批大小
standardize = True     # 是否标准化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你的 WeatherModel
class WeatherModel(nn.Module):
    def __init__(self, input_dim, feature_dim, num_layers=16):
        super(WeatherModel, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        layers = []
        # hidden_dim = input_dim // 2 if input_dim >= 2 else 1
        hidden_dim = 256

        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_dim, 2 * feature_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_gpt):
        mlp_output = self.mlp(input_gpt)
        return mlp_output


# ========== 读取 CSV ==========
print(f"读取 CSV 文件: {input_csv_path}")
df = pd.read_csv(
    input_csv_path,
    na_values=["NA","N/A","na","--","-","?","None","none","null","NULL",""," "]
)

input_dim = df.shape[1]
feature_dim = input_dim  # 保持特征维度与列数一致

# 转换为数值
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 填充 NaN
df = df.fillna(df.mean(numeric_only=True))
df = df.fillna(0.0)

# 转 numpy
data = df.values.astype(np.float32)

# 标准化
if standardize:
    means = data.mean(axis=0, dtype=np.float64)
    stds = data.std(axis=0, dtype=np.float64, ddof=0)
    stds[stds == 0] = 1.0
    data = ((data - means) / stds).astype(np.float32)

# ========== 初始化模型 ==========
model = WeatherModel(input_dim, feature_dim, num_layers=12).to(device)
model.eval()

# ========== 前向计算 ==========
print("开始用 WeatherModel 提取 mlp_output 特征（分批）...")
mlp_output_list = []
with torch.no_grad():
    for start in range(0, data.shape[0], batch_size):
        end = min(start + batch_size, data.shape[0])
        batch = torch.from_numpy(data[start:end]).to(device)

        # 只取 mlp_output，不做加权
        mlp_out = model(batch).cpu().numpy()
        mlp_output_list.append(mlp_out)

mlp_output_all = np.vstack(mlp_output_list)

print(f"最终 mlp_output 形状: {mlp_output_all.shape}")  # [样本数, 2*feature_dim]

# ========== 保存 ==========
os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
np.save(output_npy_path, mlp_output_all)
print(f"✅ mlp_output 特征已保存至: {output_npy_path}")
