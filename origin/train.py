import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_test import WeatherModel  # 从 model.py 中导入模型

# 读取 .npy 文件
gpt_embeddings = torch.from_numpy(np.load('./data/weather/100%weather_bert.npy'))  # GPT 模型的输出
patchtst_predictions = torch.from_numpy(np.load('./data/weather/100%weather_pred.npy'))  # PatchTST 的预测结果
true_weather_data = torch.from_numpy(np.load('./data/weather/100%weather_true.npy'))  # 真实的历史数据

# 动态获取数据形状
gpt_input_dim = gpt_embeddings.shape[-1]  # GPT 嵌入向量的维度
feature_dim = patchtst_predictions.shape[-1]  # 特征维度（F）
time_steps = patchtst_predictions.shape[-2]  # 时间步数（T）

# 数据预处理函数
def preprocess_data(gpt_vectors, patchtst_preds, true_data):
    # 处理 GPT 嵌入向量
    global_gpt_vector = gpt_vectors.mean(dim=0)  # 对所有记录求平均值
    expanded_gpt_vectors = global_gpt_vector.unsqueeze(0).repeat(time_steps, 1)  # 扩展到 (T, GPT_DIM)

    # 展平前两维 (B1, B2)
    if len(patchtst_preds.shape) >= 4:  # 如果形状是 (B1, B2, T, F)
        patchtst_preds = patchtst_preds.view(-1, *patchtst_preds.shape[-2:])  # 展平为 (B, T, F)
    if len(true_data.shape) >= 4:  # 如果形状是 (B1, B2, T, F)
        true_data = true_data.view(-1, *true_data.shape[-2:])  # 展平为 (B, T, F)

    # 选择第一个样本 (B=0)，得到 (T, F)
    patchtst_preds = patchtst_preds[0]  # 形状变为 (720, 21)
    true_data = true_data[0]  # 形状变为 (720, 21)

    return expanded_gpt_vectors, patchtst_preds, true_data

# 调用预处理函数
expanded_gpt_vectors, patchtst_predictions, true_weather_data = preprocess_data(
    gpt_embeddings, patchtst_predictions, true_weather_data
)

# 初始化模型、损失函数和优化器
model = WeatherModel(input_dim=gpt_input_dim, feature_dim=feature_dim)
# criterion = nn.MSELoss()
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-5) #weather最佳lr=1e-15


print("gpt_input_dim的形状：",gpt_input_dim)
print("feature_dim的形状：",feature_dim)


# # 准备训练数据
X_train_gpt = expanded_gpt_vectors.view(-1, gpt_input_dim)  # 展平时间步 (T, GPT_DIM)
X_train_patchtst = patchtst_predictions.view(-1, feature_dim)  # 展平时间步 (T, F)
y_train = true_weather_data.view(-1, feature_dim)  # 展平时间步 (T, F)

# 创建 DataLoader
dataset = TensorDataset(X_train_gpt, X_train_patchtst, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练模型

for epoch in range(100):  # 训练  epoch
    model.train()
    running_loss = 0.0
    for batch_gpt, batch_patchtst, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_gpt, batch_patchtst)
        # loss = criterion(outputs, batch_y)

        # 复合损失函数
        loss_mse = criterion(outputs, batch_y)
        loss_mae = criterion(outputs, batch_y)
        loss = 0.7 * loss_mse + 0.3 * loss_mae
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/150], Loss:{running_loss/len(dataloader):.4f}")

# 保存模型
torch.save(model.state_dict(), './modelfiles/100%weather_bert.pth')  # 保存模型参数
print("Model saved successfully.")