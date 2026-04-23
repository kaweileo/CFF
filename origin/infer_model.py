import torch
import torch.nn as nn
import numpy as np
from model_test import WeatherModel  # 从 model.py 中导入模型
import logging  # 导入 logging 模块

# 配置日志记录器
def configure_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 将日志写入文件
            logging.StreamHandler()         # 同时输出到控制台
        ]
    )
    return logging.getLogger()

# 推理函数
def evaluate_model(model, test_gpt_vectors, test_patchtst_predictions, true_data, device):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 关闭梯度计算以节省内存
        # 将数据移动到指定设备
        test_gpt_vectors = test_gpt_vectors.to(device)
        test_patchtst_predictions = test_patchtst_predictions.to(device)
        true_data = true_data.to(device)

        # 前向传播
        improved_predictions = model(test_gpt_vectors, test_patchtst_predictions)

        # 转换为 numpy 用于评估
        pred_np = improved_predictions.cpu().numpy()
        true_np = true_data.cpu().numpy()

        # 计算各项评估指标（使用 numpy）
        mse_loss = np.mean((pred_np - true_np) ** 2)  # MSE
        rmse_loss = np.sqrt(mse_loss)  # RMSE
        mae_loss = np.mean(np.abs(pred_np - true_np))  # MAE

        # MAPE (平均绝对百分比误差)
        epsilon = 1e-12  # 防止除零
        mape_loss = np.mean(np.abs((true_np - pred_np) / (true_np + epsilon)))

        # MSPE (平均平方百分比误差)
        mspe_loss = np.mean(((true_np - pred_np) / (true_np + epsilon)) ** 2)

        # RSE (Relative Squared Error) - 新方式
        def RSE(pred, true):
            sse = np.sum((true - pred) ** 2)
            sst = np.sum((true - true.mean()) ** 2)
            return np.sqrt(sse) / np.sqrt(sst)

        rse_loss = RSE(pred_np, true_np)

        return (
            improved_predictions.cpu(),  # 将结果移回 CPU
            mse_loss,
            rmse_loss,
            mae_loss,
            mape_loss,
            mspe_loss,
            rse_loss
        )

# 数据预处理函数
def preprocess_data(gpt_embeddings, patchtst_preds, true_data):
    # 获取形状信息
    B, T, F = patchtst_preds.shape

    # 处理 GPT 嵌入向量：对所有记录求平均值并扩展到 (T, GPT_DIM)
    global_gpt_vector = gpt_embeddings.mean(dim=0)  # 对所有记录求平均值
    expanded_gpt_vectors = global_gpt_vector.unsqueeze(0).repeat(T, 1)  # 扩展到 (T, GPT_DIM)

    # 归一化 GPT 嵌入向量
    expanded_gpt_vectors = (expanded_gpt_vectors - expanded_gpt_vectors.mean()) / (expanded_gpt_vectors.std() + 1e-8)

    # 展平前两维 (B1, B2)
    patchtst_preds = patchtst_preds.view(-1, T, F)  # 展平为 (B, T, F)
    true_data = true_data.view(-1, T, F)  # 展平为 (B, T, F)

    # 返回处理后的数据
    return expanded_gpt_vectors, patchtst_preds, true_data

# 加载数据
gpt_embeddings = torch.from_numpy(np.load('./data/weather/100%weather_bert.npy'))  # (52696, 768)
patchtst_predictions = torch.from_numpy(np.load('./data/weather/100%weather_pred.npy'))  # (76, 128, 720, 21)
true_weather_data = torch.from_numpy(np.load('./data/weather/100%weather_true.npy'))  # (76, 128, 720, 21)

# 数据预处理
expanded_gpt_vectors, patchtst_predictions, true_weather_data = preprocess_data(
    gpt_embeddings, patchtst_predictions, true_weather_data
)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型
gpt_input_dim = expanded_gpt_vectors.shape[-1]  # GPT 嵌入向量维度
feature_dim = patchtst_predictions.shape[-1]  # 特征维度
model = WeatherModel(input_dim=gpt_input_dim, feature_dim=feature_dim)
model.load_state_dict(torch.load('./modelfiles/100%weather_bert.pth', map_location=device))  # 加载保存的参数
model.to(device)  # 将模型移动到 GPU

# 创建 DataLoader
batch_size = 512  # 设置较大的批次大小以充分利用 GPU
dataset = torch.utils.data.TensorDataset(expanded_gpt_vectors.repeat(patchtst_predictions.shape[0], 1),
                                         patchtst_predictions.view(-1, patchtst_predictions.shape[-1]),
                                         true_weather_data.view(-1, true_weather_data.shape[-1]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 初始化误差累积变量
total_mse = 0.0
total_rmse = 0.0
total_mae = 0.0
total_mape = 0.0
total_mspe = 0.0
total_rse = 0.0  # 新增 RSE 累积变量
num_batches = 0

# 用于保存预测值的列表
all_improved_predictions = []

# 配置日志记录器
log_file = './logs/100%weather_bert.txt'  # 日志文件路径
logger = configure_logger(log_file)

# 批量推理
for batch_gpt, batch_patchtst, batch_true in dataloader:
    # 模型推理
    (
        improved_predictions,
        mse,
        rmse,
        mae,
        mape,
        mspe,
        rse
    ) = evaluate_model(model, batch_gpt, batch_patchtst, batch_true, device)

    # 累加误差
    total_mse += mse
    total_rmse += rmse
    total_mae += mae
    total_mape += mape
    total_mspe += mspe
    total_rse += rse
    num_batches += 1

    # 累积预测值
    all_improved_predictions.append(improved_predictions)

    # 记录每批次的日志（新增 RSE）
    logger.info(f"Batch [{num_batches}/{len(dataloader)}] - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RSE: {rse:.4f}")

# 计算平均误差
avg_mse = total_mse / num_batches
avg_rmse = total_rmse / num_batches
avg_mae = total_mae / num_batches
avg_mape = total_mape / num_batches
avg_mspe = total_mspe / num_batches
avg_rse = total_rse / num_batches  # 可选：每批次平均 RSE

# 输出结果
print(f"Average MSE: {avg_mse:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average MAE: {avg_mae:.4f}")
# print(f"Total RSE: {total_rse:.4f}")
print(f"Average RSE per batch: {avg_rse:.4f}")

# 记录最终的评估结果到日志
logger.info(f"Final Results - Average MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, RSE: {total_rse:.4f}")

# 合并所有批次的预测值
all_improved_predictions_tensor = torch.cat(all_improved_predictions, dim=0)

# 将预测值保存为 .npy 文件
np.save('./CFF/CFFpred_100%weather_bert.npy', all_improved_predictions_tensor.numpy())

print(f"Improved predictions saved to './CFFpred-without_Warm-up/CFFpred-without_Warm-up_weather.npy'")