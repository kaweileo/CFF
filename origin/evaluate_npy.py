# 计算两个npy文件的MSE, MAE, RSE, RMSE值
import numpy as np
import argparse
import os

def calculate_mse(pred, true):
    return np.mean((pred - true) ** 2)

def calculate_mae(pred, true):
    return np.mean(np.abs(pred - true))

def calculate_rse(pred, true):
    sse = np.sum((pred - true) ** 2)
    sst = np.sum((true - true.mean()) ** 2)
    return np.sqrt(sse) / np.sqrt(sst + 1e-12)  # 防止除零

def calculate_rmse(pred, true):
    return np.sqrt(calculate_mse(pred, true))  # RMSE = sqrt(MSE)

def main():
    parser = argparse.ArgumentParser(description="计算两个 .npy 文件的 MSE、MAE、RSE、RMSE")
    
    parser.add_argument('--true', type=str, default='./data/illness/30%illness_true.npy')
    parser.add_argument('--pred', type=str, default='./CFF/CFFpred_30%illness.npy')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.true):
        raise FileNotFoundError(f"真实文件未找到: {args.true}")
    if not os.path.exists(args.pred):
        raise FileNotFoundError(f"预测文件未找到: {args.pred}")

    # 加载 .npy 文件
    true = np.load(args.true)
    pred = np.load(args.pred)

    # 【关键修改】将真实值的形状调整为与预测值一致
    pred_shape = pred.shape
    if len(true.shape) > 2:
        # 如果真实值是三维的，比如 (B, T, F)，则展平前两个维度
        true = true.reshape(-1, true.shape[-1])  # (B*T, F)
    # 再次检查形状
    if true.shape != pred_shape:
        raise ValueError(f"调整后形状仍不一致: {true.shape} vs {pred.shape}")

    # 计算指标
    mse = calculate_mse(pred, true)
    mae = calculate_mae(pred, true)
    rse = calculate_rse(pred, true)
    rmse = calculate_rmse(pred, true)  # 新增 RMSE

    # 输出结果
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RSE:  {rse:.6f}")
    print(f"RMSE: {rmse:.6f}")

if __name__ == '__main__':
    main()


#############################################################

# 计算形状相同的两个npy文件的MSE, MAE, RSE值
# import numpy as np
# import argparse
# import os

# def calculate_mse(pred, true):
#     return np.mean((pred - true) ** 2)

# def calculate_mae(pred, true):
#     return np.mean(np.abs(pred - true))

# def calculate_rse(pred, true):
#     sse = np.sum((pred - true) ** 2)
#     sst = np.sum((true - true.mean()) ** 2)
#     return np.sqrt(sse) / np.sqrt(sst + 1e-12)  # 防止除零

# def calculate_rmse(pred, true):
#     return np.sqrt(calculate_mse(pred, true))  # RMSE = sqrt(MSE)

# def main():
#     parser = argparse.ArgumentParser(description="计算两个 .npy 文件的 MSE、MAE、RSE、RMSE")
    
#     # 修改为你自己的默认路径，或者运行时传入参数
#     parser.add_argument('--true', type=str, default='./data/illness/30%illness_true.npy')
#     parser.add_argument('--pred', type=str, default='./data/illness/30%illness_pred.npy')

#     args = parser.parse_args()

#     # 检查文件是否存在
#     if not os.path.exists(args.true):
#         raise FileNotFoundError(f"真实文件未找到: {args.true}")
#     if not os.path.exists(args.pred):
#         raise FileNotFoundError(f"预测文件未找到: {args.pred}")

#     # 加载 .npy 文件
#     true = np.load(args.true)
#     pred = np.load(args.pred)

#     # 检查形状是否一致
#     if true.shape != pred.shape:
#         raise ValueError(f"真实值和预测值的形状不一致: {true.shape} vs {pred.shape}")

#     # 计算指标
#     mse = calculate_mse(pred, true)
#     mae = calculate_mae(pred, true)
#     rse = calculate_rse(pred, true)
#     rmse = calculate_rmse(pred, true)  # 新增 RMSE

#     # 输出结果
#     print(f"MSE:  {mse:.6f}")
#     print(f"MAE:  {mae:.6f}")
#     print(f"RSE:  {rse:.6f}")
#     print(f"RMSE: {rmse:.6f}")

# if __name__ == '__main__':
#     main()