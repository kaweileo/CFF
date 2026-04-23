import torch.nn as nn
import torch
# 修改模型定义
class WeatherModel(nn.Module):
    def __init__(self, input_dim, feature_dim, num_layers=16):
        super(WeatherModel, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # # 使用单层线性网络输出调整因子
        # self.mlp = nn.Linear(input_dim, 2 * feature_dim)

         # 动态构建 MLP 层
        layers = []
        # hidden_dim = input_dim // 2  # 隐藏层维度固定为 input_dim // 2
        hidden_dim = 256  # 隐藏层维度固定为 input_dim // 2

        # 添加中间层
        for i in range(num_layers - 1):  # 根据 num_layers 构建中间层
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))  # 第一层输入维度为 input_dim
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))  # 后续层输入输出维度均为 hidden_dim
            layers.append(nn.LeakyReLU())  # 每层后添加 ReLU 激活函数


        # 最后一层输出维度为 2 * feature_dim
        layers.append(nn.Linear(hidden_dim, 2 * feature_dim))

        # 将所有层封装到 Sequential 中
        self.mlp = nn.Sequential(*layers)

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_gpt, input_patchtst):
        # MLP 层
        mlp_output = self.mlp(input_gpt)

        # 分离权重和偏置，并应用激活函数约束范围
        weights = torch.sigmoid(mlp_output[:, :self.feature_dim]) * 2  # 权重范围 [0, 2]
        bias = torch.tanh(mlp_output[:, self.feature_dim:]) * 0.1  # 偏置范围 [-0.1, 0.1]
        # 加权操作
        weighted_patchtst = input_patchtst * weights
        final_output = weighted_patchtst + bias
        return final_output