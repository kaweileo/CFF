import torch
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm

# =================== 配置参数 ===================
input_csv_path = './csvdata/weather.csv'      # 修改为你的 CSV 文件路径
output_npy_path = 'csv_weather.npy'

model_name = 'gpt2'                           # 可选：'gpt2-medium' 更强但更慢
batch_size = 16                               # 根据 GPU 显存调整（16~32）
max_length = 512                              # GPT-2 最大上下文长度
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ===============================================

# 加载预训练 GPT-2 模型和分词器
print("正在加载 GPT-2 模型...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.to(device)
model.eval()  # 推理模式

# 读取 CSV 文件
print(f"读取 CSV 文件: {input_csv_path}")
df = pd.read_csv(input_csv_path)

# 确保只有 22 列（去掉索引或多余列）
if df.shape[1] != 22:
    print(f"警告：检测到 {df.shape[1]} 列，将使用前 22 列。")
data_columns = df.columns[:22]

# 将每一行转换为“字段: 值”组成的文本描述
print("正在将结构化数据转换为文本描述...")
descriptions = []
for _, row in df.iterrows():
    # 构造文本："col1: val1, col2: val2, ..., col22: val22"
    parts = [f"{col}: {row[col]}" for col in data_columns]
    text = ", ".join(parts)
    descriptions.append(text)

print(f"共生成 {len(descriptions)} 条文本描述，每条基于 22 个指标。")

# 分批提取特征
all_embeddings = []
print("开始使用 GPT-2 提取特征（逐批处理）...")

for i in tqdm(range(0, len(descriptions), batch_size), desc="GPT-2 编码进度"):
    batch_texts = descriptions[i:i + batch_size]

    # 分词
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    # 移动到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)

    # 取最后一个 token 的隐藏状态作为句子表示
    last_hidden_states = outputs.last_hidden_state  # [B, seq_len, 768]
    last_token_embeds = last_hidden_states[:, -1, :]  # [B, 768]

    # 转为 NumPy 并移到 CPU
    last_token_embeds_cpu = last_token_embeds.cpu().numpy()
    all_embeddings.append(last_token_embeds_cpu)

    # 清理缓存
    del inputs, outputs, last_hidden_states, last_token_embeds
    torch.cuda.empty_cache()

# 合并所有批次
all_embeddings = np.vstack(all_embeddings)
print(f"最终 embedding 形状: {all_embeddings.shape}")  # 应为 [50000+, 768]

# 保存为 .npy 文件
print(f"正在保存到 {output_npy_path}...")
np.save(output_npy_path, all_embeddings)
print(f"✅ 特征提取完成，已保存至: {output_npy_path}")