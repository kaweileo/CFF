import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm  # 导入 tqdm 库用于显示进度

# 定义输入文件路径和输出文件路径
input_file_path = 'weather_descriptions_all.txt'
output_file_path = 'weather_data_gpt_last_pooling_embeddings.npy'

# 加载预训练的GPT模型和分词器
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# 为 GPT-2 分词器设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 添加 [PAD] 标记
    model.resize_token_embeddings(len(tokenizer))  # 更新模型的词汇表大小

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 定义一个函数来读取文本文件并返回描述列表
def read_descriptions_from_file(file_path):
    descriptions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉多余的空行
            if line.strip():
                descriptions.append(line.strip())
    return descriptions


# 读取天气描述
print(f"读取文件 {input_file_path}...")
descriptions = read_descriptions_from_file(input_file_path)

# 定义批量大小和最大长度
batch_size = 32  # 根据硬件资源调整批量大小
max_length = 512  # 设置最大长度

# 初始化用于存储所有特征向量的数组
all_embeddings = None

# 分批处理数据，并显示进度条
print("开始分词、提取特征并向量化...")
for i in tqdm(range(0, len(descriptions), batch_size), desc="处理进度"):
    # 获取当前批次的描述
    batch_descriptions = descriptions[i:i + batch_size]

    # 分词和转换为输入ID
    inputs = tokenizer(batch_descriptions, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    # 将输入移动到 GPU（如果适用）
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 将输入传递给GPT模型
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state

    # 使用所有标记的平均嵌入作为句子的特征向量 *******1
    # mean_pooling_embeddings = torch.mean(last_hidden_states, dim=1)
    # 使用最后一个标记的嵌入作为句子的特征向量 *******2
    last_token_embeddings = last_hidden_states[:, -1, :]  # 取每个序列的最后一个标记的嵌入

    # 将特征向量转换为NumPy数组
    pooling_embeddings_np = last_token_embeddings.cpu().numpy()  # 移回 CPU 并转换为 NumPy 数组

    # 将当前批次的特征向量追加到总结果中
    if all_embeddings is None:
        all_embeddings = pooling_embeddings_np
    else:
        all_embeddings = np.vstack((all_embeddings, pooling_embeddings_np))

    # 清理不再需要的变量以释放内存
    del inputs, outputs, last_hidden_states, pooling_embeddings_np  # 修正变量名
    torch.cuda.empty_cache()  # 如果使用 GPU，清理显存

# 打印特征向量的形状
print("pooling embeddings shape:", all_embeddings.shape)

# 保存特征向量到文件
np.save(output_file_path, all_embeddings)
print(f"特征提取完成，已保存到文件 {output_file_path}。")