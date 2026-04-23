import torch
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Example script to use GPU")

# 添加参数
parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')

# 解析参数
args = parser.parse_args()

# 检查CUDA是否可用
args.use_gpu = args.use_gpu and torch.cuda.is_available()

# 打印是否使用GPU
print(f"Using GPU: {args.use_gpu}")

if args.use_gpu:
    # 指定设备
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
    print(f"Using device: {device}")

    # 获取当前设备
    current_device = torch.cuda.current_device()
    print(f"Current Device: {current_device}")

    # 获取设备名称
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Device Name: {device_name}")

    # 获取设备总数
    device_count = torch.cuda.device_count()
    print(f"Device Count: {device_count}")

    # 获取每个设备的详细信息
    for i in range(device_count):
        print(f"\nDevice {i} details:")
        print(f"  Device Name: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
        print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
        print(f"  Cached Memory: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
        print(f"  Max Memory: {torch.cuda.max_memory_allocated(i) / (1024 ** 3):.2f} GB")
else:
    print("No GPU available or use_gpu is False")