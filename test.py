import torch
 
# 1. 打印PyTorch版本和CUDA版本信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本（编译时）: {torch.version.cuda}")
 
# 2. 检查CUDA（即GPU）是否可用（这是最关键的判断）
print(f"CUDA是否可用: {torch.cuda.is_available()}")
 
# 3. 如果可用，查看GPU设备信息
if torch.cuda.is_available():
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU设备: {torch.cuda.current_device()}")
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA不可用。当前安装的可能是CPU版本的PyTorch，或者显卡驱动/CUDA Toolkit未正确安装。")
