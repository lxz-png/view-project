import torch
print("CUDA是否可用：", torch.cuda.is_available())
print("显卡名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")
print(torch.version.cuda)