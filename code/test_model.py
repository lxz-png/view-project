import torch
from model import SceneRestorePipeline


model = SceneRestorePipeline()


test_input = torch.randn(1, 3, 256, 256)


output = model(test_input)


print("输入尺寸：", test_input.shape)
print("修复输出尺寸：", output.shape)


if test_input.shape == output.shape:
    print("✅ 网络结构验证通过，输入输出尺寸匹配要求 [1,3,256,256]")
else:
    print("❌ 尺寸不匹配，网络存在问题")