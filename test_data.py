from data_utils import get_loader


dataset_path = "./dataset/train"

train_loader = get_loader(root=dataset_path, batch=2)


lr_imgs, hr_imgs = next(iter(train_loader))

print("===== 数据维度测试结果 =====")
print(f"退化低清图(lr)形状 [batch, C, H, W]：{lr_imgs.shape}")
print(f"高清原图(hr)形状 [batch, C, H, W]：{hr_imgs.shape}")
