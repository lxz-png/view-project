import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def degrade_image(hr_image, noise_mode="gauss", sigma=20, down_scale=4):
    """
    hr_image: 256*256高清原图 numpy数组 0~255
    down_scale=4 → 256/4=64，实现64低分辨率输入
    return: 退化带噪低清图、原始高清图
    """
    h, w = hr_image.shape[:2]
    
    lr_small = cv2.resize(hr_image, (h//down_scale, w//down_scale), interpolation=1)
    
    lr_blur = cv2.resize(lr_small, (h, w), interpolation=1)

    
    if noise_mode == "gauss":
        gauss_noise = np.random.normal(0, sigma, lr_blur.shape).astype(np.int16)
        lr_noisy = np.clip(lr_blur.astype(np.int16)+gauss_noise, 0, 255).astype(np.uint8)
    elif noise_mode == "salt_pepper":
        prob = 0.04
        noise_mask = np.random.choice([0,1,2], size=lr_blur.shape[:2], p=[1-prob, prob/2, prob/2])
        lr_noisy = lr_blur.copy()
        lr_noisy[noise_mask==1] = 0
        lr_noisy[noise_mask==2] = 255
    else:
        lr_noisy = lr_blur
    return lr_noisy, hr_image


class SceneDataset(Dataset):
    def __init__(self, data_root, noise="gauss", noise_sigma=20, scale=4):
        self.root = data_root
        self.noise = noise
        self.sigma = noise_sigma
        self.scale = scale
        
        self.img_path_list = [os.path.join(data_root, f) for f in os.listdir(data_root)
                              if f.endswith(("jpg","png","jpeg","JPG","PNG"))]
        
        if len(self.img_path_list) == 0:
            raise FileNotFoundError(f"路径 {data_root} 下没有找到图片文件！")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        hr_pil = Image.open(img_path).convert("RGB")
        
        hr_pil = transforms.Resize((256, 256))(hr_pil)
        hr_np = np.array(hr_pil)
        
        lr_noisy_np, hr_np = degrade_image(hr_np, self.noise, self.sigma, self.scale)
        
        hr_tensor = self.to_tensor(hr_np)
        lr_tensor = self.to_tensor(lr_noisy_np)
        return lr_tensor, hr_tensor


def get_loader(root, batch=8, shuffle=True, noise="gauss", sigma=20, scale=4):
    dataset = SceneDataset(root, noise=noise, noise_sigma=sigma, scale=scale)
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0)
    return loader


def get_psnr_ssim(pred_tensor, gt_tensor):
    """输入模型输出预测图、真实高清图（tensor格式），返回数值"""
    
    pred = (pred_tensor.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    gt = (gt_tensor.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(gt, pred)
    ssim = structural_similarity(gt, pred, channel_axis=2)
    return round(psnr,2), round(ssim,4)
