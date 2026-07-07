import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import SceneRestorePipeline
from data_utils import degrade_image, get_psnr_ssim

parser = argparse.ArgumentParser()
parser.add_argument("--test_image", type=str, required=True, help="高清风景测试图路径")
parser.add_argument("--weight_path", type=str, default="weights/best_model.pth")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = SceneRestorePipeline().to(device)
model.load_state_dict(torch.load(args.weight_path, map_location=device))
model.eval()


hr_pil = Image.open(args.test_image).convert("RGB")
hr_np = np.array(hr_pil)
lr_noisy_np, hr_origin_np = degrade_image(hr_np, noise_mode="gauss", sigma=20)


to_tensor = transforms.ToTensor()
lr_input = to_tensor(lr_noisy_np).unsqueeze(0).to(device)
hr_gt_tensor = to_tensor(hr_origin_np).unsqueeze(0).to(device)


with torch.no_grad():
    pred_output = model(lr_input)


psnr_value, ssim_value = get_psnr_ssim(pred_output[0], hr_gt_tensor[0])
print(f"修复效果定量评估：")
print(f"PSNR峰值信噪比：{psnr_value} dB（越高越好）")
print(f"SSIM结构相似度：{ssim_value} （越接近1越好）")


pred_np = (pred_output[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
fig, axes = plt.subplots(1, 3, figsize=(18,6))

axes[0].imshow(hr_origin_np)
axes[0].set_title("原始高清风景图（Ground Truth）", fontsize=12)
axes[0].axis("off")

axes[1].imshow(lr_noisy_np)
axes[1].set_title("退化老照片（64低分辨率+高斯噪声）", fontsize=12)
axes[1].axis("off")

axes[2].imshow(pred_np)
axes[2].set_title(f"模型修复结果 PSNR={psnr_value} SSIM={ssim_value}", fontsize=12)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("result_compare.png", dpi=300)
plt.close()
print("可视化对比图已保存 result_compare.png")