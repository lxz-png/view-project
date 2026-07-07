import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import get_loader
from model import SceneRestorePipeline


parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="./dataset/train")
parser.add_argument("--val_data", type=str, default="./dataset/val")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()


device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print("当前训练设备：", device)


model = SceneRestorePipeline().to(device)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


train_loader = get_loader(args.train_data, batch=args.batch_size)
val_loader = get_loader(args.val_data, batch=args.batch_size, shuffle=False)


train_loss_record = []
val_loss_record = []
best_val_loss = float("inf")


os.makedirs("weights", exist_ok=True)


for epoch in range(args.epoch):
    
    model.train()
    total_train_loss = 0
    train_bar = tqdm(train_loader, desc=f"训练轮次{epoch+1}/{args.epoch}")
    for lr_noise, hr_gt in train_bar:
        lr_noise, hr_gt = lr_noise.to(device), hr_gt.to(device)
        pred_hr = model(lr_noise)
        loss = loss_func(pred_hr, hr_gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_bar.set_postfix({"单步loss": loss.item()})
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_record.append(avg_train_loss)

    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for lr_noise, hr_gt in val_loader:
            lr_noise, hr_gt = lr_noise.to(device), hr_gt.to(device)
            pred_hr = model(lr_noise)
            loss = loss_func(pred_hr, hr_gt)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_record.append(avg_val_loss)
    print(f"第{epoch+1}轮 | 训练loss:{avg_train_loss:.6f} | 验证loss:{avg_val_loss:.6f}")

    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "weights/best_model.pth")
        print("已更新最优模型权重 best_model.pth")


plt.figure(figsize=(10,4))
plt.plot(train_loss_record, label="训练损失 Train Loss")
plt.plot(val_loss_record, label="验证损失 Val Loss")
plt.xlabel("训练轮 Epoch")
plt.ylabel("MSE损失值")
plt.title("2B自然风景照片修复 训练Loss变化曲线")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("loss_curve.png", dpi=300)
plt.close()
print("Loss曲线图片已保存 loss_curve.png")