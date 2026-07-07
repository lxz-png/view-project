[README.md](https://github.com/user-attachments/files/29735454/README.md)
\# 2B组 自然风景老照片修复系统

任务：基于Place365风景数据集实现老照片去噪+超分辨率增强

组员：XXX、XXX、XXX



\## 1. 环境配置

1\. Python版本：3.9\~3.11

2\. 安装依赖：

cd code

pip install -r requirements.txt

3\. GPU推荐：CUDA 11.8，无GPU自动切换CPU运行（速度较慢）



\## 2. 数据集准备

本项目使用Place365子集（自然风景类：山川、湖泊、森林、原野）

1\. 数据集下载地址：http://places2.csail.mit.edu/download.html

2\. 数据存放路径：code/dataset/Place365/

&#x20;   - train/ 训练高清原图

&#x20;   - val/ 验证高清原图

3\. 数据退化逻辑（data\_utils自动生成退化老照片）：

&#x20;  - 高斯噪声 σ=10/20/30

&#x20;  - 椒盐噪声 噪声密度0.02/0.05

&#x20;  - 下采样64×64→原图256×256，模拟低分辨率老照片



\## 3. 模型训练

\### 3.1 联合训练（去噪+超分端到端）

python train.py --dataset\_path ./dataset/Place365 --epoch 100 --batch\_size 8 --device cuda



参数说明：

\--epoch：训练轮数，风景图像建议80\~120轮

\--batch\_size：显存小于6G设为4

\--device：cpu/cuda



\## 4. 模型评估

python eval.py --test\_img ./test\_old\_photo.jpg --model\_weight ./weights/best.pth

输出：PSNR、SSIM数值，保存对比图到results/



\## 5. 交互式演示 demo.ipynb

jupyter notebook demo.ipynb

可上传本地老风景照片一键修复，可视化原图/退化图/修复图对比

