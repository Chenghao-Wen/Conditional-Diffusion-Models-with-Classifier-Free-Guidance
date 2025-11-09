import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

# --- 1. 关键的 import (来自你找到的正确文件) ---
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion

# --- 2. 加载真实的 CIFAR-10 数据 ---
# (这部分不变)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1) # 标准化到 [-1, 1]
])

dataset = CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

BATCH_SIZE = 16
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------

# --- 3. 定义 UNet ---
# (这部分不变, 引用正确的 Unet 类)
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    num_classes = 10,           
    cond_drop_prob = 0.1        
)

# --- 4. 修正：定义 GaussianDiffusion ---
# (已删除不存在的 loss_type='l1' 参数)
diffusion = GaussianDiffusion(
    model,
    image_size = 32,            # CIFAR-10 是 32x32
    timesteps = 1000
    # objective = 'pred_noise'  <-- 我们可以保留默认值
)

# --- 5. 模拟训练步骤 (使用真实数据) ---
# (这部分不变)
print("--- 开始训练循环 (使用真实 CIFAR-10 数据) ---")

try:
    real_images, real_classes = next(iter(dataloader))
except StopIteration:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    real_images, real_classes = next(iter(dataloader))

print(f"成功加载一批真实数据: images shape {real_images.shape}, classes shape {real_classes.shape}")

loss = diffusion(
    real_images,
    classes = real_classes
)

print(f"使用真实数据计算的训练损失: {loss.item()}")


# --- 6. 模拟采样/推理步骤 (Inference Step) ---
# (这部分不变)
print("\n--- 模拟采样 (Inference) ---")

num_samples = 4
target_class = 3
desired_classes = torch.full((num_samples,), target_class, dtype=torch.long)

sampled_images = diffusion.sample(
    classes = desired_classes,
    cond_scale = 7.5           
)

print(f"生成图像的 shape: {sampled_images.shape}")