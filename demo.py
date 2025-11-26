import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1) 
])

dataset = CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

BATCH_SIZE = 16
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    num_classes = 10,           
    cond_drop_prob = 0.1        
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,          
    timesteps = 1000

)

print("--- Training Start ---")

try:
    real_images, real_classes = next(iter(dataloader))
except StopIteration:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    real_images, real_classes = next(iter(dataloader))

print(f"Data loaded: images shape {real_images.shape}, classes shape {real_classes.shape}")

loss = diffusion(
    real_images,
    classes = real_classes
)

print(f"Training Loss: {loss.item()}")


# --- 6. / (Inference Step) ---
# ()
print("\n--- (Inference) ---")

num_samples = 4
target_class = 3
desired_classes = torch.full((num_samples,), target_class, dtype=torch.long)

sampled_images = diffusion.sample(
    classes = desired_classes,
    cond_scale = 7.5,
    dynamic_method = "constant"   
)

print(f" Shape of Generated Images: {sampled_images.shape}")