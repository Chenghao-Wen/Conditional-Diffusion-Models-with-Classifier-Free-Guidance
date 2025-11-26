import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os

# ，
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion

# --- 1.  ---
BATCH_SIZE = 32          # ， 64  128
LEARNING_RATE = 1e-4     # 
EPOCHS = 10              #  (10， 50+)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 
os.makedirs('./results', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)

print(f"Using device: {DEVICE}")

# --- 2.  ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1) #  [-1, 1]
])

dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- 3.  ---
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
).to(DEVICE) # <--- ： GPU 

# --- 4.  ---
optimizer = optim.Adam(diffusion.parameters(), lr=LEARNING_RATE)

# --- 5.  ---
print("--- Training Start ---")

for epoch in range(EPOCHS):
    diffusion.train() # 
    epoch_loss = 0
    
    for step, (images, classes) in enumerate(dataloader):
        #  GPU
        images = images.to(DEVICE)
        classes = classes.to(DEVICE)
        
        optimizer.zero_grad() # 
        
        #  Loss ()
        loss = diffusion(images, classes=classes)
        
        # 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {step} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"==> Epoch {epoch+1} Finished. Average Loss: {avg_loss:.4f}")
    
    # --- 6.  Epoch  (Sampling) ---
    print(f"--- Sampling for Epoch {epoch+1} ---")
    diffusion.eval() # 
    
    # 
    num_samples = 16
    #  3 ()  5 ()
    target_classes = torch.randint(0, 10, (num_samples,)).to(DEVICE)
    
    with torch.no_grad():
        #  sample 
        sampled_images = diffusion.sample(
            classes = target_classes,
            cond_scale = 7.5,
            dynamic_method = "constant" # <--- 
        )
    
    # 
    # images  [-1, 1]，save_image ， [0,1]
    #  diffusion  [0, 1]， [-1, 1]  unnormalize。
    #  [0, 1]  ( unnormalize_to_zero_to_one)
    save_image(sampled_images, f'./results/sample_epoch_{epoch+1}.png', nrow=4)
    print(f"Image saved to ./results/sample_epoch_{epoch+1}.png")
    
    # 
    torch.save(diffusion.state_dict(), f'./checkpoints/model_epoch_{epoch+1}.pt')
    print(f"Model saved to ./checkpoints/model_epoch_{epoch+1}.pt\n")

print("--- Training Complete ---")