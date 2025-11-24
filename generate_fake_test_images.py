import torch
from torchvision.utils import save_image

print("Generating Dummy images")

for i in range(20):
    #Randomized images
    fake_img = torch.randn(3,32,32)
    real_img = torch.randn(3,32,32)

    save_image(fake_img,f'sham_tests_fake/img_{i}.png')
    save_image(real_img,f"sham_tests_real/img_{i}.png")

print("Done! Dummy data generated")