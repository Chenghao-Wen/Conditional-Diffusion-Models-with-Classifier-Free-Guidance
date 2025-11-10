```
git clone https://github.com/Chenghao-Wen/Conditional-Diffusion-Models-with-Classifier-Free-Guidance.git
Conditional-Diffusion-Models-with-Classifier-Free-Guidance
conda env create -f environment.yml 
conda activate CFGO
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
cd denoising-diffusion-pytorch
pip install denoising_diffusion_pytorch
cd ..
python demo.py
```


