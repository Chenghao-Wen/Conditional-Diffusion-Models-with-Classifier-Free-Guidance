import torch
import numpy as np

# Strategy 1: Dynamic Thresholding
# This prevents the "fried" high-contrast images when using high guidance scales
# Instead of a traditional clipping of [-1,1], we will clip to a percentile

def dynamic_thresholding(img, percentile = 0.995):
    magnitude = torch.quartile(torch.abs(img), percentile)

    # We will keep it at 1 if percentile is less than 1, will scale down if it is greatert than 1
    magnitude = torch.maximum(magnitude,torch.tensor(1.0, device = img.device))
    img = torch.clamp(img, -magnitude, magnitude) / magnitude
    return img

# Strategy 2: Dynamic Guidance Scheduling
# Adjust the guidance scale over time
# High guidance early, low guidance late

def get_guidance_scale(current_step, total_steps, max_guidance=3.0,method = 'linear'):
    normalized_time = current_step / total_steps

    if method == 'constant':
        return max_guidance
    elif method == 'linear':
        # This will be linear decay
        return max_guidance * (1 - normalized_time)
    elif method == 'cosine':
        return max_guidance * 0.5 * (1 + np.cos(normalized_time * np.pi))
    
    return max_guidance