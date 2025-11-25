import torch
from torch_fidelity import calculate_metrics
import argparse
import os

def evaluate_performance(real_path, fake_path):
    """
    This is the method that is used to calculate the FID and IS score 
    for generative models
    """
    print(f"---Starting Evaluation---")
    print(f"Real Images: {real_path}")
    print(f"Generated Images: {fake_path}")
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print("Error: Real or Fake image path does not exisst, please check directory")
        return;
    metrics = calculate_metrics(
        input1 = real_path,
        input2 = fake_path,
        cuda = torch.cuda.is_available(),
        isc=True,
        fid = True,
        verbose = False,
    )

    print(f"\n === Final Results ===")
    print(f"FID Score: {metrics['frechet_inception_distance']:.4f} (Lower is better)")
    print(f"Inception Score: {metrics['inception_score_mean']:.4f} (Higher is better)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type = str, required = True,help="Path to real test set images")
    parser.add_argument("--fake", type = str, required=True, help = "Path to model generated images")
    args = parser.parse_args()

    evaluate_performance(args.real, args.fake)
