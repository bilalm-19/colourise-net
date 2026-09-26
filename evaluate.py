import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.utils import save_image

from model import ColouriserUNet

NUM_EXAMPLES = 12  # how many images to show in the comparison


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the test set: 10,000 images the model has never seen (not used in training or validation)
    test_set = datasets.CIFAR10("data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=256)

    # Create the model and load the best saved weights from training
    model = ColouriserUNet().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()  # testing mode

    # --- Test score ---
    total_loss = 0.0
    with torch.no_grad():  # no weight updates, so no need to track gradients
        for colour_img_batch, _ in test_loader:
            colour_img_batch = colour_img_batch.to(device)
            pred = model(rgb_to_grayscale(colour_img_batch))
            # reduction="sum": add up every squared error instead of averaging per batch
            total_loss += F.mse_loss(pred, colour_img_batch, reduction="sum").item()

    # Average over every pixel value in the test set (images x 3 channels x 32 x 32)
    mse = total_loss / (len(test_set) * 3 * 32 * 32)
    
    # PSNR (peak signal-to-noise ratio): a standard image-quality score in decibels (higher = closer to the real image)
    psnr = 10 * math.log10(1.0 / mse)
    print(f"Test MSE: {mse:.5f} | Test PSNR: {psnr:.2f} dB")

    # --- Comparison image ---
    torch.manual_seed(0)  # pick the same random test images every run
    indices = torch.randint(len(test_set), (NUM_EXAMPLES,))
    colour_img_batch = torch.stack([test_set[i][0] for i in indices]).to(device)
    grey_img_batch = rgb_to_grayscale(colour_img_batch)

    with torch.no_grad():
        pred = model(grey_img_batch)

    # Stack three rows: greyscale input / model output / real image
    # (greyscale repeated to 3 channels so all rows have the same shape)
    rows = torch.cat([grey_img_batch.repeat(1, 3, 1, 1), pred, colour_img_batch])
    rows = F.interpolate(rows, scale_factor=2, mode="bilinear")  # smooth scaling instead of blocky
    os.makedirs("images", exist_ok=True)
    save_image(rows, "images/comparison.png", nrow=NUM_EXAMPLES, padding=4, pad_value=1.0)
    print("Saved images/comparison.png (rows: greyscale input, model output, real image)")


if __name__ == "__main__":
    main()