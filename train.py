import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms.functional import rgb_to_grayscale

from model import ColouriserUNet

BATCH_SIZE = 128
EPOCHS = 15           # number of full passes through the training data
LEARNING_RATE = 1e-3  # how big each step is when adjusting weights to minimise loss (0.001)
SEED = 42


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """
    One full pass through the data.
    If an optimizer is given, the model trains (updates weights); otherwise it only measures loss.
    Returns the average loss per image.
    """

    training = optimizer is not None  # if no optimizer, we're validating (not training)
    model.train(training)  # train mode vs eval mode (BatchNorm behaves differently in each)
    total_loss = 0.0

    # Only track gradients when training (saves memory and time during validation)
    with torch.set_grad_enabled(training):
        for colour_img_batch, _ in loader:

            colour_img_batch = colour_img_batch.to(device)  # move to GPU if available

            grey_img_batch = rgb_to_grayscale(colour_img_batch)  # convert to greyscale

            pred = model(grey_img_batch)  # run the model to get its colourisation
            loss = loss_fn(pred, colour_img_batch)  # compare prediction to the original colour image

            if training:
                optimizer.zero_grad()        # clear old gradients
                loss.backward()              # work out which direction to move weights to reduce loss
                optimizer.step()             # update weights

            # loss is this batch's average (from MSELoss); multiply by batch size to get the batch's total,
            # so the final epoch average is accurate even though the last batch is smaller.
            # e.g. batches of 128 (avg 0.02) and 72 (avg 0.04): averaging the averages gives 0.03,
            # but the true average is (0.02*128 + 0.04*72) / 200 = 0.027
            total_loss += loss.item() * colour_img_batch.size(0)

    # total loss / number of images = true average loss per image for this epoch
    return total_loss / len(loader.dataset)

def main():
    torch.manual_seed(SEED)  # fix randomness so results are repeatable
    device = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available
    print(f"Using device: {device}")

    # ----- DATA LOADING -----

    # 1. Load the dataset (CIFAR-10) - downloads to data/ the first time
    transform = transforms.ToTensor() # to convert image to tensor (grid of numbers)
    train_full = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform) # downloads/loads the 50,000 training images, convering with transform

    # 2. Split dataset into training and validation sets
    train_size = int(0.8 * len(train_full)) # 80& of the dataset for training
    train_set, val_set = random_split(train_full, [train_size, len(train_full) - train_size], generator=torch.Generator().manual_seed(SEED))

    # 3. Create DataLoaders for training and validation sets
        # DataLoader: tool that serves data to model in batches, shuffles each epoch, hands one batch at a time in a loop
        # epoch: one full pass through the training dataset

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # shuffle=True to randomize order of samples each epoch
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) # no need to shuffle validation set

    # --- Model, loss, optimizer ---

    model = ColouriserUNet().to(device)  # create the model and move it to the GPU
    loss_fn = nn.MSELoss()               # average of (predicted pixel - real pixel)^2 (for the batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # updates the weights

    # Create folders for saved models and plots (exist_ok=True: no error if they already exist)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # --- Training Loop ---
    train_losses, val_losses = [], []  # record each epoch's loss for the plot later
    best_val = float("inf")            # best validation loss so far (starts at infinity)


    # --- Training loop ---
    for epoch in range(1, EPOCHS + 1):
        # Training: updates the weights and measures loss on images it's learning from
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)

        # Validation: no updates, just measures loss on unseen images (the honest check)
        val_loss = run_epoch(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save the model whenever it does best on the validation set
        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            saved = "  (saved best model)"

        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {train_loss:.5f} | val loss {val_loss:.5f}{saved}")


    # --- Plot loss curves ---
    # Train and validation loss per epoch: both should go down; if validation rises
    # while train keeps falling, the model is overfitting
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.savefig("images/loss_curve.png", dpi=150)  # dpi=150: higher resolution image
    print("Saved images/loss_curve.png")
    
if __name__ == "__main__":
    main()