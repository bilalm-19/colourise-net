import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms.functional import rgb_to_grayscale

BATCH_SIZE = 128 # hyperparam - defines num samples processed before updating weights
SEED = 42


def main():
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

    # 4. (Sanity check) Grab one batch and check its shape
    #    Expected: [128, 3, 32, 32] = [batch size, colour channels, height, width]
    #    Why: layers expect exact shapes, so a mismatch here would crash the model later
    images, labels = next(iter(train_loader))
    print("Colour batch:", images.shape)

    # 5. Convert to greyscale and check again
    gray = rgb_to_grayscale(images)
    print("Greyscale batch:", gray.shape)


if __name__ == "__main__":
    main()