import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    """
    A reusable group of layers, inspired by the U-Net architecture: two rounds of
    (find patterns with filters -> rescale the scores -> keep only the patterns that were found)

    in_channels: number of input channels (e.g., 1 for a greyscale image, 3 for RGB images)
    out_channels: number of output channels (number of filters)
    """
    return nn.Sequential(  # run the layers below in order
        # Slide out_channels 3x3 filters over the input; padding=1 adds a border so the size stays the same
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        
        # Rescale each filter's outputs to a steady range, since they drift as weights change during training
        nn.BatchNorm2d(out_channels),
        
        # Keep only positive scores (pattern found); set negatives (not found) to 0
        # (inplace=True overwrites values instead of copying, saving memory)
        nn.ReLU(inplace=True),
        
        # Second round: same again, taking the first round's out_channels feature maps as input
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ColouriserUNet(nn.Module):
    """
    Creates a U-Net model: the FUNCTION f(greyscale_image) -> colour_image.
    nn.Module: PyTorch's base class for all models, gives inbuilt features like moving to GPU
    """

    def __init__(self):
        """
        Runs once when model is created, sets up the layers
        """
        super().__init__() # runs the __init__() method of nn.module, which sets up the model's internal machinery
        self.pool = nn.MaxPool2d(2)  # halves the image size

        # Encoder layers
        # conv_block(in, out) = "in" feature maps come in, "out" feature maps go out
        # 1 = greyscale input; 64/128/256 = number of filters (our choice, from the original U-Net)
        # Filters double each time the image halves: deeper layers find more complex patterns
        self.enc1 = conv_block(1, 64)     # 32x32, 1 -> 64 channels
        self.enc2 = conv_block(64, 128)   # 16x16, 64 -> 128 channels
        self.enc3 = conv_block(128, 256)  # 8x8, 128 -> 256 channels

        # Decoder layers: ConvTranspose2d doubles the image size (the opposite of pooling)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec2 = conv_block(256, 128)  # 128 upsampled + 128 from e2 (skip) = 256 in
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 16x16 -> 32x32
        self.dec1 = conv_block(128, 64)   # 64 upsampled + 64 from e1 (skip) = 128 in

        # Final layer: turn 64 feature maps into 3 colour channels (R, G, B)
        self.out = nn.Conv2d(64, 3, kernel_size=1) # at each pixel (one box in the image grid of 32x32 = 1024), combine its 64 feature-map values into 3 colours (R, G, B)


    def forward(self, x):
        """
        Runs for each batch (a group of images fed in at once): defines the route
        every batch takes through the model's stages (encoder blocks enc1 -> enc2 -> enc3).    
            Between encoder blocks, the image is halved (via pooling) and the number of
            channels doubles (more filters to find more complex patterns).
        """
        # x: batch of greyscale images, [batch, 1, 32, 32] (32x32 = CIFAR-10 image size)
        
        # Encoder: shrinks the image step by step, keeping only the strongest signals
        # (max pooling). After each shrink, a 3x3 filter covers features from a larger
        # area of the original image, so deeper layers can recognise objects, not just edges.

        e1 = self.enc1(x)              # [batch, 64, 32, 32]
        e2 = self.enc2(self.pool(e1))  # pool to 16x16, then -> [batch, 128, 16, 16]
        e3 = self.enc3(self.pool(e2))  # pool to 8x8, then -> [batch, 256, 8, 8]
        # Each block replaces the previous maps rather than adding to them,
        # so e3 has 256 channels (not 64 + 128 + 256).
        # e1, e2 are kept separately: the decoder will reuse them for skip connections
        
        
        # Decoder: grows the image back to full size (upsampling), turning that understanding
        # into colours. At each size, it joins in the encoder's maps from that same size
        # (e2 at 16x16, e1 at 32x32) via skip connections, to recover sharp details lost during shrinking
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))  # [batch, 128, 16, 16]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [batch, 64, 32, 32]

        # Sigmoid squashes outputs to 0-1, matching real pixel values
        return torch.sigmoid(self.out(d1))  # [batch, 3, 32, 32]



if __name__ == "__main__":
    # Quick test: pass a fake batch of 4 greyscale images through the full model
    model = ColouriserUNet()
    fake = torch.rand(4, 1, 32, 32)

    # batch size 4, 3 colour channels, back to full 32x32 size
    # therefore expected shape is [4, 3, 32, 32]
    print("Model output:", model(fake).shape)

    print("Parameters:", sum(p.numel() for p in model.parameters()))