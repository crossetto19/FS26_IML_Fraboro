import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import random
from tqdm import tqdm
from torchvision.transforms import functional as TF


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UNetInpainter(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)                     # 28 -> 14

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)                     # 14 -> 7

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )

        # Decoder
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)          # 64, 28, 28
        p1 = self.pool1(e1)        # 64, 14, 14

        e2 = self.enc2(p1)         # 128, 14, 14
        p2 = self.pool2(e2)        # 128, 7, 7

        b = self.bottleneck(p2)    # 256, 7, 7

        d2 = self.up2(b)           # 128, 14, 14
        d2 = torch.cat([d2, e2], dim=1)   # 256, 14, 14
        d2 = self.dec2(d2)         # 128, 14, 14

        d1 = self.up1(d2)          # 64, 28, 28
        d1 = torch.cat([d1, e1], dim=1)   # 128, 28, 28
        out = self.dec1(d1)        # 1, 28, 28
        return out
    

# -------------------------------
# Your original load_data (unchanged)
# -------------------------------
def load_data():
    train_data = np.load("train_data.npz")["data"]
    train_data = torch.tensor(train_data, dtype=torch.float32) / 255.0

    test_data_input = np.load("test_data.npz")["data"]
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0

    train_labels = train_data.clone()
    train_inputs = train_data.clone()
    train_inputs[:, :, 10:18, 10:18] = 0.0

    return train_inputs, train_labels, test_data_input

# -------------------------------
# Dataset with optional augmentation
# -------------------------------
class AugmentedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, augment=True):
        self.inputs = inputs      # already masked (centre zero)
        self.labels = labels      # clean images
        self.augment = augment
        # Define a simple affine transform (rotation + translation)
        self.affine = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img_clean = self.labels[idx]          # (1,28,28)
        img_masked = self.inputs[idx]         # already has centre masked

        if self.augment:
            # Apply same random transform to both clean and masked
            # We need to convert to PIL, apply, and back
            # Simpler: use torchvision's functional on tensors
            from torchvision.transforms import functional as TF
            # Random parameters
            angle = random.uniform(-10, 10)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            scale = random.uniform(0.9, 1.1)
            # Apply to clean
            img_clean = TF.affine(img_clean, angle=angle, translate=translate, scale=scale, shear=0)
            # Apply same to masked (but careful: after affine, the centre mask moves)
            # So we need to re-mask the centre after transformation.
            img_masked = TF.affine(img_masked, angle=angle, translate=translate, scale=scale, shear=0)
            # Remask centre (8x8)
            img_masked[:, 10:18, 10:18] = 0.0
        return img_masked, img_clean

# -------------------------------
# Training function (using your load_data)
# -------------------------------
def training(train_inputs, train_labels):
    # 1. Split the raw data first
    train_idx = int(0.9 * len(train_inputs))

    # Raw data split
    x_train, x_val = train_inputs[:train_idx], train_inputs[train_idx:]
    y_train, y_val = train_labels[:train_idx], train_labels[train_idx:]

    # 2. Create two separate dataset instances
    train_set = AugmentedMNISTDataset(x_train, y_train, augment=True)
    val_set = AugmentedMNISTDataset(x_val, y_val, augment=False)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

    model = UNetInpainter().to(device)   # use the improved U-Net from previous answer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    n_epochs = 100

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out[:, :, 10:18, 10:18], y[:, :, 10:18, 10:18])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                out_v = model(x_v)
                val_loss += criterion(out_v[:, :, 10:18, 10:18], y_v[:, :, 10:18, 10:18]).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

    model.load_state_dict(torch.load("best_model.pth"))
    return model

def testing(model, test_input):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(test_input), 64):
            batch = test_input[i:i+64].to(device)
            out = model(batch)                     # full predicted image [0,1]
            outputs.append(out.cpu())
    pred_full = torch.cat(outputs).numpy()         # shape (N,1,28,28) in [0,1]

    # Ground truth unmasked pixels come from the original test input (which is masked only in the center)
    test_input_np = test_input.cpu().numpy()       # also [0,1]
    submission = test_input_np.copy()
    submission[:, :, 10:18, 10:18] = pred_full[:, :, 10:18, 10:18]

    submission = (submission * 255).astype(np.uint8)
    np.savez_compressed("submit_this_test_data_output.npz", data=submission)

# ------------------------------
# Main
# -------------------------------
def main():

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    train_inputs, train_labels, test_input = load_data()
    model = training(train_inputs, train_labels)
    testing(model, test_input)   # use the fixed testing function from previous answer


if __name__ == "__main__":
    main()

