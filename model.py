import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ===============================
# OUTPUT DIRECTORIES
# ===============================
output_dirs = {
    'plots': 'output/plots',
    'generated_images': 'output/generated_images',
    'comparison_images': 'output/comparison_images',
    'models': 'output/models'
}

for d in output_dirs.values():
    Path(d).mkdir(parents=True, exist_ok=True)

# ===============================
# DATASET
# ===============================
class SimpleDataset(Dataset):
    def __init__(self, image_dir, label_dir, max_images=100, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.files = sorted(os.listdir(image_dir))[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.files[idx])).convert("RGB")
        lbl = Image.open(os.path.join(self.label_dir, self.files[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
            lbl = self.transform(lbl)
        return lbl, img  # label → real image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = SimpleDataset("img", "label", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ===============================
# U-NET BLOCK
# ===============================
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, activation="relu", dropout=False):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        else:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]

        layers.append(nn.InstanceNorm2d(out_ch))

        if activation == "relu":
            layers.append(nn.ReLU(True))
        else:
            layers.append(nn.LeakyReLU(0.2, True))

        if dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# ===============================
# GENERATOR (DEEP U-NET)
# ===============================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = UNetBlock(3, 64, down=True, activation="leaky")
        self.d2 = UNetBlock(64, 128, down=True, activation="leaky")
        self.d3 = UNetBlock(128, 256, down=True, activation="leaky")
        self.d4 = UNetBlock(256, 512, down=True, activation="leaky")
        self.d5 = UNetBlock(512, 512, down=True, activation="leaky")

        self.u1 = UNetBlock(512, 512, down=False, dropout=True)
        self.u2 = UNetBlock(1024, 256, down=False)
        self.u3 = UNetBlock(512, 128, down=False)
        self.u4 = UNetBlock(256, 64, down=False)

        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], 1))
        u3 = self.u3(torch.cat([u2, d3], 1))
        u4 = self.u4(torch.cat([u3, d2], 1))

        return self.tanh(self.final(torch.cat([u4, d1], 1)))

# ===============================
# DISCRIMINATOR (PATCHGAN)
# ===============================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))

# ===============================
# INITIALIZATION
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# ===============================
# TRAINING
# ===============================
num_epochs = 50
lambda_L1 = 100

G_epoch, D_epoch, L1_epoch = [], [], []

for epoch in range(num_epochs):
    g_loss_sum, d_loss_sum, l1_sum = 0, 0, 0

    for labels, real in dataloader:
        labels, real = labels.to(device), real.to(device)

        # -------- Generator --------
        optimizer_G.zero_grad()
        fake_img = G(labels)
        pred_fake = D(labels, fake_img)
        valid = torch.ones_like(pred_fake, device=device)

        gan_loss = criterion_GAN(pred_fake, valid)
        l1_loss = criterion_L1(fake_img, real)
        g_loss = gan_loss + lambda_L1 * l1_loss
        g_loss.backward()
        optimizer_G.step()

        # -------- Discriminator --------
        optimizer_D.zero_grad()
        pred_real = D(labels, real)
        real_loss = criterion_GAN(pred_real, torch.ones_like(pred_real, device=device))

        pred_fake = D(labels, fake_img.detach())
        fake = torch.zeros_like(pred_fake, device=device)
        fake_loss = criterion_GAN(pred_fake, fake)

        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        g_loss_sum += g_loss.item()
        d_loss_sum += d_loss.item()
        l1_sum += l1_loss.item()

    G_epoch.append(g_loss_sum / len(dataloader))
    D_epoch.append(d_loss_sum / len(dataloader))
    L1_epoch.append(l1_sum / len(dataloader))

    print(f"Epoch {epoch+1:03}/{num_epochs} | "
          f"G: {G_epoch[-1]:.4f} | "
          f"D: {D_epoch[-1]:.4f} | "
          f"L1: {L1_epoch[-1]:.4f} | "
          f"Fake img min/max: {fake_img.min().item():.3f}/{fake_img.max().item():.3f}")

    if (epoch + 1) % 10 == 0:
        save_image(fake_img[:4],
                   f"{output_dirs['generated_images']}/epoch_{epoch+1}.png",
                   normalize=True)

# ===============================
# SAVE MODELS
# ===============================
torch.save(G.state_dict(), f"{output_dirs['models']}/generator.pth")
torch.save(D.state_dict(), f"{output_dirs['models']}/discriminator.pth")

# ===============================
# BAR PLOT – IMAGE QUALITY
# ===============================
psnr = 20 * np.log10(1.0 / np.mean(L1_epoch[-10:]))

plt.figure(figsize=(8, 6))
bars = plt.bar(["L1 Error ↓", "PSNR ↑"],
               [np.mean(L1_epoch[-10:]), psnr],
               color=["#e67e22", "#2ecc71"],
               edgecolor="black")

plt.title("Image Quality Metrics", fontweight="bold")
plt.grid(axis="y", alpha=0.3)

for b in bars:
    plt.text(b.get_x() + b.get_width()/2,
             b.get_height(),
             f"{b.get_height():.3f}",
             ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{output_dirs['plots']}/image_quality_bar.png", dpi=300)
plt.show()

print("\n==============================")
print("✅ TRAINING & PLOTS COMPLETED")
print("==============================")
