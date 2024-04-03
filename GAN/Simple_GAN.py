import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# GAN works with two networks
# 1. Generator:
# It tries to generate an image from random noise
# 2. Discriminator:
# It tries to find whether the generator one is fake or not

# While training, if the generator generates an image so good
# that the discriminator thinks generated image is real and real image is fake
# Therefore, comes a perfect GAN


class Discriminator(nn.Module):
    def __init__(self, img_dim) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 150

disc = Discriminator(img_dim=img_dim).to(device)
gen = Generator(z_dim=z_dim, img_dim=img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="../dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/Simple_GAN/fake")
writer_real = SummaryWriter(f"runs/Simple_GAN/real")
step = 0

for epoch in range(num_epochs):
    # (images, labels)
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train discriminator
        # max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise) # G(z) generate fake images using some random noise
        disc_real = disc(real).view(-1) # D(real)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # max log(D(real))
        disc_fake = disc(fake).view(-1) # D(G(z))
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # log(1 - D(G(z)))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train generator
        # min log(1-D(G(z))) <-> max(log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image("MNIST fake images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST real images", img_grid_real, global_step=step)

                step += 1