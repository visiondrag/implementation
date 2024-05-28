import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Simple Diffusion Model
class SimpleDiffusionModel(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(SimpleDiffusionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alpha_cumprod[t]
        return torch.sqrt(alpha_t) * x + torch.sqrt(1.0 - alpha_t) * noise

    def reverse(self, x, t):
        return self.model(x)

def loss_fn(model, x_0, t):
    x_t = model(x_0, t)
    noise_pred = model.reverse(x_t, t)
    return nn.MSELoss()(noise_pred, x_0)

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleDiffusionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
num_timesteps = model.num_timesteps

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, _ in trainloader:
        inputs = inputs.view(inputs.size(0), -1).to(device)
        t = torch.randint(0, num_timesteps, (inputs.size(0),), device=device)

        optimizer.zero_grad()
        loss = loss_fn(model, inputs, t)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(trainloader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

print('Training complete')

# Sampling
def sample(model, num_samples, num_timesteps, device):
    model.eval()
    samples = torch.randn(num_samples, 784, device=device)
    for t in reversed(range(num_timesteps)):
        samples = model.reverse(samples, t)
    samples = samples.view(num_samples, 1, 28, 28)
    return samples

num_samples = 16
samples = sample(model, num_samples, num_timesteps, device).cpu().detach()

fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(samples[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
