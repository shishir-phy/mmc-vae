import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from VAE import VAE


## Load dataset
X = np.load("features/X_features.npy")   # or X_reduced.npy
X_tensor = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


## Loss Function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    
    return recon_loss + kl_loss


## Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(input_dim=X.shape[1], latent_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch[0].to(device)

        optimizer.zero_grad()

        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



## Extract latent features
model.eval()

latent_vectors = []

with torch.no_grad():
    for batch in loader:
        x = batch[0].to(device)
        h = model.encoder(x)
        mu = model.mu(h)   # use mean as representation
        latent_vectors.append(mu.cpu().numpy())

Z = np.vstack(latent_vectors)

os.makedirs('features', exist_ok=True)
## Save Latent vectors 
np.save("features/latent_vectors.npy", Z)
