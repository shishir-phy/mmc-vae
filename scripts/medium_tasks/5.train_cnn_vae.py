import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from cnn_vae_audio import CNN_VAE

# =========================
# LOAD DATA
# =========================
mfcc = np.load("features/mfcc_features.npy")
mfcc = mfcc[:, np.newaxis, :, :]

X = torch.tensor(mfcc, dtype=torch.float32)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_VAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# LOSS
# =========================
def loss_fn(recon, x, mu, logvar):
    # FIX SHAPE MISMATCH
    recon = recon[:, :, :x.shape[2], :x.shape[3]]

    recon_loss = torch.nn.functional.mse_loss(recon, x)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + 0.001 * kl

# =========================
# TRAIN
# =========================
epochs = 30

for epoch in range(epochs):
    total_loss = 0

    for batch in loader:
        x = batch[0].to(device)

        optimizer.zero_grad()

        recon, mu, logvar = model(x)
        loss = loss_fn(recon, x, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# EXTRACT AUDIO LATENT
# =========================
model.eval()

latent_audio = []

with torch.no_grad():
    for batch in loader:
        x = batch[0].to(device)
        h = model.encoder(x)
        mu = model.fc_mu(h)
        latent_audio.append(mu.cpu().numpy())

Z_audio = np.vstack(latent_audio)

np.save("features/latent_audio.npy", Z_audio)

print("Saved audio latent:", Z_audio.shape)
