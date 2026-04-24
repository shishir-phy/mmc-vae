import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LATENT_DIM = 32
BETA = 4
EPOCHS = 30

# =========================
# LOAD DATA
# =========================
mfcc = np.load("features/mfcc_features.npy")
mfcc = mfcc[:, np.newaxis, :, :]

X = torch.tensor(mfcc, dtype=torch.float32)
loader = DataLoader(TensorDataset(X), batch_size=32, shuffle=True)

# =========================
# MODEL
# =========================
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(32*10*33, latent_dim)
        self.fc_logvar = nn.Linear(32*10*33, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 32*10*33)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32,10,33)),
            nn.ConvTranspose2d(32,16,3,2,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,2,1,1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std

    def forward(self,x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(self.fc_dec(z))
        return recon, mu, logvar

# =========================
# LOSS
# =========================
def loss_fn(recon,x,mu,logvar):
    recon = recon[:,:,:x.shape[2],:x.shape[3]]
    recon_loss = nn.functional.mse_loss(recon,x)
    kl = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    return recon_loss + BETA*kl

# =========================
# TRAIN
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetaVAE(LATENT_DIM).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    total=0
    for batch in loader:
        x = batch[0].to(device)
        opt.zero_grad()
        recon,mu,logvar = model(x)
        loss = loss_fn(recon,x,mu,logvar)
        loss.backward()
        opt.step()
        total+=loss.item()
    print(f"Epoch {epoch+1}, Loss: {total:.4f}")

# =========================
# SAVE LATENT
# =========================
model.eval()
Z=[]
with torch.no_grad():
    for batch in loader:
        x=batch[0].to(device)
        h=model.encoder(x)
        mu=model.fc_mu(h)
        Z.append(mu.cpu().numpy())

Z=np.vstack(Z)
np.save("features/latent_audio_beta.npy",Z)

print("Saved:", Z.shape)
