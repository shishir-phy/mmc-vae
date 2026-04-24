import torch
import torch.nn as nn

class CNN_VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # =========================
        # ENCODER
        # =========================
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # (40x130 → 20x65)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → (10x33)
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(32 * 10 * 33, latent_dim)
        self.fc_logvar = nn.Linear(32 * 10 * 33, latent_dim)

        # =========================
        # DECODER
        # =========================
        self.fc_dec = nn.Linear(latent_dim, 32 * 10 * 33)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 10, 33)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_dec(z)
        recon = self.decoder(h_dec)

        return recon, mu, logvar
