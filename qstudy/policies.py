"""qstudy policies — WindowGRU feature extractor (Stage 10.1)."""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WindowGRUExtractor(BaseFeaturesExtractor):
    """Input observation (K*D,) laid out [oldest..newest] frames; reshape to (batch, K, D),
    one nn.GRU(input_size=D, hidden_size=64, batch_first=True); features = last hidden state
    (features_dim=64)."""

    def __init__(self, observation_space, K=8, D=5, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.K = int(K)
        self.D = int(D)
        self.gru = nn.GRU(input_size=self.D, hidden_size=features_dim, batch_first=True)

    def forward(self, observations):
        b = observations.shape[0]
        x = observations.view(b, self.K, self.D)
        out, _ = self.gru(x)
        return out[:, -1, :]
