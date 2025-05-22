import numpy as np
import torch
from fairseq2.typing import DataType, Device
from torch import Tensor, nn


class KmeansModel(nn.Module):
    def __init__(self, km_path: str, device: Device, dtype: DataType):
        super().__init__()
        km_model = np.load(km_path)
        centroids_numpy = km_model.transpose()
        # self.embed = torch.from_numpy(centroids_numpy).to(device=device, dtype=dtype)
        # self.embed_norm = self.embed.pow(2).sum(0, keepdim=True)
        self.embed = nn.Parameter(
            torch.from_numpy(centroids_numpy).to(device=device, dtype=dtype)
        )
        self.register_buffer("embed_norm", self.embed.pow(2).sum(0, keepdim=True))
        self.dim, self.n_codes = self.embed.shape

    def forward(self, x: Tensor) -> Tensor:
        flatten = x.reshape(-1, self.dim)
        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed_norm
        )
        _, embed_ind = dist.max(1)
        return embed_ind.view(*x.shape[:-1])


if __name__ == "__main__":

    km = KmeansModel(
        "data/checkpoints/kmeans_10k.npy",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    features = torch.load("data/w2v_features.pt")

    a = km(features[0])
    b = km.old_forward(features[0])
