from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.io import loadmat
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler


DEFAULT_DATA = Path(__file__).parent / "data/var_u.mat"


@dataclass(frozen=True)
class Settings:
    data_path: str = str(DEFAULT_DATA)
    hidden_sizes: tuple[int, ...] = (10, 7, 5, 4, 3)
    activation: str = "tanh"
    learning_rate: float = 0.0004
    batch_size: int = 512
    epochs: int = 8000
    train_fraction: float = 0.8
    seed: int = 0
    split_seed: int = 0
    random_split: bool = False
    shuffle_batches: bool = False


def load_dataset(path: str | Path) -> TensorDataset:
    arrays = loadmat(path)
    inputs = np.asarray(arrays["F"], dtype=np.float32)
    labels = np.asarray(arrays["y"], dtype=np.float32).reshape(-1)
    # Convert labels to two class probabilities in the authors' order.
    targets = np.column_stack((labels, 1 - labels))
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))


class IBData(pl.LightningDataModule):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.dataset = load_dataset(settings.data_path)
        indices = np.arange(len(self.dataset))
        if settings.random_split:
            indices = np.random.default_rng(settings.split_seed).permutation(indices)
        # Divide the indices into training and test sets.
        train_size = int(np.rint(len(indices) * settings.train_fraction))
        if not 0 < train_size < len(indices):
            raise ValueError("The split must leave samples in both sets.")
        self.train_indices = indices[:train_size]
        self.test_indices = indices[train_size:]
        self.train_data = Subset(self.dataset, self.train_indices.tolist())
        self.test_data = Subset(self.dataset, self.test_indices.tolist())

    def train_dataloader(self):
        # Lightning restores the sampler's epoch. One replica keeps every sample once.
        sampler = DistributedSampler(
            self.train_data,
            num_replicas=1,
            rank=0,
            shuffle=self.settings.shuffle_batches,
            seed=self.settings.seed,
        )
        return DataLoader(
            self.train_data,
            batch_size=self.settings.batch_size,
            sampler=sampler,
            num_workers=0,
        )

    def val_dataloader(self):
        # Evaluate both sets at the end of each epoch.
        return [
            DataLoader(self.train_data, batch_size=self.settings.batch_size, num_workers=0),
            DataLoader(self.test_data, batch_size=self.settings.batch_size, num_workers=0),
        ]


class IBNetwork(pl.LightningModule):
    def __init__(self, input_size: int, settings: Settings | dict):
        super().__init__()
        if isinstance(settings, dict):
            settings = Settings(**settings)
        self.save_hyperparameters({"input_size": input_size, "settings": asdict(settings)})
        self.settings = settings
        sizes = (input_size, *settings.hidden_sizes, 2)
        self.layers = nn.ModuleList(
            nn.Linear(input_width, output_width)
            for input_width, output_width in zip(sizes[:-1], sizes[1:])
        )
        # Scale initial weights by the layer's input width.
        for layer in self.layers:
            standard_deviation = layer.in_features**-0.5
            nn.init.trunc_normal_(
                layer.weight,
                std=standard_deviation,
                a=-2 * standard_deviation,
                b=2 * standard_deviation,
            )
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        # Apply activations to hidden layers and return the output logits.
        activation = {"tanh": torch.tanh, "relu": torch.relu}[self.settings.activation]
        for layer in self.layers[:-1]:
            inputs = activation(layer(inputs))
        return self.layers[-1](inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = functional.cross_entropy(self(inputs), targets)
        self.log("batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        logits = self(inputs)
        prefix = "train" if dataloader_idx == 0 else "test"
        loss = functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets.argmax(dim=1)).float().mean()
        self.log_dict(
            {f"{prefix}_loss": loss, f"{prefix}_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            batch_size=len(inputs),
            add_dataloader_idx=False,
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.settings.learning_rate, weight_decay=0)