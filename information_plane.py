"""Experiment 1: information-plane dynamics.

Trains the fixed network from shared.py and records I(X;T) and I(T;Y) for every
layer (hidden layers + softmax output) during training, following the authors'
released code (ravidziv/IDNNs): activations of all 4096 patterns, 30 equal bins
in [-1, 1], measured at the start of ~400 log-spaced epochs.
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pytorch_lightning.callbacks import Callback
import torch

from shared import Settings
from train import train_runs


def entropy(ids):
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))


def conditional_entropy(t_ids, group_ids):
    # H(T|G) = sum_g p(g) H(T|G=g) = H(T,G) - H(G)
    return entropy(t_ids.astype(np.int64) * (group_ids.max() + 1) + group_ids) - entropy(group_ids)


def mutual_information(activations, x_ids, y_ids, num_bins=30):
    bins = np.linspace(-1, 1, num_bins)
    binned = np.digitize(activations, bins)  # same bin index as bins[digitize - 1] in the authors' code
    _, t_ids = np.unique(binned, axis=0, return_inverse=True)
    t_ids = t_ids.reshape(-1)
    h_t = entropy(t_ids)
    return h_t - conditional_entropy(t_ids, x_ids), h_t - conditional_entropy(t_ids, y_ids)


def measurement_epochs(epochs, num_points=400):
    # Authors: np.unique(np.logspace(log2(1), log2(epochs), 400, base=2, dtype=int)) - 1
    return set((np.unique(np.logspace(0, np.log2(epochs), num_points, base=2, dtype=int)) - 1).tolist())


class InformationPlane(Callback):
    def __init__(self, num_bins=30):
        self.num_bins = num_bins

    def on_train_start(self, trainer, pl_module):
        inputs, targets = trainer.datamodule.dataset.tensors
        self.inputs = inputs
        _, x_ids = np.unique(inputs.numpy(), axis=0, return_inverse=True)
        self.x_ids = x_ids.reshape(-1)
        self.y_ids = targets.argmax(dim=1).numpy()
        self.epochs = measurement_epochs(pl_module.settings.epochs)
        self.path = Path(trainer.logger.save_dir) / "information.csv"
        # On resume, keep earlier rows and only add new epochs.
        self.done = set()
        if self.path.is_file():
            with self.path.open() as file:
                self.done = {int(row["epoch"]) for row in csv.DictReader(file)}

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch not in self.epochs or epoch in self.done:
            return
        activation = {"tanh": torch.tanh, "relu": torch.relu}[pl_module.settings.activation]
        layers = []
        with torch.no_grad():
            hidden = self.inputs.to(pl_module.device)
            for layer in pl_module.layers[:-1]:
                hidden = activation(layer(hidden))
                layers.append(hidden.cpu().numpy())
            layers.append(torch.softmax(pl_module.layers[-1](hidden), dim=1).cpu().numpy())
        new_file = not self.path.is_file()
        with self.path.open("a", newline="") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["epoch", "layer", "I_XT", "I_TY"])
            for index, values in enumerate(layers, 1):
                i_xt, i_ty = mutual_information(values, self.x_ids, self.y_ids, self.num_bins)
                writer.writerow([epoch, index, f"{i_xt:.6f}", f"{i_ty:.6f}"])
        self.done.add(epoch)


def read_information(run_dir):
    with (Path(run_dir) / "information.csv").open() as file:
        rows = list(csv.DictReader(file))
    epochs = sorted({int(row["epoch"]) for row in rows})
    layers = max(int(row["layer"]) for row in rows)
    values = np.full((len(epochs), layers, 2), np.nan)
    position = {epoch: index for index, epoch in enumerate(epochs)}
    for row in rows:
        values[position[int(row["epoch"])], int(row["layer"]) - 1] = float(row["I_XT"]), float(row["I_TY"])
    return np.array(epochs), values


def plot(root):
    root = Path(root)
    runs = [read_information(path.parent) for path in sorted(root.glob("run-*/information.csv"))]
    # Average over runs on the epochs every run has (the paper averages 50 runs).
    shared_epochs = sorted(set.intersection(*(set(epochs.tolist()) for epochs, _ in runs)))
    stacked = np.stack([values[np.isin(epochs, shared_epochs)] for epochs, values in runs])
    mean = stacked.mean(axis=0)
    epochs = np.array(shared_epochs) + 1
    norm = LogNorm(vmin=1, vmax=epochs.max())
    cmap = plt.get_cmap("gnuplot")

    figure, axis = plt.subplots(figsize=(6, 5))
    for index in range(len(epochs)):
        axis.plot(mean[index, :, 0], mean[index, :, 1], color=cmap(norm(epochs[index])), alpha=0.1, linewidth=0.5)
    for layer in range(mean.shape[1]):
        points = axis.scatter(mean[:, layer, 0], mean[:, layer, 1], c=epochs, cmap=cmap, norm=norm, s=12, zorder=3)
    figure.colorbar(points, ax=axis, label="Epochs")
    axis.set_xlabel("$I(X;T)$ [bits]")
    axis.set_ylabel("$I(T;Y)$ [bits]")
    axis.set_title(f"Information plane, mean of {len(runs)} run(s)")
    figure.tight_layout()
    figure.savefig(root / "information_plane.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for layer in range(mean.shape[1]):
        label = f"Layer {layer + 1}" if layer < mean.shape[1] - 1 else "Output"
        axes[0].plot(epochs, mean[:, layer, 0], label=label)
        axes[1].plot(epochs, mean[:, layer, 1], label=label)
    for axis, name in zip(axes, ("$I(X;T)$", "$I(T;Y)$")):
        axis.set_xscale("log")
        axis.set_xlabel("# Epochs")
        axis.set_ylabel(f"{name} [bits]")
    axes[0].legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(root / "information_vs_epochs.png", dpi=200)
    plt.close(figure)
    np.savez(root / "information_mean.npz", epochs=epochs, mean=mean, all_runs=stacked)
    print(f"Saved plots for {len(runs)} run(s) in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--accelerator", default="cpu", help="cpu is usually fastest for this tiny network")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    folder = "information_plane" if args.epochs == 8000 else f"information_plane_{args.epochs}epochs"
    root = Path(__file__).parent / "runs" / folder
    if not args.plot_only:
        settings = replace(Settings(), epochs=args.epochs)
        train_runs(settings, args.runs, root, accelerator=args.accelerator, callback=InformationPlane)
    plot(root)
