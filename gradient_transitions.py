import argparse
import csv
from dataclasses import replace
from math import isfinite
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import torch
from torch.nn import functional

from shared import Settings
from train import train_runs


class GradientTransitions(Callback):
    """Post-update weight gradients, equal batch weights, epoch-end weight norms."""

    def on_train_epoch_start(self, trainer, pl_module):
        self.gradients = [[] for _ in pl_module.layers]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        inputs, targets = batch
        # Measure after the update, like the authors' code, without changing .grad.
        with torch.enable_grad():
            loss = functional.cross_entropy(pl_module(inputs), targets)
            gradients = torch.autograd.grad(loss, [layer.weight for layer in pl_module.layers])
        for batches, gradient in zip(self.gradients, gradients):
            batches.append(gradient.detach().flatten())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {}
        for layer_index, layer in enumerate(pl_module.layers, 1):
            gradients = torch.stack(self.gradients[layer_index - 1])
            mean = gradients.mean(dim=0)
            # Compare gradient vectors, not just their lengths, to capture direction changes.
            variability = (gradients - mean).square().sum(dim=1).mean().sqrt()
            weight_norm = layer.weight.detach().norm()
            metrics[f"gradient_mean_layer_{layer_index}"] = mean.norm() / weight_norm
            metrics[f"gradient_std_layer_{layer_index}"] = variability / weight_norm
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, batch_size=1)

    def on_train_end(self, trainer, pl_module):
        trainer.logger.save()
        plot(trainer.logger.save_dir, pl_module.settings.optimizer, pl_module.settings.learning_rate)


def read_metrics(run_dir: str | Path):
    """Combine saved training sessions, keeping the latest measurement per epoch."""
    run_dir = Path(run_dir)
    records = {}
    metrics_paths = sorted(
        run_dir.glob("logs/version_*/metrics.csv"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    for metrics_path in metrics_paths:
        with metrics_path.open(newline="") as metrics_file:
            for row in csv.DictReader(metrics_file):
                if row.get("train_loss") and row.get("gradient_mean_layer_1"):
                    records[int(row["epoch"])] = row
    return [records[epoch] for epoch in sorted(records)]


def plot(run_dir: str | Path, optimizer: str = "adam", learning_rate: float = 0.002):
    run_dir = Path(run_dir)
    rows = read_metrics(run_dir)
    epochs = [int(row["epoch"]) + 1 for row in rows]
    losses = [float(row["train_loss"]) for row in rows]
    layer_count = sum(key.startswith("gradient_mean_layer_") for key in rows[0])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    marker = "." if len(rows) == 1 else None
    colors = ("red", "c", "blue", "green", "orange", "purple")
    for layer_index in range(1, layer_count + 1):
        mean = [float(row[f"gradient_mean_layer_{layer_index}"]) for row in rows]
        variability = [float(row[f"gradient_std_layer_{layer_index}"]) for row in rows]
        color = colors[(layer_index - 1) % len(colors)]
        axes[0].plot(epochs, mean, marker=marker, linewidth=2, color=color, label=f"Layer {layer_index}")
        axes[0].plot(epochs, variability, marker=marker, linestyle=":", linewidth=2, color=color)
        axes[1].scatter(losses, mean, marker=".", s=8, alpha=0.5, color=color)
        axes[1].scatter(losses, variability, marker="x", s=8, alpha=0.5, color=color)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("# Epochs")
    axes[1].set_xlabel("Full training loss")
    axes[0].set_ylabel("Normalized Mean and STD")
    axes[0].set_title("Mean (solid), STD (dotted)")
    axes[1].set_title("Mean (dots), STD (crosses)")
    for axis in axes:
        axis.set_yscale("log")
    axes[0].legend(fontsize="small", ncol=2)
    label = "Released-code-style baseline" if optimizer == "adam" else "Optimizer comparison"
    figure.suptitle(f"{optimizer.upper()}: {label} (lr={learning_rate:g})")
    figure.tight_layout()
    figure.savefig(run_dir / "gradient_transitions.png", dpi=200)
    plt.close(figure)


def sweep(settings: Settings, run_dir: str | Path, accelerator="auto"):
    """One paired SGD run per rate. Rank stable rates by the last 100 epochs' mean loss."""
    run_dir = Path(run_dir)
    results = []
    best = None
    for rate in (0.002, 0.004, 0.008):
        rate_dir = run_dir / f"lr-{rate:g}"
        train_runs(
            replace(settings, optimizer="sgd", learning_rate=rate),
            1, rate_dir, accelerator=accelerator, callback=GradientTransitions,
        )
        rows = read_metrics(rate_dir / "run-1")
        losses = [float(row["train_loss"]) for row in rows]
        tail = losses[-100:]
        tail_loss = fmean(tail)
        rebound = max((right - left for left, right in zip(tail, tail[1:])), default=0.0)
        rebound = max(0.0, rebound)
        stable = all(isfinite(loss) for loss in losses) and tail_loss < losses[0] and rebound <= 0.01
        result = {
            "learning_rate": rate,
            "first_epoch_loss": losses[0],
            "tail_loss": tail_loss,
            "max_tail_rebound": rebound,
            "stable": stable,
        }
        results.append(result)
        if stable and (best is None or tail_loss < best["tail_loss"]):
            best = result
    for result in results:
        result["selected"] = result is best
        result["selection_rule"] = "Lowest last-100 mean, finite losses, mean < first loss, last-100 rebound <= 0.01 nats"
    with (run_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Stable: finite losses, lower tail mean, late rebounds <= 0.01 nats.")
    for result in results:
        print(result)
    print(f"Pilot choice: lr={best['learning_rate']:g}" if best else "No stable rate found.")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--sweep", action="store_true", help="Probe three SGD rates with one paired run each.")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.learning_rate is not None and (not isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate must be finite and positive")
    settings = Settings(epochs=args.epochs, optimizer=args.optimizer)
    if args.learning_rate is not None:
        settings = replace(settings, learning_rate=args.learning_rate)
    root = Path(__file__).parent / "runs"
    if args.sweep:
        sweep(settings, root / "gradient_transitions_sgd_sweep")
    else:
        folder = "gradient_transitions" if args.optimizer == "adam" else "gradient_transitions_sgd"
        if args.learning_rate is not None:
            folder += f"_lr-{args.learning_rate:g}"
        train_runs(settings, 50, root / folder, accelerator="auto", callback=GradientTransitions)