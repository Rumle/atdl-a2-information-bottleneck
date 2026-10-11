import argparse
import csv
from dataclasses import asdict, replace
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
        # Measure batch gradients after the weight update, following the authors' code.
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
            # Measure the spread of batch gradient vectors around their mean.
            variability = (gradients - mean).square().sum(dim=1).mean().sqrt()
            weight_norm = layer.weight.detach().norm()
            metrics[f"gradient_mean_layer_{layer_index}"] = mean.norm() / weight_norm
            metrics[f"gradient_std_layer_{layer_index}"] = variability / weight_norm
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, batch_size=1)

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


def plot(run_dir: str | Path, num_runs: int = 50, epochs: int = 8000):
    """Plot minibatch gradient statistics averaged across runs."""
    run_dir = Path(run_dir)
    runs = [read_metrics(run_dir / f"run-{index + 1}") for index in range(num_runs)]
    for index, rows in enumerate(runs, 1):
        if [int(row["epoch"]) for row in rows] != list(range(epochs)):
            raise ValueError(f"Run {index} must contain exactly {epochs} consecutive epochs.")
    fields = [key for key in runs[0][0] if key.startswith("gradient_") or key == "train_loss"]
    rows = []
    # Average each metric across runs at the same epoch.
    for epoch in range(epochs):
        row = {"epoch": epoch}
        for field in fields:
            values = [float(run[epoch][field]) for run in runs]
            if not all(isfinite(value) for value in values):
                raise ValueError(f"Non-finite {field} at epoch {epoch}.")
            row[field] = fmean(values)
        rows.append(row)
    with (run_dir / "gradient_metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["epoch", *fields])
        writer.writeheader()
        writer.writerows(rows)
    epoch_numbers = [int(row["epoch"]) + 1 for row in rows]
    layer_count = sum(key.startswith("gradient_mean_layer_") for key in rows[0])
    figure, axis = plt.subplots(figsize=(8, 5.5))
    marker = "." if len(rows) == 1 else None
    colors = ("red", "c", "blue", "green", "orange", "purple")
    mean_lines, std_lines = [], []
    for layer_index in range(1, layer_count + 1):
        mean = [float(row[f"gradient_mean_layer_{layer_index}"]) for row in rows]
        variability = [float(row[f"gradient_std_layer_{layer_index}"]) for row in rows]
        color = colors[(layer_index - 1) % len(colors)]
        mean_line, = axis.plot(epoch_numbers, mean, marker=marker, linewidth=1, color=color)
        std_line, = axis.plot(epoch_numbers, variability, marker=marker, linestyle=":", linewidth=2, color=color)
        mean_lines.append(mean_line)
        std_lines.append(std_line)
    axis.set(xscale="log", yscale="log", xlabel="# Epochs", ylabel="Normalized Mean and STD")
    axis.set_xlim(1, 9000)
    axis.set_ylim(0.0003, 10)
    axis.set_xticks([1, 10, 100, 1000, 9000], labels=["1", "10", "100", "1000", "9000"])
    axis.set_yticks([0.001, 0.01, 0.1, 1, 10])
    axis.tick_params(which="both", direction="in", top=True, right=True)
    std_legend = axis.legend(std_lines, [""] * layer_count, title=r"$STD(\nabla W_i)$",
                             loc="upper left", bbox_to_anchor=(0, 1), fontsize="small",
                             title_fontsize="small", framealpha=1, fancybox=False)
    axis.add_artist(std_legend)
    axis.legend(mean_lines, [f"Layer {index}" for index in range(1, layer_count + 1)], title=r"$\|Mean(\nabla W_i)\|$",
                loc="upper left", bbox_to_anchor=(0.14, 1), fontsize="small",
                title_fontsize="small", framealpha=1, fancybox=False)
    figure.tight_layout()
    figure.savefig(run_dir / "gradient_transitions.png", dpi=200)
    plt.close(figure)
    return rows


def sweep(settings: Settings, run_dir: str | Path, accelerator="auto"):
    """One paired SGD run per rate. Rank stable rates by the last 100 epochs' mean loss."""
    run_dir = Path(run_dir)
    results = []
    best = None
    for rate in (0.002, 0.004, 0.008):
        rate_dir = run_dir / f"lr-{rate:g}"
        train_runs(
            replace(settings, learning_rate=rate),
            1, rate_dir, accelerator=accelerator, callback=GradientTransitions,
        )
        rows = read_metrics(rate_dir / "run-1")
        if [int(row["epoch"]) for row in rows] != list(range(settings.epochs)):
            raise ValueError(f"Pilot at lr={rate:g} must contain {settings.epochs} consecutive epochs.")
        losses = [float(row["train_loss"]) for row in rows]
        # Check final loss improvement and upward steps across the full run.
        tail = losses[-100:]
        tail_loss = fmean(tail)
        rebound = max((right - left for left, right in zip(losses, losses[1:])), default=0.0)
        rebound = max(0.0, rebound)
        stable = all(isfinite(loss) for loss in losses) and tail_loss < losses[0] and rebound <= 0.01
        result = {
            "learning_rate": rate,
            "first_epoch_loss": losses[0],
            "tail_loss": tail_loss,
            "max_rebound": rebound,
            "stable": stable,
        }
        results.append(result)
        if stable and (best is None or tail_loss < best["tail_loss"]):
            best = result
    for result in results:
        result["selected"] = result is best
        result["selection_rule"] = "Choose the lowest average loss in the last 100 epochs. All losses must be finite, this average must be below the first loss and loss must rise by at most 0.01 nats per epoch."
    with (run_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Stable: finite losses, lower tail mean, full-run rebounds <= 0.01 nats.")
    for result in results:
        print(result)
    print(f"Pilot choice: lr={best['learning_rate']:g}" if best else "No stable rate found.")
    return best


def experiment(settings: Settings, run_dir: str | Path, sweep_epochs=8000, num_runs=50, accelerator="auto", workers=1):
    run_dir = Path(run_dir)
    pilot_settings = Settings(**{**asdict(settings), "epochs": sweep_epochs})
    best = sweep(pilot_settings, run_dir / f"sweep-{sweep_epochs}", accelerator)
    if best is None:
        raise ValueError("No stable pilot rate found. The full experiment was not started.")
    # Use fresh seeds for the final experiment.
    settings = replace(settings, learning_rate=best["learning_rate"], seed=settings.seed + 1)
    results_dir = run_dir / f"lr-{settings.learning_rate:g}"
    train_runs(settings, num_runs, results_dir, accelerator=accelerator, callback=GradientTransitions, workers=workers)
    plot(results_dir, num_runs, settings.epochs)
    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Probe three SGD rates with one paired run each.")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--sweep-epochs", type=int, default=8000)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1, help="Number of runs to train at once.")
    args = parser.parse_args()
    if min(args.epochs, args.sweep_epochs, args.runs, args.workers) < 1:
        parser.error("--epochs, --sweep-epochs, --runs and --workers must be positive")
    settings = Settings(epochs=args.epochs)
    root = Path(__file__).parent / "runs/gradient_transitions_experiment"
    if args.sweep:
        pilot_settings = Settings(epochs=args.sweep_epochs)
        sweep(pilot_settings, root / f"sweep-{args.sweep_epochs}")
    else:
        experiment(settings, root, args.sweep_epochs, args.runs, workers=args.workers)