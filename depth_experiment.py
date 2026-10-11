import logging
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from shared import Settings
from train import train_runs

warnings.filterwarnings("ignore")
for _name in ("lightning", "pytorch_lightning"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# ---------- parameters ----------
# DEPTHS = (2, 4, 6)              # number of hidden layers
# WIDTHS = (12, 10, 8, 6, 4, 2)   # depth d uses the first d widths (+ the output layer)
DEPTHS = (1, 3, 5)              # number of hidden layers
WIDTHS = (10, 7, 5, 4, 3)   # depth d uses the first d widths (+ the output layer)
EPOCHS = 8000                   # fixed length, no early stopping
RUNS = 10                        # runs per depth
LR = 0.002                      # learning rate
EVERY = 10                     # log I(X;T), I(T;Y) of every layer every 200 epochs
TRAIN_FRACTION = 0.8            # paper, Fig. 5
ACC_TARGET = 0.95               # test-accuracy threshold: declare it BEFORE looking at results
ROOT = Path(__file__).parent / "runs" / "depth_minimal"


# ---------- MI (same estimator as experiment 1: 30 bins on [-1, 1], all 4096 patterns) ----------
def entropy(ids):
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))


def cond_entropy(t_ids, g_ids):  # H(T|G) = H(T,G) - H(G)
    return entropy(t_ids.astype(np.int64) * (g_ids.max() + 1) + g_ids) - entropy(g_ids)


def mutual_information(act, x_ids, y_ids):
    binned = np.digitize(act, np.linspace(-1, 1, 30))
    t_ids = np.unique(binned, axis=0, return_inverse=True)[1].reshape(-1)
    h_t = entropy(t_ids)
    return h_t - cond_entropy(t_ids, x_ids), h_t - cond_entropy(t_ids, y_ids)


# ---------- logging during training ----------
class Log(Callback):
    def on_train_start(self, trainer, pl_module):
        self.x, targets = trainer.datamodule.dataset.tensors
        self.y = targets.argmax(dim=1)
        self.x_ids = np.unique(self.x.numpy(), axis=0, return_inverse=True)[1].reshape(-1)
        self.dir = Path(trainer.logger.save_dir)
        self.test = None
        (self.dir / "log.csv").write_text("epoch,layer,I_XT,I_TY,test_acc,test_loss\n")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % EVERY:  # measure after 200, 400, ..., 8000 epochs
            return
        if self.test is None:  # held-out patterns = everything not in the training split
            train = np.load(self.dir / "split.npz")["train"]
            self.test = np.setdiff1d(np.arange(len(self.x)), train)
        act = getattr(torch, pl_module.settings.activation)
        outs = []
        with torch.no_grad():
            h = self.x.to(pl_module.device)
            for layer in pl_module.layers[:-1]:
                h = act(layer(h))
                outs.append(h.cpu().numpy())
            logits = pl_module.layers[-1](h).cpu()
        outs.append(torch.softmax(logits, dim=1).numpy())
        acc = (logits[self.test].argmax(1) == self.y[self.test]).float().mean().item()
        loss = F.cross_entropy(logits[self.test], self.y[self.test]).item()
        with (self.dir / "log.csv").open("a") as f:
            for i, o in enumerate(outs, 1):
                ixt, ity = mutual_information(o, self.x_ids, self.y.numpy())
                f.write(f"{epoch},{i},{ixt:.6f},{ity:.6f},{acc:.6f},{loss:.6f}\n")
        self.last = (acc, ity)  # ity is now the output layer's I(T;Y)

    def on_train_end(self, trainer, pl_module):
        print(f"  finished: test acc {self.last[0]:.3f}, I(T_out;Y) {self.last[1]:.3f} bits")


def data_path():
    """First real MATLAB file among data/ and the submodule copy (the data/ symlink breaks on Windows)."""
    here = Path(__file__).resolve().parent
    for p in (here / "data" / "var_u.mat", here / "references" / "IDNNs" / "data" / "var_u.mat"):
        if p.is_file() and p.read_bytes()[:8].startswith((b"MATLAB", b"\x89HDF")):
            return str(p)
    raise SystemExit("No real var_u.mat found in data/ or references/IDNNs/data/")


def run():
    path = data_path()
    print("Using data file:", path)
    for d in DEPTHS:
        print(f"\n=== {d} hidden layers, widths {WIDTHS[:d]} ===")
        s = replace(Settings(), data_path=path, hidden_sizes=WIDTHS[:d], learning_rate=LR, epochs=EPOCHS,
                    train_fraction=TRAIN_FRACTION)
        train_runs(s, RUNS, ROOT / f"d{d}", accelerator="cpu", callback=Log)
    summarize()


# ---------- summary: all run logs -> one file ----------
def summarize():
    """Collect every run's log.csv into runs/depth_minimal/summary.npz (analyze reads only this file)."""
    data = {}
    for d in DEPTHS:
        runs = [np.genfromtxt(p, delimiter=",", names=True) for p in sorted((ROOT / f"d{d}").glob("run-*/log.csv"))]
        if not runs:
            print(f"d{d}: no log.csv found, skipping")
            continue
        L = int(np.max(runs[0]["layer"]))
        col = lambda name: np.stack([np.atleast_1d(r[name]).reshape(-1, L) for r in runs])  # runs x epochs x layers
        data[f"d{d}_epochs"] = col("epoch")[0, :, 0].astype(int) + 1
        data[f"d{d}_all_runs"] = np.stack([col("I_XT"), col("I_TY")], axis=-1)  # runs x epochs x layers x (I_XT, I_TY)
        data[f"d{d}_test_acc"] = col("test_acc")[..., 0]                         # runs x epochs
        data[f"d{d}_test_loss"] = col("test_loss")[..., 0]                       # runs x epochs
    if not data:
        raise SystemExit(f"No log.csv files found under {ROOT.resolve()}. Did `run` finish?")
    np.savez(ROOT / "summary.npz", **data)
    print(f"Summary of {len(data) // 4} depth(s) saved in {(ROOT / 'summary.npz').resolve()}")


# ---------- analysis: summary.npz -> table + plots (as in experiment 1, one set per depth) ----------
def analyze():
    path = ROOT / "summary.npz"
    if not path.is_file():
        raise SystemExit(f"{path.resolve()} not found. Run `python depth_minimal.py summarize` first.")
    data = np.load(path)
    depths = sorted(int(k[1:].split("_")[0]) for k in data.files if k.endswith("_epochs"))
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    fig_last, axes_last = plt.subplots(1, 2, figsize=(11, 4))  # output layer I(X;T) and I(T;Y), all depths
    header = f"{'depth':>5} {'I(X;T)out':>10} {'I(T;Y)out':>10} {'test acc':>9} {'test loss':>10} {'epochs to acc>=' + str(ACC_TARGET):>20}"
    print(header)
    table = ["depth,I_XT_out,I_TY_out,test_acc,test_loss,epochs_to_threshold_median,epochs_to_threshold_per_run"]
    for d in depths:
        epochs = data[f"d{d}_epochs"]
        all_runs, acc, loss = data[f"d{d}_all_runs"], data[f"d{d}_test_acc"], data[f"d{d}_test_loss"]
        ixt, ity = all_runs[..., 0], all_runs[..., 1]
        L = all_runs.shape[2]
        mean = all_runs.mean(axis=0)  # epochs x layers x (I_XT, I_TY), mean over runs

        # table: epochs to threshold per run (inf = never reached)
        reach = [epochs[np.argmax(a >= ACC_TARGET)] if (a >= ACC_TARGET).any() else np.inf for a in acc]
        med = np.median(reach)
        med_text = f">{epochs[-1]}" if np.isinf(med) else str(int(med))
        per_run = [int(r) if np.isfinite(r) else None for r in reach]
        print(f"{d:>5} {ixt[:, -1, -1].mean():>10.3f} {ity[:, -1, -1].mean():>10.3f} {acc[:, -1].mean():>9.3f} "
              f"{loss[:, -1].mean():>10.3f} {med_text:>20}   (per run: {per_run})")
        table.append(f"{d},{ixt[:, -1, -1].mean():.4f},{ity[:, -1, -1].mean():.4f},{acc[:, -1].mean():.4f},"
                     f"{loss[:, -1].mean():.4f},{med_text},{' '.join(str(p) for p in per_run)}")

        # figure 1: information plane (same as experiment 1)
        norm = LogNorm(vmin=1, vmax=epochs.max())
        cmap = plt.get_cmap("gnuplot")
        figure, axis = plt.subplots(figsize=(6, 5))
        for i in range(len(epochs)):
            axis.plot(mean[i, :, 0], mean[i, :, 1], color=cmap(norm(epochs[i])), alpha=0.1, linewidth=0.5)
        for layer in range(L):
            points = axis.scatter(mean[:, layer, 0], mean[:, layer, 1], c=epochs, cmap=cmap, norm=norm, s=12, zorder=3)
        figure.colorbar(points, ax=axis, label="Epochs")
        axis.set_xlabel("$I(X;T)$ [bits]")
        axis.set_ylabel("$I(T;Y)$ [bits]")
        axis.set_title(f"Information plane, {d} hidden layers, mean of {len(all_runs)} run(s)")
        figure.tight_layout()
        figure.savefig(out / f"d{d}_information_plane.png", dpi=200)
        plt.close(figure)

        # figure 2: information vs epochs, per layer (same as experiment 1)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for layer in range(L):
            label = f"Layer {layer + 1}" if layer < L - 1 else "Output"
            axes[0].plot(epochs, mean[:, layer, 0], label=label)
            axes[1].plot(epochs, mean[:, layer, 1], label=label)
        for axis, name in zip(axes, ("$I(X;T)$", "$I(T;Y)$")):
            axis.set_xscale("log")
            axis.set_xlabel("# Epochs")
            axis.set_ylabel(f"{name} [bits]")
        axes[0].legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(out / f"d{d}_information_vs_epochs.png", dpi=200)
        plt.close(figure)

        ax_acc.plot(epochs, acc.mean(0), label=f"{d} hidden")
        axes_last[0].plot(epochs, mean[:, -1, 0], label=f"{d} hidden")  # output layer I(X;T)
        axes_last[1].plot(epochs, mean[:, -1, 1], label=f"{d} hidden")  # output layer I(T;Y)

    # figure 3: test accuracy per depth
    ax_acc.axhline(ACC_TARGET, color="grey", ls=":")
    ax_acc.set(xscale="log", xlabel="# Epochs", ylabel="test accuracy")
    ax_acc.legend(fontsize="small")
    fig_acc.tight_layout()
    fig_acc.savefig(out / "test_accuracy.png", dpi=200)

    # figure 4: I(X;T) and I(T;Y) of the last (output) layer, one line per depth
    for axis, name in zip(axes_last, ("$I(X;T)$", "$I(T;Y)$")):
        axis.set_xscale("log")
        axis.set_xlabel("# Epochs")
        axis.set_ylabel(f"{name} of output layer [bits]")
    axes_last[1].legend(fontsize="small")
    fig_last.tight_layout()
    fig_last.savefig(out / "last_layer_information.png", dpi=200)
    plt.close(fig_last)

    (ROOT / "summary.csv").write_text("\n".join(table) + "\n")
    print(f"\nTable saved in {(ROOT / 'summary.csv').resolve()}\nFigures saved in {out.resolve()}:")
    for f in sorted(out.glob("*.png")):
        print("  ", f.name)

    # figure 3: test accuracy per depth
    ax_acc.axhline(ACC_TARGET, color="grey", ls=":")
    ax_acc.set(xscale="log", xlabel="# Epochs", ylabel="test accuracy")
    ax_acc.legend(fontsize="small")
    fig_acc.tight_layout()
    fig_acc.savefig(out / "test_accuracy.png", dpi=200)
    (ROOT / "summary.csv").write_text("\n".join(table) + "\n")
    print(f"\nTable saved in {(ROOT / 'summary.csv').resolve()}\nFigures saved in {out.resolve()}:")
    for f in sorted(out.glob("*.png")):
        print("  ", f.name)


if __name__ == "__main__":
    {"run": run, "summarize": summarize, "analyze": analyze}[sys.argv[1]]()