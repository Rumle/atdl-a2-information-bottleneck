import numpy as np
from shared import IBNetwork
import torch
from scipy.io import loadmat
from shared import IBNetwork
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
def entropy(probabilities):
    """Entropy in bits. Zero-probability entries contribute zero."""
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]

    return -np.sum(p * np.log2(p))


def layer_mutual_information(activations, p_y1, bins=30):
    """
    Parameters
    ----------
    activations : array, shape (4096, number_of_neurons)
        Post-activation outputs for ALL input patterns, in dataset order.

    p_y1 : array, shape (4096,)
        True probabilities P(Y=1 | X=x), in the same order.
        These are NOT model predictions or sampled binary labels.

    bins : int
        Number of equal-width bins over [-1, 1].

    Returns
    -------
    Dictionary containing I(X;T), I(T;Y), and distributions.
    """
    a = np.asarray(activations, dtype=np.float64)
    p_y1 = np.asarray(p_y1, dtype=np.float64).reshape(-1)

    if a.ndim != 2:
        raise ValueError("activations must have shape (N, neurons).")

    n_inputs = a.shape[0]

    if n_inputs != 4096:
        raise ValueError("Evaluate all 4096 input patterns.")

    if len(p_y1) != n_inputs:
        raise ValueError("Each input must have one label probability.")

    if not np.isfinite(a).all():
        raise ValueError("Activations contain NaN or infinity.")

    if np.any(a < -1 - 1e-6) or np.any(a > 1 + 1e-6):
        raise ValueError("This binning requires activations in [-1, 1].")

    if (
        not np.isfinite(p_y1).all()
        or np.any(p_y1 < 0)
        or np.any(p_y1 > 1)
    ):
        raise ValueError("Label probabilities must be between 0 and 1.")

    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer >= 2.")

    # 1. Discretize each neuron's activation.
    # 29 internal boundaries define exactly 30 intervals.
    edges = np.linspace(-1, 1, bins + 1)[1:-1]
    binned = np.digitize(a, edges)

    # 2. Treat each COMPLETE row as a layer state.
    #
    # Example:
    # [4, 12, 20] and [4, 12, 20] -> same state
    # [4, 12, 20] and [4, 12, 21] -> different states
    states, state_id, counts = np.unique(
        binned,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    # 3. Uniform P(X=x) = 1/4096 gives:
    # P(T=t) = number of inputs mapped to t / 4096.
    p_t = counts / n_inputs

    # 4. Construct P(T,Y) using the true conditional label probabilities.
    #
    # P(T=t,Y=1) = sum_{x: T(x)=t} P(Y=1|x) / 4096.
    p_t_y1 = np.bincount(
        state_id,
        weights=p_y1,
        minlength=len(states),
    ) / n_inputs

    p_t_y0 = np.bincount(
        state_id,
        weights=1 - p_y1,
        minlength=len(states),
    ) / n_inputs

    # Columns correspond to Y=0 and Y=1.
    p_ty = np.column_stack((p_t_y0, p_t_y1))

    # 5. Obtain P(Y) and P(Y|T).
    p_y = p_ty.sum(axis=0)
    p_y_given_t = p_ty / p_t[:, None]

    # 6. Compute entropies.
    h_t = entropy(p_t)
    h_y = entropy(p_y)

    h_y_given_t = sum(
        probability * entropy(conditional)
        for probability, conditional in zip(p_t, p_y_given_t)
    )

    # The binned representation is deterministic given X:
    # H(T|X)=0, hence I(X;T)=H(T).
    i_xt = h_t
    i_ty = h_y - h_y_given_t

    return {
        "I_XT": float(i_xt),
        "I_TY": float(i_ty),
        "n_states": len(states),
        "P_T": p_t,
        "P_TY": p_ty,
        "P_Y_given_T": p_y_given_t,
        "state_id": state_id,
    }


# part 2



def collect_activations(model, X):
    # Fixed [-1, 1] bins are appropriate for your tanh model.
    if model.settings.activation != "tanh":
        raise ValueError(
            "This calculation expects a model trained with tanh. "
            "ReLU requires a different binning range."
        )

    parameter = next(model.parameters())

    inputs = torch.as_tensor(
        X,
        dtype=parameter.dtype,
        device=parameter.device,
    )

    activations = {}
    was_training = model.training

    try:
        model.eval()

        with torch.no_grad():
            # Exclude the final linear classification layer.
            for i, layer in enumerate(model.layers[:-1]):
                inputs = torch.tanh(layer(inputs))

                activations[f"hidden_{i + 1}"] = (
                    inputs.cpu().numpy().copy()
                )

    finally:
        model.train(was_training)

    return activations

# Load all inputs and the underlying label probabilities.
data = loadmat("spherical_binary_dataset.mat")

X = data["F"].astype(np.float32)
p_y1 = data["p_y1"].reshape(-1)

checkpoint_path = "results/fraction-0.03/run-1/checkpoints/last.ckpt"

model = IBNetwork.load_from_checkpoint(
    checkpoint_path,
    map_location="cpu",
)
model.eval()

def mutual_information_per_fraction(X, p_y1, root="results", bins=30):

    root = Path(root)
    X = torch.as_tensor(X, dtype=torch.float32, device="cpu")

    fraction_dirs = sorted(
        (path for path in root.glob("fraction-*") if path.is_dir()),
        key=lambda path: float(path.name.removeprefix("fraction-")),
    )

    if not fraction_dirs:
        raise FileNotFoundError(f"No fraction folders found in {root}")

    results = {}

    for fraction_dir in fraction_dirs:
        fraction = float(fraction_dir.name.removeprefix("fraction-"))

        checkpoint_path = (
            fraction_dir / "run-1" / "checkpoints" / "last.ckpt"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        model = IBNetwork.load_from_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
        )
        model.eval()

        activations = collect_activations(model, X)
        results[fraction] = {}

        print(f"\nTraining fraction: {fraction:.0%}")

        for name, layer_activations in activations.items():
            mi = layer_mutual_information(
                layer_activations,
                p_y1,
                bins=bins,
            )

            results[fraction][name] = {
                "I_XT": mi["I_XT"],
                "I_TY": mi["I_TY"],
            }

            print(
                f"{name}: "
                f"I(X;T) = {mi['I_XT']:.4f}, "
                f"I(T;Y) = {mi['I_TY']:.4f} bits"
            )

    return results


def plot_information_plane(results):
    fractions = sorted(results)

    # Black → purple → red → yellow, similar to your reference.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "training_fraction",
        ["black", "#8000ff", "#ff0000", "#ffff00"],
    )

    norm = mcolors.Normalize(
        vmin=min(fractions),
        vmax=max(fractions),
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    for fraction in fractions:
        layers = results[fraction]

        # Preserve network order: hidden_1, hidden_2, ...
        layer_names = sorted(
            layers,
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )

        i_xt = [layers[name]["I_XT"] for name in layer_names]
        i_ty = [layers[name]["I_TY"] for name in layer_names]

        color = cmap(norm(fraction))

        ax.plot(
            i_xt,
            i_ty,
            color=color,
            linewidth=0.8,
            alpha=0.8,
        )

        ax.scatter(
            i_xt,
            i_ty,
            color=color,
            s=55,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.04,
    )
    colorbar.ax.set_title("Training data", fontsize=13, pad=12)

    ticks = (
        fractions if len(fractions) <= 10
        else np.linspace(min(fractions), max(fractions), 6)
    )
    colorbar.set_ticks(ticks)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    ax.set_xlabel(r"$I(X;T)$ [bits]", fontsize=14)
    ax.set_ylabel(r"$I(T;Y)$ [bits]", fontsize=14)
    ax.set_title("Information plane — one run per training fraction")
    ax.tick_params(direction="in", top=True, right=True)
    ax.margins(x=0.05, y=0.08)

    fig.tight_layout()
    fig.savefig("information_plane.png", dpi=300, bbox_inches="tight")
    plt.show()



results = mutual_information_per_fraction(X, p_y1)


for fraction, layers in sorted(results.items()):
    print(f"\nTraining fraction: {fraction:.0%}")

    for layer_name, mi in layers.items():
        print(
            f"  {layer_name}: "
            f"I(X;T) = {mi['I_XT']:.4f} bits, "
            f"I(T;Y) = {mi['I_TY']:.4f} bits"
        )
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
import numpy as np



plot_information_plane(results)