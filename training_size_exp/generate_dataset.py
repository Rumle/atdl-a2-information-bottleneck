import numpy as np
from scipy.special import eval_legendre, expit, xlogy
from scipy.optimize import brentq
from scipy.io import savemat


def binary_entropy(p):
    """Binary entropy in bits; handles probabilities 0 and 1."""
    p = np.asarray(p, dtype=np.float64)

    return -(
        xlogy(p, p) + xlogy(1 - p, 1 - p)
    ) / np.log(2)


def sphere_points():
    """12 evenly arranged points: vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2

    points = np.array(
        [(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)]
        + [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)]
        + [(b * phi, 0, a) for a in (-1, 1) for b in (-1, 1)],
        dtype=np.float64,
    )

    # Normalize each point to lie on the unit sphere.
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def power_spectrum(X, points, max_degree=5):
    """Calculate rotation/reflection-invariant spherical harmonic powers."""
    dots = np.clip(points @ points.T, -1, 1)
    powers = []

    for degree in range(max_degree + 1):
        # Spherical harmonic addition theorem:
        #
        # E_l(x) = (2l+1)/(4*pi) *
        #          sum_ij x_i*x_j*P_l(point_i dot point_j)
        #
        # P_l is a Legendre polynomial.
        kernel = eval_legendre(degree, dots)

        energy = (
            (2 * degree + 1)
            / (4 * np.pi)
            * np.einsum("bi,ij,bj->b", X, kernel, X)
        )

        powers.append(energy)

    return np.column_stack(powers)


def generate_dataset(seed=42):
    # 1. Generate every possible 12-bit input.
    # Shape: (4096, 12)
    numbers = np.arange(2**12)
    bit_positions = np.arange(11, -1, -1)

    X = (
        (numbers[:, None] >> bit_positions) & 1
    ).astype(np.uint8)

    # Every input pattern is equally likely.
    p_x = np.full(len(X), 1 / len(X))

    # 2. Associate the 12 input features with points on the sphere.
    points = sphere_points()

    # 3. Calculate invariant power-spectrum features.
    powers = power_spectrum(X, points)

    # 4. Define our scalar function f(x).
    # These coefficients are our choice, not specified in the paper.
    coefficients = np.array([1.0, 0.37, -0.23, 0.19, 0.11, -0.07])

    raw_score = powers @ coefficients
    score_center = raw_score.mean()
    score_scale = raw_score.std()
    f = (raw_score - score_center) / score_scale

    # 5. Choose a threshold giving approximately balanced classes.
    # Group numerically equal scores and place theta between score levels.
    levels, counts = np.unique(
        np.round(f, decimals=10),
        return_counts=True,
    )

    cumulative_fraction = np.cumsum(counts)[:-1] / len(X)
    cut = np.argmin(np.abs(cumulative_fraction - 0.5))

    theta = (levels[cut] + levels[cut + 1]) / 2

    # 6. Calculate MI for a candidate sigmoid gain.
    def mutual_information(gamma):
        p = expit(gamma * (f - theta))

        h_y = binary_entropy(p.mean())
        h_y_given_x = binary_entropy(p).mean()

        return float(h_y - h_y_given_x)

    # Find a search interval that reaches the target MI.
    target_mi = 0.99
    upper = 1.0

    while mutual_information(upper) < target_mi:
        upper *= 2

        if upper > 1e12:
            raise ValueError("Cannot reach the target MI with this rule.")

    # Solve I(X;Y) = 0.99 bits.
    gamma = brentq(
        lambda g: mutual_information(g) - target_mi,
        0,a<
        upper,
        xtol=1e-12,
    )

    # 7. True conditional probabilities P(Y=1 | X=x).
    p_y1 = expit(gamma * (f - theta))

    # Deterministic labels before softening.
    y_hard = (f >= theta).astype(np.uint8)

    # Sample one binary training label per input.
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, p_y1).astype(np.uint8)

    # Joint distribution: columns correspond to Y=0 and Y=1.
    p_xy = np.column_stack([
        p_x * (1 - p_y1),
        p_x * p_y1,
    ])

    # 8. Validate the generated distribution.
    assert np.unique(X, axis=0).shape == (4096, 12)
    assert abs(p_y1.mean() - 0.5) < 0.01
    assert abs(mutual_information(gamma) - target_mi) < 1e-10
    assert np.isclose(p_xy.sum(), 1)

    # 9. Save inputs, labels, probabilities, and generation parameters.
    data = {
        "F": X,
        "y": y,
        "p_y1": p_y1,
        "y_hard": y_hard,
        "p_x": p_x,
        "p_xy": p_xy,
        "points": points,
        "spectrum": powers,
        "f": f,
        "theta": theta,
        "gamma": gamma,
        "coefficients": coefficients,
        "score_center": score_center,
        "score_scale": score_scale,
        "seed": seed,
    }

    np.savez_compressed("spherical_binary_dataset.npz", **data)

    savemat(
        "spherical_binary_dataset.mat",
        data,
        do_compression=True,
        oned_as="column",
    )

    print(f"Input shape: {X.shape}")
    print(f"P(Y=1):      {p_y1.mean():.8f}")
    print(f"I(X;Y):      {mutual_information(gamma):.8f} bits")
    print(f"Theta:       {theta:.8f}")
    print(f"Gamma:       {gamma:.8f}")

    return data


if __name__ == "__main__":
    generate_dataset()