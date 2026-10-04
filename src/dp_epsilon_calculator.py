import math
import numpy as np


def compute_epsilon_gaussian_rdp(sigma=0.3, steps=30, delta=1e-5):
    """
    Approximate RDP accountant for the Gaussian mechanism without subsampling.
    Assumption: all clients participate in every communication round (q = 1).
    """

    orders = np.arange(2, 256)
    eps_values = []

    for alpha in orders:
        # RDP of Gaussian mechanism per step
        rdp = steps * alpha / (2 * sigma ** 2)

        # Convert RDP to (epsilon, delta)-DP
        epsilon = rdp + math.log(1 / delta) / (alpha - 1)
        eps_values.append(epsilon)

    best_index = int(np.argmin(eps_values))
    best_epsilon = eps_values[best_index]
    best_alpha = orders[best_index]

    return best_epsilon, delta, best_alpha


if __name__ == "__main__":
    sigma = 1.1
    steps = 30
    delta = 1e-5

    epsilon, delta, alpha = compute_epsilon_gaussian_rdp(
        sigma=sigma,
        steps=steps,
        delta=delta
    )

    print("DP Privacy Budget")
    print("=================")
    print(f"Noise multiplier sigma: {sigma}")
    print(f"Communication rounds: {steps}")
    print(f"Delta: {delta}")
    print(f"Epsilon: {epsilon:.2f}")
    print(f"Best alpha: {alpha}")
    print(f"Privacy budget: ({epsilon:.2f}, {delta})-DP")