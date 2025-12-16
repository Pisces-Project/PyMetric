# noqa: D100
import matplotlib.pyplot as plt
import numpy as np

import pymetric as pym

# Define spherical coordinate system and grid
cs = pym.coordinates.SphericalCoordinateSystem()
grid = pym.grids.GenericGrid(
    cs,
    [
        np.linspace(0.1, 4.9, 300),  # r
        np.linspace(0.01, np.pi - 0.01, 100),  # θ
        np.linspace(0.01, 2 * np.pi - 0.01, 100),  # φ
    ],
    center="cell",
    bbox=[(0, 5), (0, np.pi), (0, 2 * np.pi)],
    ghost_zones=2,
)

# Define scalar field F(r, θ) = r * cos(θ)
field = pym.DenseField.from_function(
    lambda r, theta: r * np.cos(theta),
    grid,
    axes=["r", "theta"],
)

# Compute Laplacian and trim ghost zones
F = field[2:-2, 2:-2]
F_lap = field.element_wise_laplacian()[2:-2, 2:-2]
R, Theta = grid.compute_domain_mesh(axes=["r", "theta"], origin="active")

# Plot field and Laplacian
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"hspace": 0.0})
vmin, vmax = -5, 5
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = "seismic"

axes[0].pcolormesh(R, Theta, F, cmap=cmap, norm=norm)
axes[0].set_ylabel(r"$\theta$")
axes[0].text(
    0.5,
    0.9,
    r"$F(r,\theta) = r\cos(\theta)$",
    ha="center",
    va="bottom",
    transform=axes[0].transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.7),
)

axes[1].pcolormesh(R, Theta, F_lap, cmap=cmap, norm=norm)
axes[1].set_xlabel(r"$r$")
axes[1].set_ylabel(r"$\theta$")
axes[1].text(
    0.5,
    0.9,
    r"$\nabla^2 F(r,\theta)$",
    ha="center",
    va="bottom",
    transform=axes[1].transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.7),
)

cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    orientation="vertical",
    fraction=0.04,
    pad=0.03,
)
cbar.set_label(r"$F(r,\theta)$ and $\nabla^2 F(r,\theta)$")

plt.savefig("fig1.png", dpi=600)
