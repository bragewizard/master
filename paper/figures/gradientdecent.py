import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def franke_function(x, y):
    """Computes the Franke Function."""
    term1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4.0 - (9 * y - 2) ** 2 / 4.0)
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - (9 * y + 1) / 10.0)
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - (9 * y - 3) ** 2 / 4.0)
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def franke_gradient(x, y):
    """Computes the analytical gradient (df/dx, df/dy) of the Franke Function."""
    # Common exponential terms
    e1 = np.exp(-((9 * x - 2) ** 2) / 4.0 - (9 * y - 2) ** 2 / 4.0)
    e2 = np.exp(-((9 * x + 1) ** 2) / 49.0 - (9 * y + 1) / 10.0)
    e3 = np.exp(-((9 * x - 7) ** 2) / 4.0 - (9 * y - 3) ** 2 / 4.0)
    e4 = np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)

    # Partial derivatives wrt x
    df_dx = (
        0.75 * e1 * (-4.5 * (9 * x - 2))
        + 0.75 * e2 * (-18 / 49 * (9 * x + 1))
        + 0.5 * e3 * (-4.5 * (9 * x - 7))
        - 0.2 * e4 * (-18 * (9 * x - 4))
    )

    # Partial derivatives wrt y
    df_dy = (
        0.75 * e1 * (-4.5 * (9 * y - 2))
        + 0.75 * e2 * (-0.9)
        + 0.5 * e3 * (-4.5 * (9 * y - 3))
        - 0.2 * e4 * (-18 * (9 * y - 7))
    )

    return df_dx, df_dy


# 1. Setup Data
x_range = np.linspace(0, 1, 100)
y_range = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = franke_function(X, Y)

# 2. Choose a point and calculate its gradient
px, py = 0.3, 0.3
pz = franke_function(px, py)
gx, gy = franke_gradient(px, py)

# 3. Plotting Configuration
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "Geist Mono",  # Change to 'sans-serif' or specific font as needed
        "font.weight": "medium",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
    }
)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
surf = ax.plot_surface(
    X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none", antialiased=True
)

# Plot the point
ax.scatter([px], [py], [pz], color="red", s=100, label=f"Point ({px}, {py})", zorder=5)

# Plot the gradient vector
# The gradient vector (gx, gy) lives in the xy-plane.
# Here we plot it as a 3D vector (gx, gy, 0) starting from (px, py, pz)
scale = 0.05  # Scale length for visibility
ax.quiver(
    px,
    py,
    pz,
    gx,
    gy,
    0,
    color="red",
    length=0.15,
    arrow_length_ratio=0.3,
    linewidth=2,
    label="Gradient Vector",
)

# Formatting
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("f(X, Y)")
ax.set_title("Franke Function with Gradient")
ax.view_init(elev=30, azim=220)  # Change angle to best see the gradient

# 4. Export as SVG
plt.savefig("franke_function_3d.svg", format="svg", bbox_inches="tight")
plt.show()
