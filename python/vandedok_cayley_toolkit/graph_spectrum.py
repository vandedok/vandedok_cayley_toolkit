import numpy as np
from .matplotlib_utils import get_fig_ax


def wigner_semicircle(x, R=2):
    y = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= R
    y[mask] = (2 / (np.pi * R**2)) * np.sqrt(R**2 - x[mask] ** 2)
    return y


def normalize_adj_matrix(W):
    # print(n, np.linalg.eigvalsh(W))
    p = np.mean(W)
    variance = p * (1 - p)
    W = (W - p) / np.sqrt(variance * W.shape[0])
    return W


def show_multiple_spectra(spectra, logscale=True, bins=40, figsize=(20, 10)):
    fig, ax = get_fig_ax(None, None, figsize=figsize)
    x = np.linspace(-2.5, 2.5, 1000)
    y = wigner_semicircle(x)

    ax.plot(x, y, "r-", lw=2, label="Wigner semicircle")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Frequency")
    if logscale:
        ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle="--")

    for nl, spectrum in enumerate(spectra):
        ax.hist(
            spectrum,
            bins=bins,
            density=True,
            alpha=0.6,
            label=f"Number of layers = {nl}",
        )

    min_ev = np.stack(spectrum).min()
    max_ev = np.stack(spectrum).max()
    extr_ev = np.max(np.abs([min_ev, max_ev]))
    x_range = np.array([-extr_ev, extr_ev]) * 1.1
    ax.set_xlim(x_range)

    ax.set_title("Eigenvalue distribution, comparison with Wigner law")
    ax.set_xlabel("Value")
    ax.set_ylabel("PDF")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()

    return fig, ax


def show_spectrum(eigenvals, bins=None, figsize=(10, 7), fig=None, ax=None):
    fig, ax = get_fig_ax(fig, ax, figsize=figsize)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linestyle="--")
    min_ev = eigenvals.min()
    max_ev = eigenvals.max()
    extr_ev = np.max(np.abs([min_ev, max_ev]))
    x_range = np.array([-extr_ev, extr_ev]) * 1.1
    ax.set_xlim(x_range)
    ax.hist(eigenvals, bins=bins, color="skyblue", edgecolor="black", align="mid")
    return fig, ax
