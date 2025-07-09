import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_fig_ax(fig, ax, **kwargs):
    if ax is None:
        assert fig is None, "fig and ax must be both non or not None"
        fig, ax = plt.subplots(**kwargs)
        return fig, ax
    else:
        assert fig is not None, "fig and ax must be both non or not None"
        return fig, ax


def lighten(color, factor=1.0):
    r, g, b = mcolors.to_rgb(color)
    return (
        (1 - factor) * 1 + factor * r,
        (1 - factor) * 1 + factor * g,
        (1 - factor) * 1 + factor * b,
    )
