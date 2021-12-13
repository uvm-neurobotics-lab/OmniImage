import pickle
from pathlib import Path
from random import shuffle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torchvision.utils import make_grid

from .image import path2np, paths2tensors, read_folder


def get_best_fits(stats="statsV3"):
    stats_folder = Path(stats)
    fits = [
        (pkl.stem, pickle.load(pkl.open("rb"))["fits"])
        for pkl in stats_folder.glob("*.pkl")
    ]
    best_fits = [(cls, min(fit)) for cls, fit in fits]
    best_fits = sorted(best_fits, key=lambda x: x[1])
    return best_fits


def make_overview(
    root="OmniImage64", title="overview_consistent.png", best_fits=None, side=31
):
    import matplotlib.pyplot as plt
    import torch
    from im_utils import paths2tensors
    from torchvision.utils import make_grid

    omni64_dir = Path(root)
    if best_fits is not None:
        omni64_selection = [omni64_dir / cls for cls, _ in best_fits][: side * side]
    else:
        omni64_selection = [
            omni64_dir / cls.name for cls in sorted(omni64_dir.iterdir())
        ][: side * side]
    tens1 = [paths2tensors(cls.iterdir()) for cls in omni64_selection]
    blocks1 = torch.stack([make_grid(ten, nrow=5, padding=0) for ten in tens1])
    grid = make_grid(blocks1, nrow=31)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(title, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close()


def show_evolution(cls, distances, stats, save=False):
    """
    cls = "n11939491"  # daisy
    npy = Path(f"torchvision.models.vgg_ce631fc9ca0278a2/{cls}.npy")
    distances = feats2distances(npy)

    # N = 1000  # population size
    # M = 20  # dna size
    # k = 200  # how many get to fuck (out of N)
    # muts = M // 4
    params = Params(N=1000, M=20, k=100, muts=2, its=2_000)
    best, stats = select_best(distances, params, progress=True)

    show_evolution(cls, distances, stats)
    """
    vmin, vmax = distances.min(), distances.max()
    fits = stats["fits"]
    top_dogs = stats["top_dogs"]
    n = len(fits)
    maxfit = max(fits)
    improvements = improvements_only(fits)
    xs, ys = trans(improvements)
    cmap = "viridis"
    # plotting
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(stats["fits"])
    ax.scatter(xs, ys, color="red", alpha=0.5, s=10)
    for i, f in improvements[1::2]:
        axin = inset_axes(
            ax,
            width=0.4,
            height=0.4,
            bbox_transform=ax.transData,
            bbox_to_anchor=(i, f, 1, 1),
            loc="lower left",
        )
        sol = distances[np.ix_(top_dogs[i], top_dogs[i])]
        axin.imshow(sol, vmin=vmin, vmax=vmax, interpolation=None, cmap=cmap)
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (-0.5, -0.5), 20, 20, linewidth=0.5, edgecolor="white", facecolor="none"
        )
        # Add the patch to the Axes
        axin.add_patch(rect)
        axin.set_axis_off()
    ax.set_xlabel("Generation", fontsize=16)
    ax.set_ylabel("Min. Distance", fontsize=16)
    ax.grid()
    size = 0.4
    axin = inset_axes(
        ax,
        width="30%",
        height="60%",
        loc="upper right",
    )
    paths = get_imagenet_class_paths(cls, selected=top_dogs[-1])
    final = paths2grid(paths)
    axin.imshow(final)
    axin.set_xlabel("Best", fontsize=16)
    axin.set_xticks([])
    axin.set_yticks([])
    axin = inset_axes(
        ax,
        width="90%",
        height="60%",
        loc="upper right",
    )
    paths = get_imagenet_class_paths(cls)
    random = paths2grid(paths)
    axin.imshow(random)
    axin.set_xlabel("Random", fontsize=16)
    axin.set_xticks([])
    axin.set_yticks([])
    plt.tight_layout()
    if save:
        plt.savefig("evol_progress.pdf")
    plt.show()


def genotype_over_time(candidates, grid=False):
    sorted_values = sorted(list(set([el for el in candidates.flatten()])))
    val2ord = {x: i for i, x in enumerate(sorted_values)}
    ord_candidates = np.vectorize(lambda x: val2ord[x])(candidates)
    plt.figure(figsize=(16, 10))
    for i, y in enumerate(np.sort(ord_candidates, axis=1).T):
        plt.plot(y)
    if grid:
        for i in range(len(ord_candidates)):
            plt.axvline(i, lw=0.5, alpha=0.2, color="black")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Genotype", fontsize=12)
    plt.tight_layout()
    plt.show()


def improvements_only(fitnesses):
    improvements = [
        (i, a) for i, (a, b) in enumerate(zip(fitnesses, fitnesses[1:])) if a > b
    ]
    improvements.append((len(fitnesses) - 1, fitnesses[-1]))
    return improvements


def trans(l):
    return list(zip(*l))


def get_imagenet_class_paths(cls, selected=None, imagenet_dir="imagenet_dir"):
    ims_paths = read_folder(Path(imagenet_dir) / cls)
    if selected is None:
        shuffle(ims_paths)
        ims_paths = ims_paths[:20]
    else:
        ims_paths = np.array(ims_paths)[selected]
    return ims_paths


def paths2grid(ims_paths):
    tens = paths2tensors(ims_paths, size=64)
    grid = make_grid(tens, nrow=4)
    return grid.numpy().transpose(1, 2, 0)
