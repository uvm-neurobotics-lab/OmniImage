from collections import defaultdict, namedtuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numba import njit, prange
from numpy.random import choice
from tqdm import trange

from utils import feats2distances


@njit
def subset(matrix, cols):
    """Optimized version of np.ix_ for symmetric mat with zero diag:
    N,P = 10,5
    m = np.random.rand(N, N)
    np.fill_diagonal(m,0)
    a = np.sort(np.random.choice(np.arange(N), size=P, replace=False))
    sub = m[np.ix_(a, a)]
    uptri = np.triu_indices(len(sub), k=1)
    assert np.allclose(sub[uptri],subset(m, a))
    """
    N = cols.shape[0]
    res = np.empty((N * (N - 1)) // 2, dtype=matrix.dtype)
    k = 0
    for i in range(N):
        for j in cols[i + 1 :]:
            res[k] = matrix[cols[i], j]
            k += 1
    return res


@njit
def mutate(parent, space, loci, num_muts=2):
    valid = np.array(list(space - set(parent)))
    mutations = choice(valid, size=num_muts, replace=False)
    positions = choice(loci, size=num_muts, replace=False)
    child = np.copy(parent)
    child[positions] = mutations
    return child


@njit
def cross(dad, mom):
    valid = np.array(list(set(dad) | set(mom)))
    child = choice(valid, size=len(dad), replace=False)
    return child


# top_k[0], mutate(top_k[0], num_muts)
# top_k[0], top_k[1], cross(top_k[0], top_k[1])


@njit(parallel=True)
def evolve(top_k, params, space):
    # k, M = top_k.shape
    loci = np.arange(params.M)
    children = np.empty(((params.N - params.k), params.M))
    # mutate top_k to get children
    for i in prange(params.N - params.k):
        children[i] = mutate(top_k[i % params.k], space, loci, params.muts)
    # cross pairs of children, keep the first one (more fit), update the second
    for i in range(0, params.N - params.k - 1, 2):
        children[i + 1] = cross(children[i], children[i + 1])
    return children


def score(pop, distances):
    # return np.array([distances[np.ix_(m, m)].sum() for m in pop])
    return np.array([subset(distances, m).sum() for m in pop])


def step(pop, distances, params, space):
    fitnesses = score(pop, distances)
    indexes = np.argsort(fitnesses)
    good_idxs = indexes[: params.k]
    bad_idxs = indexes[params.k :]
    top_k = np.copy(pop[good_idxs])
    children = evolve(top_k, params, space)
    pop[bad_idxs] = children
    return pop, fitnesses


def select_best(distances, params, progress=False):

    space = set(np.arange(distances.shape[0]))
    pop = np.array(
        [
            np.sort(choice(list(space), size=params.M, replace=False))
            for _ in range(params.N)
        ]
    )

    stats = defaultdict(list)
    if progress:
        for _ in trange(params.its):
            pop, fitnesses = step(pop, distances, params, space)
            stats["fits"].append(fitnesses.min())
            best = pop[np.argmin(score(pop, distances))].copy()
            stats["top_dogs"].append(best)
    else:
        for _ in range(params.its):
            pop, fitnesses = step(pop, distances, params, space)
            stats["fits"].append(fitnesses.min())
            best = pop[np.argmin(score(pop, distances))].copy()
            stats["top_dogs"].append(best)

    best = pop[np.argmin(score(pop, distances))]

    return best, stats


def trans(l):
    return list(zip(*l))


def improvements_only(fitnesses):
    improvements = [
        (i, a) for i, (a, b) in enumerate(zip(fitnesses, fitnesses[1:])) if a > b
    ]
    improvements.append((len(fitnesses) - 1, fitnesses[-1]))
    return improvements


def show_evolution(distances, stats):
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
        axin.set_axis_off()
        # axin.set_title(f"{f:.1f}",fontsize=8,loc="right")
    ax.set_xlabel("Generation", fontsize=16)
    ax.set_ylabel("Min. Distance", fontsize=16)
    ax.grid()
    size = 0.7
    axin = inset_axes(
        ax,
        width="50%",
        height="80%",
        # bbox_transform=ax.transAxes,
        # bbox_to_anchor=(1-size, 1-size, size, size),
        loc="upper right",
    )
    axin.imshow(distances, vmin=vmin, vmax=vmax, interpolation=None, cmap=cmap)
    best = set(top_dogs[-1])
    for l in best:
        axin.axvline(l, lw=0.5, color="red", label="last")
        axin.axhline(l, lw=0.5, color="red")
    best = set(top_dogs[0])
    for l in best:
        axin.axvline(l, lw=0.5, color="yellow", label="first")
        axin.axhline(l, lw=0.5, color="yellow")
    # from https://stackoverflow.com/a/26339101 REMOVE DUPLICATES IN LEGEND
    handles, labels = axin.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    axin.legend(newHandles, newLabels)
    axin.set_xlabel("Full distance matrix")
    plt.tight_layout()
    # plt.savefig("evol_progress.pdf")
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


def path2np(im, size=64):
    return cv2.cvtColor(
        cv2.resize(
            cv2.imread(im.as_posix()),
            (size, size),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_BGR2RGB,
    )


def show_sample(ims, selected):
    paths = [im for im in np.array(ims)[selected]]
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(14, 10))
    for p, ax in zip(paths, axs.flatten()):
        im = path2np(p)
        ax.imshow(im)
        ax.set_title(p.name)
        ax.set_axis_off()
    plt.show()


Params = namedtuple("Params", "N M k muts its")

#%%

if __name__ == "__main__":

    # cls = "n03047690" # shoe?
    cls = "n11939491" # daisy
    npy = Path(f"torchvision.models.vgg_ce631fc9ca0278a2/{cls}.npy")
    distances, ims = feats2distances(npy)

    # N = 1000  # population size
    # M = 20  # dna size
    # k = 200  # how many get to fuck (out of N)
    # muts = M // 4
    params = Params(N=1000, M=20, k=100, muts=2, its=2_000)
    best, stats = select_best(distances, params, progress=True)

    print(stats["fits"][-1])

    #%%

    from process import paths2tensors
    from torchvision.utils import make_grid
    from random import shuffle
    import matplotlib.patches as patches


    def show_stage( cls, selected=None,root="imagenet_dir"):
        omni64_dir = Path(root)
        cls_dir = omni64_dir/cls
        if selected is None:
            ims_paths = list(cls_dir.iterdir())
            shuffle(ims_paths)
            ims_paths = ims_paths[:20]
        else:
            ims_paths = np.array(sorted(cls_dir.iterdir()))[selected]
        tens = paths2tensors(ims_paths,size=64)
        grid = make_grid(tens, nrow=4)
        return grid.numpy().transpose(1,2,0)

    # plt.imshow(grid.numpy().transpose(1,2,0))
    # plt.tight_layout()
    # plt.axis("off")
    # plt.show()

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
        rect = patches.Rectangle((-0.5, -0.5), 20, 20, linewidth=0.5, edgecolor='white', facecolor='none')
        # Add the patch to the Axes
        axin.add_patch(rect)
        axin.set_axis_off()
        # axin.set_title(f"{f:.1f}",fontsize=8,loc="right")
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
    final = show_stage(cls, top_dogs[-1])
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
    random = show_stage(cls)
    axin.imshow(random)
    axin.set_xlabel("Random", fontsize=16)
    axin.set_xticks([])
    axin.set_yticks([])
    plt.tight_layout()
    plt.savefig("evol_progress.pdf")
    plt.show()



    #%%



    show_evolution(distances, stats)
    show_sample(ims, stats["top_dogs"][-1])

    refined = np.array(
        list(set(np.arange(len(distances))) - set(stats["top_dogs"][-1]))
    )
    distances2 = distances[np.ix_(refined, refined)]

    best2, stats2 = select_best(distances2, params, progress=True)

    show_evolution(distances2, stats2)
    show_sample(ims, stats2["top_dogs"][-1])

    # [is_rgb_pil(p) for p in ims] == [is_rgb(p) for p in ims]
    # %timeit all([is_rgb_pil(p) for p in ims])

    # # steps at which fitness change
    # t = np.array([i[0] for i in improvements])
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.diff(np.sort(np.array(top_dogs)[t], axis=1), axis=0))
    # plt.tight_layout()
    # plt.show()

    candidates = np.array(stats["top_dogs"])
    genotype_over_time(candidates)

    # steps at which fitness change
    improvements = improvements_only(stats["fits"])
    t = np.array([i[0] for i in improvements])
    candidates = np.array(stats["top_dogs"])[t]
    genotype_over_time(candidates)
