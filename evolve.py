from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.random import choice
from tqdm import trange

from cosines import feats2distances
from utils.image import class2paths


@njit
def subset(matrix, cols):
    """
    Optimized version of np.ix_ for symmetric mat with zero diag
    
    N,C = 5,3 # N size of matrix, C columns selected
    m = np.random.rand(N, N)
    m = (m + m.T)/2 # symmetric
    np.fill_diagonal(m,0)
    select = np.sort(np.random.choice(np.arange(N), size=C, replace=False))
    sub = m[np.ix_(select, select)]
    triu = np.triu_indices(len(sub), k=1)
    assert np.allclose(sub[triu], subset(m, select))
    
    e.g. N = 5, C = 3
    m:
          | 0  a  b  c  d |
          | a  0  e  f  g |
          | b  e  0  h  i |
          | c  f  h  0  j |
          | d  g  i  j  0 |
            
    select: [1,3,4]
    
    sub:
          | 0  f  g |
          | f  0  j |
          | g  j  0 |
            
    sub[triu]: [f, g, j] # len = 3*(3-1)//2
    """
    C = cols.shape[0]
    res = np.empty((C * (C - 1)) // 2, dtype=matrix.dtype)
    k = 0
    for i in range(C):
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


def step(pop, fitnesses, params, space):
    indexes = np.argsort(fitnesses)
    good_idxs = indexes[: params.k]
    bad_idxs = indexes[params.k :]
    top_k = np.copy(pop[good_idxs])
    children = evolve(top_k, params, space)
    pop[bad_idxs] = children
    return pop


def select_best(distances, params, inverse=False, progress=False):

    space = set(np.arange(distances.shape[0]))
    pop = np.array(
        [
            np.sort(choice(list(space), size=params.M, replace=False))
            for _ in range(params.N)
        ]
    )

    stats = defaultdict(list)
    range_fun = trange if progress else range
    for _ in range_fun(params.its):
        fitnesses = score(pop, distances)
        if inverse:
            fitnesses *= -1

        best = pop[np.argmin(fitnesses)].copy()
        stats["top_dogs"].append(best)
        stats["fits"].append(fitnesses.min())

        pop = step(pop, fitnesses, params, space)

    fitnesses = score(pop, distances)
    if inverse:
        fitnesses *= -1
    best = pop[np.argmin(fitnesses)].copy()

    return best, stats


Params = namedtuple("Params", "N M k muts its")

#%%

if __name__ == "__main__":

    from utils.viz import genotype_over_time, improvements_only, show_evolution

    # cls = "n03047690" # shoe?
    cls = "n11939491"  # daisy
    npy = Path(f"torchvision.models.vgg_ce631fc9ca0278a2/{cls}.npy")
    distances = feats2distances(npy)
    ims = class2paths(cls)

    # N = 1000  # population size
    # M = 20  # dna size
    # k = 200  # how many get to fuck (out of N)
    # muts = M // 4
    params = Params(N=1000, M=20, k=100, muts=2, its=2_000)
    best, stats = select_best(distances, params, progress=True)

    print("Final fit:", stats["fits"][-1])

    show_evolution(cls, distances, stats)

    # REPEAT EVOLUTION AFTER REMOVING THE TOP DOGS
    # refined = np.array(
    #     list(set(np.arange(len(distances))) - set(stats["top_dogs"][-1]))
    # )
    # distances2 = distances[np.ix_(refined, refined)]
    # best2, stats2 = select_best(distances2, params, progress=True)
    # show_evolution(distances2, stats2)
    # candidates = np.array(stats["top_dogs"])
    # genotype_over_time(candidates)
    # # steps at which fitness change
    # improvements = improvements_only(stats["fits"])
    # t = np.array([i[0] for i in improvements])
    # candidates = np.array(stats["top_dogs"])[t]
    # genotype_over_time(candidates)
