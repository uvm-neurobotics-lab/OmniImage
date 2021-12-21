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
def mutate(parent, valid, loci, M):
    # how to mutate (only new values)
    mutations = choice(valid, size=M, replace=False)
    # where to mutate
    positions = choice(loci, size=M, replace=False)
    child = np.copy(parent)
    child[positions] = mutations
    return child


@njit
def cross(dad, mom):
    valid = np.array(list(set(dad) | set(mom)))  # cast to np for numba reasons
    child = choice(valid, size=len(dad), replace=False)
    return child


def score(pop, distances, inverse):
    fitnesses = np.array([subset(distances, m).sum() for m in pop])
    if inverse:
        return -fitnesses
    return fitnesses


@njit(parallel=True)
def step(pop, fitnesses, P, S, K, M, O, space):
    indexes = np.argsort(fitnesses)
    fit_idxs = indexes[:K]
    unfit_idxs = indexes[K:]
    top_k = pop[fit_idxs]
    loci = np.arange(S)
    # compute valid indices to sample from during mutation
    valid = np.empty((K, len(space) - S))
    for i in prange(K):
        valid[i] = np.array(list(space - set(top_k[i])))
    children = np.empty(((P - K), S))

    ##### MUTATION ######
    for i in prange(P - K):
        j = i % K
        children[i] = mutate(top_k[j], valid[j], loci, M)

    ##### CROSSOVER ######
    for i in prange(0, P - K - 1):
        if i % O == 0:  # only cross 1/O of the offspring
            # children are arranged in sorted bands so fit[i]>fit[i+1] (apart from in between bands)
            children[i + 1] = cross(children[i], children[i + 1])

    pop[unfit_idxs] = children

    return pop


def select_best(distances, P, S, K, M, O, its, inverse=False, progress=False):
    # distances : [N,L] N vectors of length L
    # P : population size
    # S : subset size
    # K : number of parents (P-K children)
    # M : number of mutations
    # O : fraction of crossovers e.g. O=2 -> 1/2, O=10 -> 1/10, (bigger=faster)
    N = distances.shape[0]
    assert S <= N, f"Error: value of S={S} is > than the #distances={N}"
    assert O >= 1, f"Error: value of O={S} is < 1"
    space = set(range(N))
    pop = np.array([np.arange(S) for _ in range(P)])
    fitnesses = np.zeros(P)

    stats = defaultdict(list)
    range_fun = trange if progress else range
    for _ in range_fun(its):

        pop = step(pop, fitnesses, P, S, K, M, O, space)

        fitnesses = score(pop, distances, inverse)

        best = np.copy(pop[np.argmin(fitnesses)])
        stats["bests"].append(best)
        stats["fits"].append(fitnesses.min())

    fitnesses = score(pop, distances, inverse)
    best = np.copy(pop[np.argmin(fitnesses)])

    return best, stats


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
    best, stats = select_best(
        distances, P=1000, S=20, K=100, M=2, O=2, its=2_000, progress=True
    )
    print("Final fit:", stats["fits"][-1])

    # show_evolution(cls, distances, stats)

    # REPEAT EVOLUTION AFTER REMOVING THE TOP DOGS
    # refined = np.array(
    #     list(set(np.arange(len(distances))) - set(stats["bests"][-1]))
    # )
    # distances2 = distances[np.ix_(refined, refined)]
    # best2, stats2 = select_best(distances2, params, progress=True)
    # show_evolution(distances2, stats2)
    # candidates = np.array(stats["bests"])
    # genotype_over_time(candidates)
    # # steps at which fitness change
    # improvements = improvements_only(stats["fits"])
    # t = np.array([i[0] for i in improvements])
    # candidates = np.array(stats["bests"])[t]
    # genotype_over_time(candidates)
