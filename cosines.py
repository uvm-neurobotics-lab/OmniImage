import math
import unittest
from itertools import combinations
from pathlib import Path
from random import choice

import numpy as np
from numba import cuda, f4, njit, prange
from scipy.spatial.distance import cosine


def cosine_distance(u: np.ndarray, v: np.ndarray):
    """
    ..math:: \mbox{Cosine Distance} = 1 - \frac{\mathbf{U}\mathbf{V}}{\lVert\mathbf{U}\rVert\lVert\mathbf{V}\rVert}
    """
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = 1 - uv / math.sqrt(uu * vv)
    return cos_theta


cosine_numba = njit(fastmath=True, cache=True)(cosine_distance)
cosine_cuda = cuda.jit(device=True)(cosine_distance)


@njit(parallel=True, cache=True)
def compute_distances_parallel(features):
    """
    Compute pairwise cosine distances in parallel, using numba.
    data    : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    N = features.shape[0]
    distances = np.zeros((N, N))
    for row in prange(N):
        for other in prange(row + 1):
            d = cosine_numba(features[row], features[other])
            distances[row, other] = d
            distances[other, row] = d
    return distances


# set fastmath for the sqrt in the device function
@cuda.jit(fastmath=True)
def self_cosine_kernel(features, distances):
    """
    Compute one entry of the distances matrix
    features  : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    i, j = cuda.grid(2)
    if i < features.shape[0] and j <= i:
        sim = cosine_cuda(features[i], features[j])
        distances[i, j] = sim
        distances[j, i] = sim


def self_cosine_distances_cuda(features):
    """
    Compute pairwise cosine distances with cuda acceleration.
    features  : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    features = features.astype(np.float32)
    N = features.shape[0]
    distances = np.zeros((N, N), dtype=np.float32)
    features_cuda = cuda.to_device(features)
    distances_cuda = cuda.to_device(distances)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    self_cosine_kernel[blockspergrid, threadsperblock](features_cuda, distances_cuda)
    distances = distances_cuda.copy_to_host()
    return distances


def feats2distances(npy: Path) -> f4[:]:
    feats = np.load(npy)
    distances = cosine_distances_cuda(feats)
    return distances


def cosine_distances_cuda(v1, v2=None):
    """
    Compute pairwise cosine distances with cuda acceleration.
    v1        : (N,K) features matrix
    v2        : (M,K) features matrix
    distances : (N,M) distances matrix
    """
    if v2 is None:
        return self_cosine_distances_cuda(v1)
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    distances = np.zeros((v1.shape[0], v2.shape[0]), dtype=np.float32)
    v1_cuda = cuda.to_device(v1)
    v2_cuda = cuda.to_device(v2)
    distances_cuda = cuda.to_device(distances)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(v1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(v2.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cosine_kernel2[blockspergrid, threadsperblock](v1_cuda, v2_cuda, distances_cuda)
    distances = distances_cuda.copy_to_host()
    return distances


# set fastmath for the sqrt in the device function
@cuda.jit(fastmath=True)
def cosine_kernel2(v1, v2, distances):
    """
    Compute one entry of the distances matrix
    v1        : (N,K) features matrix
    v2        : (M,K) features matrix
    distances : (N,M) distances matrix
    """
    i, j = cuda.grid(2)
    if i < v1.shape[0] and j < v2.shape[0]:
        sim = cosine_cuda(v1[i], v2[j])
        distances[i, j] = sim


#%%


def scipy_cosine(v1, v2):
    N, M = v1.shape[0], v2.shape[0]
    ret = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            ret[i, j] = cosine(v1[i], v2[j])
    return ret


def comparer(ds, atol):
    return [np.allclose(a, b, atol=atol) for a, b in combinations(ds, 2)]


class TestCosines(unittest.TestCase):
    atol = 1e-6

    def test_rect(self):

        for _ in range(100):
            N = choice(range(2, 7))
            M = choice(range(2, 7))
            K = choice(range(3, 10))

            v1 = np.random.random((N, K))
            v2 = np.random.random((M, K))

            d1 = scipy_cosine(v1, v2)
            d2 = cosine_distances_cuda(v1, v2)

            comparison = np.allclose(d1, d2, atol=self.atol)
            # print(N, M, K, comparison)
            # print(np.abs(d1 - d2).mean())
            self.assertTrue(comparison)

    def test_square(self):

        for _ in range(100):
            N = choice(range(2, 7))
            K = choice(range(2, 100))

            v1 = np.random.random((N, K))

            ds = [
                scipy_cosine(v1, v1),
                cosine_distances_cuda(v1),
                cosine_distances_cuda(v1, v1),
                self_cosine_distances_cuda(v1),
            ]

            comparisons = comparer(ds, atol=self.atol)
            # print(np.mean([np.abs(a - b).mean() for a, b in combinations(ds, 2)]))
            # print(N, K, comparisons)
            self.assertTrue(all(comparisons))


if __name__ == "__main__":
    unittest.main()
