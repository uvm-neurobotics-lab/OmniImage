import math
import numpy as np
from numba import cuda, njit, prange


def flatten(l):
    return [i for seq in l for i in seq]


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
def cosine_kernel(features, distances):
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


def compute_distances_cuda(features):
    """
    Compute pairwise cosine distances with cuda acceleration.
    features  : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    features = features.astype(np.float32)
    distances = np.zeros((features.shape[0], features.shape[0]), dtype=np.float32)
    d_features = cuda.to_device(features)
    d_distances = cuda.to_device(distances)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(features.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(features.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cosine_kernel[blockspergrid, threadsperblock](d_features, d_distances)
    distances = d_distances.copy_to_host()
    return distances


def feats2distances(npy):
    feats = np.load(npy)
    distances = compute_distances_cuda(feats)
    # images_list = npy.parent / (npy.stem + "_names.pkl")
    # images_paths = pickle.load(images_list.open("rb"))
    # images_paths = read_folder(npy.stem)
    return distances


#%%

if __name__ == "__main__":
    pass
