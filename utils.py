import pickle
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cosines import compute_distances_cuda


def feats2distances(npy):
    feats = np.load(npy)
    distances = compute_distances_cuda(feats)
    images_list = npy.parent / (npy.stem + "_names.pkl")
    images_paths = pickle.load(images_list.open("rb"))
    return distances, images_paths


def parallel_imap_unord(fn, els, chunksize=100, procs=12):
    with Pool(procs) as p:
        for _ in tqdm(p.imap_unordered(fn, els, chunksize=chunksize), total=len(els)):
            pass


def parallel_imap(fn, els, chunksize=100, procs=12):
    with Pool(procs) as p:
        return list(tqdm(p.imap(fn, els, chunksize=chunksize), total=len(els)))


# start = time()
# valid = get_files("./valid", extensions=image_extensions, recurse=True)
# end = time()
# print(f"Files {len(valid)} Elapsed: {end-start:.2f}s")
# print(set([f.suffix for f in valid]))

# start = time()
# train = get_files("./train", extensions=image_extensions, recurse=True)
# end = time()
# print(f"Files {len(train)} Elapsed: {end-start:.2f}s")
# print(set([f.suffix for f in train]))


# parallel_imap_unord(resize64, valid)
# parallel_imap_unord(resize64, train)


def get_size(path):
    try:
        size = Image.open(path).size
        return size
    except:
        print("Failed", path)
        path.unlink()


def downsize(im, size=64):
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


def is_rgb(path):
    return Image.open(path).mode == "RGB"


# GRAVEYARD

# def resize64(im_path):
#     split, cls, file = im_path.parts
#     new_path = Path("/".join([split + "64", cls, file]))
#     if not new_path.exists():
#         im = cv2.imread(im_path.as_posix())
#         shape = im.shape
#         assert len(shape) == 3 and shape[-1] == 3
#         side = min(im.shape[:2])
#         if side > 64:
#             try:
#                 small_im = downsize(im, size=64)
#                 new_path.parent.mkdir(parents=True, exist_ok=True)
#                 cv2.imwrite(new_path.as_posix(), small_im)
#             except Exception as e:
#                 print("Failed", im_path, e)


# def squarify(im_path):
#     split, cls, file = im_path.parts
#     new_path = Path("/".join([split + "_square", cls, file]))
#     if not new_path.exists():
#         im = Image.open(im_path)
#         side = min(*im.size)
#         if side > 64:
#             try:
#                 sq_im = CenterCrop(size=side)(im)
#                 new_path.parent.mkdir(parents=True, exist_ok=True)
#                 sq_im.save(new_path)
#             except:
#                 print("Failed", im_path)

# def paths2df(paths):
#     rows = []
#     for el in tqdm(paths[:10]):
#         _, cls, file = el.parts
#         w, h = Image.open(el).size
#         row = {"cls": cls, "file": file, "path": el, "W": w, "H": h, "WH": (w, h)}
#         rows.append(row)
#     return pd.DataFrame(rows)
