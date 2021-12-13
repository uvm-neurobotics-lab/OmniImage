import mimetypes
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm import tqdm


def file2paths(file="OmniImage.txt"):
    with open(file, "r") as f:
        paths = f.read().splitlines()
        paths = [Path(p) for p in paths]
    return paths


def class2paths(cls, file="OmniImage.txt"):
    paths = file2paths(file)
    return sorted([path for path in paths if path.parent.name is cls])


def read_folder(folder):
    if type(folder) is str:
        folder = Path(folder)
    return sorted(folder.iterdir())


def path2np(im, size=64):
    return cv2.cvtColor(
        cv2.resize(
            cv2.imread(im.as_posix()),
            (size, size),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_BGR2RGB,
    )


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


def get_images(path):
    def _get_files(p, fs, extensions=None):
        res = [
            p / f
            for f in fs
            if not f.startswith(".")
            and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
        ]
        return res

    extensions = set(
        k for k, v in mimetypes.types_map.items() if v.startswith("image/")
    )
    res = []
    for p, d, f in os.walk(path):
        res += _get_files(Path(p), f, extensions)
    return res


def load(im):
    return cv2.cvtColor(
        cv2.imread(im.as_posix()),
        cv2.COLOR_BGR2RGB,
    )


def load_resize(im, size=64):
    # ugly AF but fast
    # NOTE: cv2 reads images as BGR so need to convert
    return cv2.cvtColor(
        cv2.resize(
            cv2.imread(im.as_posix()),
            (size, size),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_BGR2RGB,
    )


def paths2tensors(paths, size=None):
    # ugly AF but 3 times faster than Image.open
    # toten = torchvision.transforms.ToTensor()
    # pil = torch.stack([toten(Image.open(im)) for im in tqdm(paths)])
    # NOTE: need to transpose chan. dim. to front
    if size is None:
        tens = torch.stack(
            [torch.from_numpy(load(im).transpose(2, 0, 1)) / 255 for im in paths]
        )
    else:
        tens = torch.stack(
            [
                torch.from_numpy(load_resize(im, size=size).transpose(2, 0, 1)) / 255
                for im in paths
            ]
        )
    return tens


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
