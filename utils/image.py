import mimetypes
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List

import cv2
import torch
from numba import i8, u8
from PIL import Image
from tqdm import tqdm


def read_folder(folder: Path) -> List[Path]:
    if type(folder) is str:
        folder = Path(folder)
    return sorted(folder.iterdir())


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
    return sorted(res)


def load(im: Path) -> u8[:, :, :]:
    return cv2.cvtColor(
        cv2.imread(im.as_posix()),
        cv2.COLOR_BGR2RGB,
    )


def load_resize(im: Path, size=64) -> u8[:, :]:
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


def paths2tensors(paths: List[Path], size: i8 = 64):
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
