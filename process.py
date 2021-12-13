import mimetypes
import os
import pickle
from hashlib import blake2b
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vgg19_bn
from tqdm import tqdm

from evolve import select_best, show_evolution, Params
from hooker import hook_it_up
from cosines import compute_distances_cuda
from utils import downsize, feats2distances


def model2name(net):
    return f"{net.__module__.split}_{weights2hash(net)}"


def par2bytes(p):
    return p.detach().cpu().numpy().tobytes()


def weights2hash(model, dsize=8):
    # compute hash of a torch.nn.Module weights or a list of tensors
    import torch

    h = blake2b(digest_size=dsize)
    # state = {name:par2bytes(p) for name, p in net.named_parameters()}
    # names = sorted(state.keys()) # sort names for reproducibility
    # for name in names:
    #   b = state[name]
    #   h.update(b)
    if issubclass(model.__class__, torch.nn.Module):
        model = model.parameters()
    for p in model:
        h.update(par2bytes(p))
    return h.hexdigest()


def get_files(path, extensions=None, recurse=False, include=False):
    path = Path(path)
    res = []
    if recurse:
        for p, d, f in os.walk(path):
            if include:
                d[:] = [o for o in d]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, f, extensions)


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)


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


def extract_features(net, paths, layer=None, device="cuda", bsize=512):
    net = net.to(device)
    if layer is not None:
        net, unhook = hook_it_up(net, layer)
    tens = paths2tensors(paths)
    dataset = TensorDataset(tens.to(device))
    # cumbersome but you won't cry when you pass 20k examples at once and you RAM gives up
    loader = DataLoader(dataset, batch_size=bsize, shuffle=False)
    feats = np.vstack([net(inps).detach().cpu().numpy() for inps, in loader])
    torch.cuda.empty_cache()
    if layer is not None:
        unhook()
    return feats


def store_activations(net, name, classes, to_hook=None):
    folder = Path(name)
    folder.mkdir(parents=True, exist_ok=True)
    for cls in tqdm(classes):
        fname = folder / (cls.name + ".npy")
        if not fname.exist():
            ims = [im for im in sorted(cls.iterdir())]
            feats = extract_features(net, ims, layer=to_hook, device="cuda")
            with open(folder / f"{cls.name}_names.pkl", "wb") as f:
                pickle.dump(ims, f)
            np.save(fname, feats)


# DON'T go straight from images to distances, store the features, you have to compute
# them anyways and it's the longest step
# def process_class(cls, net, params, debug=False):
#     ims = [im for im in cls.iterdir()]
#     feats = extract_features(net, ims)
#     distances = compute_distances_cuda(feats)
#     best, stats = select_best(distances, params, progress=False)
#     best_ims = np.array(ims)[best]
#     if debug:
#         show_evolution(distances, stats)
#     with open("./OmniNetV3.txt", "a") as f:
#         for path in best_ims:
#             if debug:
#                 print(path.as_posix())
#             else:
#                 f.write(path.as_posix() + "\n")
#     with open(Path("statsV3") / (cls.name + ".pkl"), "wb") as f:
#         if debug:
#             print(f)
#         else:
#             pickle.dump(stats, f)


def process_class_feats(npy, params, version, debug=False):
    distances, ims = feats2distances(npy)
    best, stats = select_best(distances, params, progress=debug)
    best_ims = np.array(ims)[best]

    if debug:
        show_evolution(distances, stats)

    with open(f"./OmniImage{version}.txt", "a") as f:
        for path in best_ims:
            cls, file = path.parts[-2:]
            path = Path(cls) / file
            if debug:
                print(path.as_posix())
            else:
                f.write(path.as_posix() + "\n")
    folder = Path(f"./stats{version}")
    with open(folder / (npy.stem + ".pkl"), "wb") as f:
        if debug:
            print(f)
        else:
            folder.mkdir(exist_ok=True)
            pickle.dump(stats, f)


def generate(im_path, imagenet_dir, folder="OmniImage64"):
    # read im_path from original imagenet, resize and move to new folder
    cls, file = im_path.parts[-2:]
    new_path = Path(folder) / cls / file
    new_path.parent.mkdir(parents=True, exist_ok=True)

    src = str((imagenet_dir / im_path).resolve())
    dest = str(new_path.resolve())

    raw_im = cv2.imread(src)
    small_im = downsize(raw_im, size=64)
    cv2.imwrite(dest, small_im)
    # print(src, dest, small_im.shape)


#%%

if __name__ == "__main__":

    # Extract features and store them
    imagenet_dir = "imagenet_dir"
    version = "V3"
    net = vgg19_bn(pretrained=True, progress=True)
    to_hook = net.classifier[3]
    name = model2name(net)  # torchvision.models.vgg_ce631fc9ca0278a2
    classes = [folder for folder in sorted(Path(imagenet_dir).iterdir())]
    # store_activations(net, name, classes, to_hook)

    # params = Params(N=500, M=20, k=100, muts=5, its=1_000)
    # for cls in tqdm(classes):
    #     try:
    #         process_class(cls, params, debug=False)
    #     except:
    #         print(f"Failed {cls}")

    npys = [
        npy for npy in Path("torchvision.models.vgg_ce631fc9ca0278a2").glob("*.npy")
    ]

    # Params for V3, took approximately 13 hours
    params = Params(N=1_000, M=20, k=100, muts=2, its=2_000)
    for npy in tqdm(npys):
        process_class_feats(npy, params, version=version, debug=False)

    #%%

    with open(f"./OmniImage.txt", "r") as f:
        datav3 = f.read().splitlines()
        datav3 = [Path(p) for p in datav3]

    for im_path in tqdm(datav3):
        generate(im_path, "imagenet_dir")
