import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vgg19_bn
from tqdm import tqdm

from cosines import feats2distances
from evolve import Params, select_best
from hooker import hook_it_up
from utils.image import downsize, paths2tensors, read_folder
from utils.viz import show_evolution
from utils.torch import model2name


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
        if not fname.exists():
            ims = read_folder(cls)
            feats = extract_features(net, ims, layer=to_hook, device="cuda")
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


def generate(im_path, imagenet_dir, folder="OmniImage64", im_size=64):
    # read im_path from original imagenet, resize and move to new folder
    cls, file = im_path.parts[-2:]
    new_path = Path(folder) / cls / file
    new_path.parent.mkdir(parents=True, exist_ok=True)

    src = str((imagenet_dir / im_path).resolve())
    dest = str(new_path.resolve())

    raw_im = cv2.imread(src)
    small_im = downsize(raw_im, size=im_size)
    cv2.imwrite(dest, small_im)


#%%

if __name__ == "__main__":

    # Extract features and store them
    imagenet_dir = "imagenet_dir"
    version = "V3"
    net = vgg19_bn(pretrained=True, progress=True)
    to_hook = net.classifier[3]
    name = model2name(net)  # torchvision.models.vgg_ce631fc9ca0278a2
    classes = read_folder(imagenet_dir)
    store_activations(net, name, classes, to_hook)

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
        # generate(im_path, "imagenet_dir", "OmniImage28", 28)
        # generate(im_path, "imagenet_dir", "OmniImage32", 32)
