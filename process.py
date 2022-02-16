from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vgg19_bn
from tqdm import tqdm

from hooker import hook_it_up
from utils.image import paths2tensors, read_folder
from utils.torch import model2name


def extract_features(net, source, layer=None, device="cuda", bsize=512):
    # source can either be a torch.Tensor or a pathlib.Path.iterdir
    net = net.to(device)
    if layer is not None:
        net, unhook = hook_it_up(net, layer)
    if type(source) == torch.Tensor:
        tens = source
    else:
        tens = paths2tensors(source)
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


def generate(im_path, imagenet_dir, folder):
    """
    Read im_path from original imagenet, resize and move to new folder

    Parameters
    ----------

    im_path : list[Path]
        images to read

    imagenet_dir : Path
        location of original imagenet images (referenced by im_path)

    folder: Path
        output folder to store the resized images to (e.g. OmniImage64)

    Returns
    -------
    None

    """
    cls, file = im_path.parts[-2:]
    new_path = Path(folder) / cls / file
    new_path.parent.mkdir(parents=True, exist_ok=True)

    src = (imagenet_dir / im_path).resolve()
    dest = str(new_path.resolve())

    small_im = cv2.imread(src.as_posix())
    cv2.imwrite(dest, small_im)


#%%

if __name__ == "__main__":

    # Extract features and store them
    def extract():
        imagenet_dir = "imagenet"
        net = vgg19_bn(pretrained=True, progress=True)
        to_hook = net.classifier[3]
        name = model2name(net)  # torchvision.models.vgg_ce631fc9ca0278a2
        classes = read_folder(imagenet_dir)
        store_activations(net, name, classes, to_hook)

    #%%

    with open(f"./OmniImage_100.txt", "r") as f:
        data = f.read().splitlines()
        data = [Path(p) for p in data]
    data

    for im_path in tqdm(data):
        generate(im_path, "imagenet64", "OmniImage64_100")
        # generate(im_path, "imagenet", "OmniImage28", 28)
        # generate(im_path, "imagenet", "OmniImage32", 32)
