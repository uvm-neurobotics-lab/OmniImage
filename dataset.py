import random

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image
from tqdm import trange

from utils.image import get_images


class OmniImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        get_class = lambda x: x.parent.name
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(get_images(self.img_dir))
        self.classes = [get_class(im) for im in self.images]
        self.uniq_classes = list(sorted(set(self.classes)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.uniq_classes)}
        self.labels = [self.class_to_idx[get_class(im)] for im in self.images]
        self.memo = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.memo:
            image = read_image(self.images[idx].as_posix()) / 255
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            self.memo[idx] = (image, label)
        else:
            image, label = self.memo[idx]
        return image, label


def split(dataset, p=0.8, samples=20, verbose=False, seed=None):
    # randomly split the dataset between train/test
    # e.g. samples=3, nclasses=100, p=0.8
    # labels is a list of ints #[0,0,0,1,1,1,2,2,...,100]
    if verbose:
        print("Preparing splits...")
        range_fun = trange
    else:
        range_fun = range
    rng = random.Random(seed)
    labels = [dataset[i][1] for i in range_fun(len(dataset))]
    assert is_sorted(labels)
    classes = list(set(labels))  # [0,1,2,...,100]
    ntrain = int(len(classes) * p)  # 100*0.8 = 80
    rng.shuffle(classes)
    train_classes = sorted(classes[:ntrain])  # [0,3,4,...,93] : 80
    test_classes = sorted(classes[ntrain:])  # [1,2,5,...,100] : 20
    train_idxs = [
        (i * samples) + j for i in train_classes for j in range(samples)
    ]  # [0,1,2,9,10,11,...,276]
    test_idxs = [
        (i * samples) + j for i in test_classes for j in range(samples)
    ]  # [3,4,5,6,7,8,5,5,5,...,300]
    if verbose:
        print(f"Splits ready: Train:{len(train_idxs)} Test:{len(test_idxs)}")
    return (
        Subset(dataset, train_idxs),
        Subset(dataset, test_idxs),
        train_classes,
        test_classes,
    )


def is_sorted(l):
    return all(i <= j for i, j in zip(l, l[1:]))


class OMLSampler:
    def __init__(self, dataset, device="cpu", nsamples_per_class=20):
        # assumes indexes are sorted per class 00..0011..1122...
        # assert is_sorted([dataset[i][1] for i in range(len(dataset))])
        self.dataset = dataset
        self.device = device
        assert self.device in [
            "cpu",
            "cuda",
        ], f"Device must be either 'cpu' or 'cuda': {device} found."
        self.samples = np.arange(len(dataset))
        self.class_idxs = self.samples.reshape(-1, nsamples_per_class)
        self.classes = np.arange(self.class_idxs.shape[0])
        self.balanced = np.copy(self.class_idxs).T
        self.rng = np.random.default_rng()
        self.rng.permuted(self.balanced, axis=0, out=self.balanced)
        self.slice = 0

    def __repr__(self):
        return f"Sampler: {len(self.dataset)} SAMPLES, {len(self.classes)} CLASSES, DEVICE={self.device}"

    def get(self, idxs):
        # get batch of ims,labels from a list of indices
        ims = [self.dataset[i][0] for i in idxs]
        lbs = [self.dataset[i][1] for i in idxs]
        return torch.stack(ims).to(self.device), torch.tensor(lbs).to(self.device)

    def sample_class(self, N=20):
        # get N ims of a single class, used for inner loop
        # NOTE: always gets the same N ims
        cls = np.random.choice(self.classes)
        return self.get(self.class_idxs[cls][:N])

    def sample_random(self, N=64):
        # get N at random from the whole dataset, used for the outer loop
        samples = np.random.choice(self.samples, size=N, replace=False)
        return self.get(samples)

    def sample(self, inner_size=20, outer_size=64, expand=True):
        # get 20 ims for the inner and 20+64 ims for the outer
        inner_ims, inner_labels = self.sample_class(inner_size)
        outer_ims, outer_labels = self.sample_random(outer_size)
        outer_ims = torch.cat([inner_ims, outer_ims])
        outer_labels = torch.cat([inner_labels, outer_labels])
        if expand:
            inner_ims = inner_ims.unsqueeze(1)
        return inner_ims, inner_labels, outer_ims, outer_labels

    def sample_balanced(self):
        # get a sample with exactly 1 example from every class
        idxs = self.balanced[self.slice]
        samples = self.get(idxs)
        if self.slice >= len(self.balanced) - 1:
            self.slice = 0
            self.rng.permuted(self.balanced, axis=0, out=self.balanced)
            # NOTE: this will change idxs, use them before shuffling
        else:
            self.slice += 1
        return samples


#%%

if __name__ == "__main__":

    dataset = OmniImageDataset("OmniImage64")

    train, test, tr_cls, te_cls = split(dataset, verbose=True, seed=4)
    print(len(train), len(test))

    device = "cuda"
    sampler = OMLSampler(train, device)
    print(sampler)

    # OMLSampler(dataset) to use the whole dataset instead

    # inner/outer sample sizes
    I, O = 15, 32
    inner_ims, inner_labels, outer_ims, outer_labels = sampler.sample(
        inner_size=I, outer_size=O
    )
    print(inner_ims.shape)
    print(inner_labels.shape)
    print(outer_ims.shape)
    print(outer_labels.shape)
    assert (
        inner_ims.shape[0] == I
    ), f"Inner loop mismatch: Expected {I} found {inner_ims.shape[0]}"
    assert (
        outer_ims.shape[0] == I + O
    ), f"Outer loop mismatch: Expected {I+O} found {outer_ims.shape[0]}"

    ims, labels = sampler.sample_balanced()
    assert ims.device.type == device
    assert len(set([label.item() for label in labels])) == len(labels)
