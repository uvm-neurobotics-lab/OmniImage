import torch
from hashlib import blake2b


def model2name(net):
    return f"{net.__module__}_{weights2hash(net)}"


def par2bytes(p):
    return p.detach().cpu().numpy().tobytes()


def weights2hash(model, dsize=8):
    # compute hash of a torch.nn.Module weights or a list of tensors

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
