import torch


class ForwardStopped(Exception):
    def __init__(self, module):
        self.m = module

    def __str__(self):
        return f"Stopped @ {self.m}"


class FeatsExtractor:
    def __init__(self, net, layer, debug=False):
        self.debug = debug

        activations = None

        def my_hook(m, i, o):
            nonlocal activations
            activations = o
            raise ForwardStopped(m)

        def forward(inp):
            try:
                with torch.no_grad():
                    out = net(inp)
                    print("YOU SHOULDN'T SEE THIS", out.shape)
            except ForwardStopped as e:
                if debug:
                    print(e)
                else:
                    pass
            return activations

        self.handle = layer.register_forward_hook(my_hook)
        self.extract = forward
        net.eval()

    def __enter__(self):
        if self.debug:
            print("Starting context: ", self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            print("Exiting context: ", self, exc_type, exc_value, traceback)
        self.handle.remove()

        # suppress errors
        return True


def hook_it_up(net, layer, debug=False):
    activations = None

    def my_hook(m, i, o):
        nonlocal activations
        activations = o
        raise ForwardStopped(m)

    handle = layer.register_forward_hook(my_hook)
    net.eval()

    def forward(inp):
        try:
            with torch.no_grad():
                out = net(inp)
                print("YOU SHOULDN'T SEE THIS", out.shape)
        except ForwardStopped as e:
            if debug:
                print(e)
            else:
                pass
        return activations

    return forward, handle.remove


if __name__ == "__main__":
    from torchvision.models import vgg19_bn

    dummy_in = torch.rand(8, 3, 64, 64)
    test_net = vgg19_bn(pretrained=True, progress=True)
    to_hook = test_net.classifier[3]

    extract, unhook = hook_it_up(test_net, to_hook, debug=True)
    a = [extract(dummy_in) for _ in range(2)]
    assert torch.allclose(a[0], a[1])
    unhook()

    # with FeatsExtractor(test_net, to_hook, debug=False) as h:
    #     a = [h.extract(dummy_in) for _ in range(3)]
    #     assert torch.allclose(a[0], a[1])
    #     assert torch.allclose(a[1], a[2])

#%%

# dummy_in = torch.rand(8, 3, 64, 64)
# net(dummy_in).shape

# net = vgg19_bn(pretrained=True, progress=True)
# activs = []


# def hook(m, i, o):
#    global activs
#    print("inputs:", [el.shape for el in i], len(i))
#    print("outputs:", o.shape)
#    print(type(i), type(o))
#    activs = o


## handler.remove()
## handler = net.features[49].register_forward_hook(hook)
# to_hook = net.classifier[3]
# print("Hooking", to_hook)
# handler = to_hook.register_forward_hook(hook)
# net(dummy_in).shape
# print(activs.shape)
# handler.remove()


##%%


# class Net(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.fc1 = torch.nn.Linear(3 * 64 * 64, 1000)
#        self.fc2 = torch.nn.Linear(1000, 100)

#    def forward(self, x):
#        x = torch.flatten(x, start_dim=1)
#        x = self.fc1(x)
#        x = self.fc2(x)
#        return x


## test_net = Net()
## to_hook = test_net.fc1
# net = vgg19_bn(pretrained=True, progress=True)
# to_hook = net.classifier[3]


# extract, unhook = hook_it_up(net, to_hook, debug=True)
# a = [extract(dummy_in) for _ in range(2)]
# assert torch.allclose(a[0], a[1])
# unhook()
