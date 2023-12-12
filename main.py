from tqdm import tqdm

import numpy
import matplotlib.pyplot as plt

import os

import torch
from torch import nn

import mmdet
from  mmrazor.models.task_modules.recorder import ModuleOutputsRecorder


def load_data():
    pass

def load_teacher():
    pass

def load_student():
    pass

def eval():
    pass

def train():
    pass

def predict():
    pass

def main():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 1, 1)
            self.conv2 = nn.Conv2d(1, 1, 1)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv1(x + 1)
            return self.conv2(x1 + x2)

    model = ToyModel()
    # instantiate with specified module name.
    r1 = ModuleOutputsRecorder('conv1')

    # initialize is to make specified module can be recorded by
    # registering customized forward hook.
    r1.initialize(model)

    x = torch.randn(1, 1, 1, 1)
    with r1:
        out = model(x)

    print(r1.data_buffer)
    # [tensor([[[[0.0820]]]], grad_fn=<ThnnConv2DBackward0>), tensor([[[[-0.0894]]]], grad_fn=<ThnnConv2DBackward0>)]

    print(torch.equal(r1.data_buffer[0], model.conv1(x)))
    # True
    print(torch.equal(r1.data_buffer[1], model.conv1(x + 1)))
    # True

if __name__ == '__main__':
    main()
    pass