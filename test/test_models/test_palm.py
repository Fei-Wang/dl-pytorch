import torch

from luffy.models.palm import *


def test_palm_tony():
    model = PaLMTony(num_tokens=20000)

    tokens = torch.randint(0, 20000, (1, 2048))
    feat = model(tokens)
    assert feat.shape == (1, 2048, 20000)
