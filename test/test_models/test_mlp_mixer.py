import torch

from luffy.models.mlp_mixer import *


def test_mixer_s32():
    model = MixerS32(image_size=224, num_classes=1000)

    images = torch.randn(1, 3, 224, 224)
    feat = model(images)
    assert feat.shape == (1, 1000)
