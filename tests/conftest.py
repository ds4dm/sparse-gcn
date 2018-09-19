# coding:utf-8

"""Pytest objects."""

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Device to run test on."""
    _device = torch.device(request.param)
    if _device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip()
    return _device
