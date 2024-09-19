import pytest
import torch
from app.sum import add as sum_triton


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_sum_small_tensor(device):
    x = torch.tensor([1, 2, 3, 4, 5], device=device)
    y = torch.tensor([6, 7, 8, 9, 10], device=device)
    expected = x + y
    result = sum_triton(x, y)
    assert torch.allclose(result, expected)

def test_sum_large_tensor(device):
    x = torch.randn(1000000, device=device)
    y = torch.randn(1000000, device=device)
    expected = x + y
    result = sum_triton(x, y)
    assert torch.allclose(result, expected)

def test_sum_negative_values(device):
    x = torch.tensor([-1, -2, 3, -4, 5], device=device)
    y = torch.tensor([-6, -7, -8, -9, -10], device=device)
    expected = x + y
    result = sum_triton(x, y)
    assert torch.allclose(result, expected)