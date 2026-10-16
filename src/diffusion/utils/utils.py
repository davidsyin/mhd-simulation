from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast
import torch
import math

import os
import torch
import torch.special as ts
import numpy as np
import operator
import matplotlib.pyplot as plt

from scipy import interpolate
from functools import reduce


T = TypeVar("T")
FuncType = Callable[..., Any]


def exists(val: Optional[Any]) -> bool:
    """
    Check if a value exists (is not None).

    Args:
        val: The value to check for existence

    Returns:
        bool: True if the value is not None, False otherwise
    """
    return val is not None


def default(val: Optional[T], d: Union[T, Callable[[], T]]) -> T:
    """
    Return val if it exists, otherwise return default d.

    Args:
        val: The value to check
        d: The default value or function to return if val is None

    Returns:
        The value val if it exists, otherwise the default value
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def once(fn: FuncType) -> FuncType:
    """
    Decorator to ensure a function is only called once.

    Args:
        fn: The function to wrap

    Returns:
        Wrapped function that will only execute on the first call
    """
    called = False

    @wraps(fn)
    def inner(x: Any) -> Optional[Any]:
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return cast(FuncType, inner)


print_once = once(print)


def cast_tuple(t: Any, length: int = 1) -> Tuple[Any, ...]:
    """
    Cast input to a tuple of specified length.

    Args:
        t: The input to cast to a tuple
        length: The desired length of the output tuple

    Returns:
        A tuple of the specified length containing the input value(s)
    """
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer: int, denom: int) -> bool:
    """
    Check if number is divisible by another.

    Args:
        numer: The numerator (number to be divided)
        denom: The denominator (number to divide by)

    Returns:
        bool: True if numer is divisible by denom, False otherwise
    """
    return (numer % denom) == 0


def log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    Safe logarithm with minimum clamping to avoid numerical issues.

    Args:
        t: The input tensor
        eps: Small epsilon value for numerical stability

    Returns:
        Tensor containing the log of the input with minimum clamping
    """
    return torch.log(t.clamp(min=eps))


def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all({s})
    '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("=" * 50)
        print(message)
        print("=" * 50)




def get_num_params(model):
    '''
    a single entry in cfloat and cdouble count as two parameters
    see https://github.com/pytorch/pytorch/issues/57518
    '''
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_params = 0
    # for p in model_parameters:
    #     # num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
    #     num_params += p.numel() * (1 + p.is_complex())
    # return num_params

    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))  #### there is complex weight
    return c



### x: list of tensors
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)


    def __getitem__(self, item):
        # return torch.rand_like(self.x[item])
        return self.x[item]
