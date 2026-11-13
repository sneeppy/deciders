import hashlib
import math

import torch


def block_size(width):
    block = 1
    while block < 1024 and width % (block * 2) == 0:
        block *= 2
    if block < 16:
        return None
    return block


def group_size(width):
    for group in (128, 64, 32, 16):
        if width % group == 0:
            return group
    return None


def sign_vector(name, width):
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    seed %= 2**63 - 1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8)
    return bits.mul(2).sub_(1)


def fwht(values):
    width = values.shape[-1]
    if width & (width - 1):
        raise ValueError(f"длина блока Адамара должна быть степенью двойки, получено {width}")
    output = values.float()
    step = 1
    while step < width:
        output = output.reshape(*output.shape[:-1], width // (2 * step), 2, step)
        first = output[..., 0, :]
        second = output[..., 1, :]
        output = torch.stack((first + second, first - second), dim=-2)
        output = output.reshape(*values.shape[:-1], width)
        step *= 2
    return output / math.sqrt(width)


def block_hadamard(values, block):
    if values.shape[-1] % block != 0:
        raise ValueError(f"ширина {values.shape[-1]} не делится на блок {block}")
    values = values.contiguous()
    folded = values.reshape(*values.shape[:-1], values.shape[-1] // block, block)
    rotated = fwht(folded)
    return rotated.reshape(values.shape)
