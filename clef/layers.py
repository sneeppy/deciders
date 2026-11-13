import torch
import torch.nn.functional as functional

from .hadamard import block_hadamard


class HadamardInt4(torch.nn.Module):
    def __init__(self, packed, scales, signs, bias, block, group):
        super().__init__()
        self.out_features, self.in_features = scales.shape[0], scales.shape[1] * group
        self.block = block
        self.group = group
        self.register_buffer("packed", packed.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("signs", signs.to(torch.float32).contiguous())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.contiguous())

    def forward(self, inputs):
        leading = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).float() * self.signs
        flat = block_hadamard(flat, self.block).to(torch.bfloat16)
        weight = dequantize_int4(self.packed, self.scales, self.group)
        output = functional.linear(flat, weight, self.bias)
        return output.reshape(*leading, self.out_features).to(inputs.dtype)


class Int8Rows(torch.nn.Module):
    """Строки матрицы в int8. И lookup эмбеддинга, и индексация lm_head идут через forward/weight."""

    def __init__(self, codes, scales, group):
        super().__init__()
        self.group = group
        self.register_buffer("codes", codes.contiguous())
        self.register_buffer("scales", scales.contiguous())

    @property
    def weight(self):
        return _RowIndex(self.codes, self.scales, self.group)

    def forward(self, token_ids):
        flat = token_ids.reshape(-1)
        rows = gather_rows(self.codes, self.scales, self.group, flat)
        return rows.reshape(*token_ids.shape, self.codes.shape[-1])


class _RowIndex:
    def __init__(self, codes, scales, group):
        self.codes = codes
        self.scales = scales
        self.group = group

    def __getitem__(self, index):
        if isinstance(index, tuple):
            raise TypeError("у таблицы токенов ожидается один индекс")
        return gather_rows(self.codes, self.scales, self.group, index)


def gather_rows(codes, scales, group, index):
    selected = codes[index]
    selected_scales = scales[index]
    hidden = codes.shape[-1]
    values = selected.float().reshape(*selected.shape[:-1], hidden // group, group)
    values = values * selected_scales.unsqueeze(-1)
    return values.reshape(*selected.shape[:-1], hidden).to(torch.bfloat16)


def dequantize_int4(packed, scales, group):
    low = packed & 0x0F
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(scales.shape[0], -1)
    quantized = nibbles.to(torch.float32) - 7
    out_features, groups = scales.shape
    values = quantized.reshape(out_features, groups, group) * scales.float().unsqueeze(-1)
    return values.reshape(out_features, -1).to(torch.bfloat16)
