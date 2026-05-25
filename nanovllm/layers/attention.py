import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _flash_attn_supported() -> bool:
    if flash_attn_varlen_func is None or flash_attn_with_kvcache is None:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    assert num_heads % x.size(1) == 0
    return x.repeat_interleave(num_heads // x.size(1), dim=1)


def _paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = k_cache[block_table].flatten(0, 1)
    v = v_cache[block_table].flatten(0, 1)
    return k, v


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.use_flash_attn = _flash_attn_supported()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if not self.use_flash_attn:
            return self.torch_attention(q, k, v)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

    def torch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            return self.torch_prefill(q, k, v)
        return self.torch_decode(q)

    def torch_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        outputs = []
        cu_q = context.cu_seqlens_q.tolist()
        cu_k = context.cu_seqlens_k.tolist()
        for i in range(len(cu_q) - 1):
            q_start, q_end = cu_q[i], cu_q[i + 1]
            k_start, k_end = cu_k[i], cu_k[i + 1]
            q_i = q[q_start:q_end]
            if context.block_tables is None:
                k_i = k[k_start:k_end]
                v_i = v[k_start:k_end]
            else:
                k_i, v_i = _paged_kv_cache(self.k_cache, self.v_cache, context.block_tables[i])
                k_i = k_i[:k_end - k_start]
                v_i = v_i[:k_end - k_start]
            outputs.append(self.scaled_dot_product_attention(q_i, k_i, v_i))
        return torch.cat(outputs, dim=0)

    def torch_decode(self, q: torch.Tensor):
        context = get_context()
        outputs = []
        max_len = context.block_tables.size(1) * self.k_cache.size(1)
        key_positions = torch.arange(max_len, device=q.device)
        for i in range(q.size(0)):
            k_i, v_i = _paged_kv_cache(self.k_cache, self.v_cache, context.block_tables[i])
            k_i = k_i[:max_len]
            v_i = v_i[:max_len]
            attn_mask = key_positions < context.context_lens[i]
            outputs.append(self.scaled_dot_product_attention(q[i:i + 1], k_i, v_i, attn_mask))
        return torch.cat(outputs, dim=0)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = _repeat_kv(k, self.num_heads)
        v = _repeat_kv(v, self.num_heads)
        q_len, k_len = q.size(0), k.size(0)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        if attn_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        else:
            attn_mask = attn_mask.view(1, 1, q_len, k_len)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        return o.squeeze(0).transpose(0, 1)
