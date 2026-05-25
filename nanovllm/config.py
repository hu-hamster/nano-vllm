import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


def resolve_dtype(dtype: str, hf_config: AutoConfig):
    dtype_map = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype != "auto":
        assert dtype in dtype_map, f"unsupported dtype: {dtype}"
        return dtype_map[dtype]
    config_dtype = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float16
    if config_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return torch.float16
    return config_dtype


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    warmup_max_tokens: int = 512
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    distributed_init_method: str = "tcp://localhost:2333"
    peer_load_rank: int = -1
    peer_load_world_size: int = 1
    peer_load_src: int = 0
    peer_load_init_method: str = "tcp://localhost:29501"
    peer_load_src_url: str = ""
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.warmup_max_tokens > 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.peer_load_world_size >= 1
        assert 0 <= self.peer_load_src < self.peer_load_world_size
        assert self.peer_load_rank == -1 or 0 <= self.peer_load_rank < self.peer_load_world_size
        assert self.peer_load_world_size == 1 or self.tensor_parallel_size == 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.dtype = resolve_dtype(self.dtype, self.hf_config)
        self.hf_config.torch_dtype = self.hf_config.dtype
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
