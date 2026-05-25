import os
from time import perf_counter
from glob import glob
import torch
import torch.distributed as dist
from torch import nn
from safetensors import safe_open
from tqdm import tqdm


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_weight(
    model: nn.Module,
    weight_name: str,
    loaded_weight: torch.Tensor,
    packed_modules_mapping: dict,
):
    for k in packed_modules_mapping:
        if k in weight_name:
            v, shard_id = packed_modules_mapping[k]
            param_name = weight_name.replace(k, v)
            param = model.get_parameter(param_name)
            weight_loader = getattr(param, "weight_loader")
            weight_loader(param, loaded_weight, shard_id)
            break
    else:
        param = model.get_parameter(weight_name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def load_model_from_files(model: nn.Module, path: str, packed_modules_mapping: dict):
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    for file in tqdm(files, desc="Loading model weights"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in tqdm(f.keys(), desc=os.path.basename(file), leave=False):
                load_weight(model, weight_name, f.get_tensor(weight_name), packed_modules_mapping)


def _model_tensors(model: nn.Module):
    yield from model.named_parameters()
    yield from model.named_buffers()


def _ensure_cuda_tensor(name: str, tensor: torch.Tensor):
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} is on {tensor.device}; peer broadcast requires CUDA tensors")


def broadcast_loaded_model(model: nn.Module, src: int = 0):
    """Copy an already loaded full model replica from src to peer ranks via NCCL."""
    start = perf_counter()
    rank = dist.get_rank()
    tensors = list(_model_tensors(model))
    desc = "Broadcasting model weights" if rank == src else "Receiving model weights"
    iterator = tqdm(tensors, desc=desc)
    for name, tensor in iterator:
        _ensure_cuda_tensor(name, tensor)
        if rank == src:
            send_tensor = tensor.detach()
            if not send_tensor.is_contiguous():
                send_tensor = send_tensor.contiguous()
            dist.broadcast(send_tensor, src=src)
        else:
            dist.broadcast(tensor.data, src=src)
    dist.barrier()
    action = "broadcasted" if rank == src else "received"
    tqdm.write(f"Model weights {action} in {perf_counter() - start:.2f}s")


def load_model_from_peer(model: nn.Module, path: str, src: int = 0):
    """Load src rank from files, then broadcast that loaded replica to peers."""
    rank = dist.get_rank()
    if rank == src:
        load_model(model, path)
    broadcast_loaded_model(model, src=src)


def load_model(model: nn.Module, path: str):
    start = perf_counter()
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    load_model_from_files(model, path, packed_modules_mapping)
    tqdm.write(f"Model weights loaded in {perf_counter() - start:.2f}s")
