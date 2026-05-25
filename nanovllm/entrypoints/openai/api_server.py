from __future__ import annotations

import argparse
import os
import time
import uuid
from threading import Lock, Thread
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from nanovllm import LLM, SamplingParams


app = FastAPI(title="Nano-vLLM OpenAI API")
_llm: LLM | None = None
_model_name = "nano-vllm"
_generation_lock = Lock()


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = 64
    temperature: float = 1.0
    ignore_eos: bool = False
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 64
    temperature: float = 1.0
    ignore_eos: bool = False
    stream: bool = False
    add_generation_prompt: bool = True


class PeerBroadcastRequest(BaseModel):
    init_method: str
    world_size: int
    src: int = 0


def init_engine(model: str, served_model_name: str | None = None, **kwargs: Any):
    global _llm, _model_name
    _llm = LLM(model, **kwargs)
    _model_name = served_model_name or model


def get_engine() -> LLM:
    if _llm is None:
        raise HTTPException(status_code=503, detail="Nano-vLLM engine is not initialized")
    return _llm


def make_sampling_params(request: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
    if request.temperature <= 1e-10:
        raise HTTPException(status_code=400, detail="temperature must be greater than 1e-10")
    if request.max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive")
    return SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
    )


def generate(prompts: list[str], sampling_params: SamplingParams):
    llm = get_engine()
    with _generation_lock:
        return llm.generate(prompts, sampling_params, use_tqdm=False)


def count_prompt_tokens(prompts: list[str]) -> int:
    llm = get_engine()
    return sum(len(llm.tokenizer.encode(prompt)) for prompt in prompts)


def usage(prompts: list[str], outputs: list[dict[str, Any]]) -> dict[str, int]:
    prompt_tokens = count_prompt_tokens(prompts)
    completion_tokens = sum(len(output["token_ids"]) for output in outputs)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def completion_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def chat_prompt(messages: list[ChatMessage], add_generation_prompt: bool) -> str:
    llm = get_engine()
    message_dicts = [dump_model(message) for message in messages]
    tokenizer = llm.tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return "\n".join(f"{message.role}: {message.content}" for message in messages) + "\nassistant:"


@app.get("/health")
def health():
    get_engine()
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name,
                "object": "model",
                "created": created,
                "owned_by": "nano-vllm",
            }
        ],
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    sampling_params = make_sampling_params(request)
    outputs = await run_in_threadpool(generate, prompts, sampling_params)
    return {
        "id": completion_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [
            {
                "index": i,
                "text": output["text"],
                "logprobs": None,
                "finish_reason": "stop",
            }
            for i, output in enumerate(outputs)
        ],
        "usage": usage(prompts, outputs),
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    prompt = chat_prompt(request.messages, request.add_generation_prompt)
    sampling_params = make_sampling_params(request)
    outputs = await run_in_threadpool(generate, [prompt], sampling_params)
    output = outputs[0]
    return {
        "id": completion_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage([prompt], outputs),
    }


def run_peer_broadcast(request: PeerBroadcastRequest):
    llm = get_engine()
    with _generation_lock:
        llm.model_runner.broadcast_model_to_peer(
            request.init_method,
            request.world_size,
            request.src,
        )


@app.post("/admin/peer_broadcast")
def start_peer_broadcast(request: PeerBroadcastRequest):
    get_engine()
    thread = Thread(target=run_peer_broadcast, args=(request,), daemon=True)
    thread.start()
    return {"status": "started"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Nano-vLLM OpenAI-compatible API server.")
    parser.add_argument("--model", required=True, help="Path to the local model directory.")
    parser.add_argument("--served-model-name", default=None, help="Model name returned by /v1/models.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=None)
    parser.add_argument("--devices", default=None, help="Comma-separated CUDA device IDs, e.g. 0,1 or 2,3.")
    parser.add_argument("--distributed-init-method", default="tcp://localhost:2333")
    parser.add_argument("--peer-load-rank", type=int, default=-1)
    parser.add_argument("--peer-load-world-size", type=int, default=1)
    parser.add_argument("--peer-load-src", type=int, default=0)
    parser.add_argument("--peer-load-init-method", default="tcp://localhost:29501")
    parser.add_argument("--peer-load-src-url", default="")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--warmup-max-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "half", "bfloat16", "float32"])
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    args.devices = parse_devices(args.devices, parser)
    args.tensor_parallel_size = resolve_tensor_parallel_size(
        args.tensor_parallel_size,
        args.devices,
        parser,
    )
    validate_peer_loading_args(args, parser)
    return args


def parse_devices(devices: str | None, parser: argparse.ArgumentParser) -> list[str] | None:
    if devices is None:
        return None
    device_ids = [device.strip() for device in devices.split(",")]
    if not device_ids or any(not device for device in device_ids):
        parser.error("--devices must be a comma-separated list, e.g. 0,1")
    if len(set(device_ids)) != len(device_ids):
        parser.error("--devices must not contain duplicate device IDs")
    return device_ids


def resolve_tensor_parallel_size(
    tensor_parallel_size: int | None,
    devices: list[str] | None,
    parser: argparse.ArgumentParser,
) -> int:
    if tensor_parallel_size is None:
        return len(devices) if devices else 1
    if tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if tensor_parallel_size > 1 and not devices:
        parser.error("--devices is required when --tensor-parallel-size is greater than 1")
    if devices and len(devices) != tensor_parallel_size:
        parser.error("--devices count must match --tensor-parallel-size")
    return tensor_parallel_size


def validate_peer_loading_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.peer_load_world_size < 1:
        parser.error("--peer-load-world-size must be positive")
    if args.peer_load_world_size == 1:
        return
    if args.tensor_parallel_size != 1:
        parser.error("peer replica loading currently requires --tp 1")
    if args.peer_load_rank < 0:
        parser.error("--peer-load-rank is required when --peer-load-world-size is greater than 1")
    if args.peer_load_rank >= args.peer_load_world_size:
        parser.error("--peer-load-rank must be less than --peer-load-world-size")
    if not 0 <= args.peer_load_src < args.peer_load_world_size:
        parser.error("--peer-load-src must be in [0, peer-load-world-size)")
    if args.peer_load_rank != args.peer_load_src and not args.peer_load_src_url:
        parser.error("--peer-load-src-url is required for non-source peer loading ranks")


def main():
    args = parse_args()
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.devices)
    init_engine(
        args.model,
        served_model_name=args.served_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_init_method=args.distributed_init_method,
        peer_load_rank=args.peer_load_rank,
        peer_load_world_size=args.peer_load_world_size,
        peer_load_src=args.peer_load_src,
        peer_load_init_method=args.peer_load_init_method,
        peer_load_src_url=args.peer_load_src_url,
        max_model_len=args.max_model_len,
        warmup_max_tokens=args.warmup_max_tokens,
        dtype=args.dtype,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.kvcache_block_size,
        enforce_eager=args.enforce_eager,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
