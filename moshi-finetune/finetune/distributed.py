import logging
import os
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


@lru_cache()
def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


@lru_cache()
def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def visible_devices() -> List[int]:
    """CUDA_VISIBLE_DEVICES가 정수 인덱스 또는 UUID 형식 모두 처리.
    UUID 형식(예: 'GPU-xxxx-...')일 경우 device_count 기반으로 fallback.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw:
        return list(range(torch.cuda.device_count()))
    entries = [d.strip() for d in raw.split(",")]
    try:
        return [int(d) for d in entries]
    except ValueError:
        # UUID 형식 — 개수만 맞추고 0-based 인덱스로 반환
        return list(range(len(entries)))


def set_device():
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
    logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")

    assert torch.cuda.is_available()

    if torch.cuda.device_count() == 1:
        # 단일 GPU
        torch.cuda.set_device(0)
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Set cuda device to {local_rank}")

    assert 0 <= local_rank < torch.cuda.device_count(), (
        local_rank,
        torch.cuda.device_count(),
    )
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
