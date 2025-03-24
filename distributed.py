"""
===========================================================
Distributed Initialization Script
===========================================================

This script initializes the distributed process group for training models using multiple GPUs. It includes the following steps:
1. Setting the device for the current process
2. Initializing the process group using the NCCL backend

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import os
import torch
import torch.distributed as dist

def init_distributed():
    """
    Initialize the distributed process group.
    Environment variables (RANK, WORLD_SIZE, LOCAL_RANK) must be set.

    Returns:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        local_rank (int): Local rank of the current process.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank
