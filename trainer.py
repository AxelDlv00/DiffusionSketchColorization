"""
===========================================================
Trainer Script for Sketch Colorization Using Diffusion Models
===========================================================

This script handles the training process for the sketch colorization model using diffusion models and photo-sketch correspondence techniques. It includes the following steps:
1. Distributed initialization
2. Loading the PSC model
3. Loading the dataset
4. Initializing the models
5. Wrapping models in DistributedDataParallel (DDP)
6. Setting up the optimizer and scheduler
7. Training loop with checkpointing

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
from functools import partial

from models.reference_unet import ReferenceUNet
from models.denoising_unet import DenoisingUNet
from models.psc_diffusion import PSCGaussianDiffusion
from psc_project.PSCNet import PSCNet
from psc_project.moco import MoCo
from psc_project.resnet_cbn import resnet101
from utils.data.dataset_utils import CustomAnimeDataset, worker_init_fn
from train.distributed import init_distributed

def save_loss_history(loss_list, checkpoint_dir):
    """Save the loss history to a JSON file."""
    loss_file = os.path.join(checkpoint_dir, "loss_history.json")
    with open(loss_file, "w") as f:
        json.dump(loss_list, f, indent=4)

def load_loss_history(checkpoint_dir):
    """Load the loss history from a JSON file."""
    loss_file = os.path.join(checkpoint_dir, "loss_history.json")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            return json.load(f)
    return []

def custom_train(dataset_path,
                 epochs,
                 batch_size,
                 patch_size,
                 lr,
                 train_ratio,
                 subset_percentage,
                 psc_checkpoint_path,
                 checkpoint_dir,
                 N_refresh_log,
                 image_size):
    """
    Custom training function for the sketch colorization model.

    Args:
        dataset_path (str): Path to the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        patch_size (int): Size of the patches.
        lr (float): Learning rate.
        train_ratio (float): Ratio of training data.
        subset_percentage (float): Percentage of the dataset to use.
        psc_checkpoint_path (str): Path to the PSC model checkpoint.
        checkpoint_dir (str): Directory to save checkpoints.
        N_refresh_log (int): Frequency of logging and checkpointing.
        image_size (int): Size of the input images.

    Returns:
        ref_unet, denoising_unet, gaussian_diffusion, PSC_MODEL_WORKER, Loss
    """
    # Distributed initialization
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 1. Load the main PSC model onto the correct device
    PSC_MODEL_WORKER = PSCNet(MoCo, resnet101, dim=128, K=8192, corr_layer=[2, 3]).to(device)
    if os.path.isfile(psc_checkpoint_path):
        checkpoint_psc = torch.load(psc_checkpoint_path, map_location=device)
        state_dict = checkpoint_psc["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("module."):
                state_dict[k[len("module."):]] = state_dict.pop(k)
        PSC_MODEL_WORKER.load_state_dict(state_dict, strict=False)
    else:
        if rank == 0:
            print(f"[Warning] PSC checkpoint not found at {psc_checkpoint_path} - continuing anyway")

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    last_checkpoint_file = os.path.join(checkpoint_dir, "last_checkpoint.pth.tar")
    best_checkpoint_file = os.path.join(checkpoint_dir, "best_checkpoint.pth.tar")

    # 2. Load the dataset
    sketch_dir = os.path.join(dataset_path, "sketch")
    reference_dir = os.path.join(dataset_path, "reference")
    train_dataset = CustomAnimeDataset(None, sketch_dir, reference_dir, subset_percentage=subset_percentage)
    train_dataset.set_psc_model(PSC_MODEL_WORKER)
    if rank == 0:
        print(f"Dataset size: {len(train_dataset)} total triplets.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    worker_fn = partial(worker_init_fn, local_rank=local_rank)
    n_gpu = torch.cuda.device_count()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=False, worker_init_fn=worker_fn)

    # 3. Initialize the other models
    ref_unet = ReferenceUNet(in_channels=3, base_ch=64, num_res_blocks=3, num_attn_blocks=3, cbam=False).to(device)
    denoising_unet = DenoisingUNet(channel_in=9, channel_out=3, channel_base=64, channel_features=64,
                                   n_res_blocks=2, dropout=0.1, channel_mult=(1,2,4,8),
                                   attention_head=4, cbam=True).to(device)
    gaussian_diffusion = PSCGaussianDiffusion(time_step=1000,
                                              betas={"linear_start": 0.0001, "linear_end": 0.02},
                                              denoising_unet=denoising_unet,
                                              psc_model=PSC_MODEL_WORKER).to(device)

    # 4. Wrap models in DDP with find_unused_parameters=True
    from torch.nn.parallel import DistributedDataParallel as DDP
    ref_unet = DDP(ref_unet, device_ids=[local_rank], find_unused_parameters=True)
    denoising_unet = DDP(denoising_unet, device_ids=[local_rank], find_unused_parameters=True)
    gaussian_diffusion = DDP(gaussian_diffusion, device_ids=[local_rank], find_unused_parameters=True)
    PSC_MODEL_WORKER = DDP(PSC_MODEL_WORKER, device_ids=[local_rank], find_unused_parameters=True)

    # 5. Optimizer and scheduler
    optimizer = optim.AdamW(list(ref_unet.parameters()) +
                            list(denoising_unet.parameters()) +
                            list(gaussian_diffusion.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    Loss = load_loss_history(checkpoint_dir)
    start_epoch = 0
    if os.path.exists(last_checkpoint_file) and rank == 0:
        checkpoint = torch.load(last_checkpoint_file, map_location=device)
        ref_unet.module.load_state_dict(checkpoint["ref_unet_state_dict"])
        denoising_unet.module.load_state_dict(checkpoint["denoising_unet_state_dict"])
        gaussian_diffusion.module.load_state_dict(checkpoint["gaussian_diffusion_state_dict"])
        PSC_MODEL_WORKER.module.load_state_dict(checkpoint["psc_model_state_dict"])

        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("[WARNING] Optimizer state does not match, reinitializing optimizer.")
            optimizer = optim.AdamW(set(ref_unet.parameters()) | set(denoising_unet.parameters()) | set(gaussian_diffusion.parameters()), lr=lr)

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")
    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    torch.distributed.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())

    if rank == 0:
        print(f"Starting training on {len(train_dataset)} samples for {epochs} epochs...")

    ref_unet.train()
    denoising_unet.train()
    gaussian_diffusion.train()
    PSC_MODEL_WORKER.eval()  # PSC remains in evaluation mode

    try:
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            total_loss = 0.0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(rank != 0)), 1):
                sketch, image, other_image, patched_ref, flow_warp = [x.to(device) for x in batch]
                t = torch.randint(0, 1000, (sketch.shape[0],), device=device).long()

                ref_feats = ref_unet(patched_ref)
                loss = gaussian_diffusion(image, t, sketch, flow_warp, ref_feats)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            Loss.append(avg_loss)
            scheduler.step()

            if rank == 0:
                print(f"Epoch [{epoch+1}/{epochs}] average loss: {avg_loss:.4f}")
                save_loss_history(Loss, checkpoint_dir)

                checkpoint_dict = {
                    "epoch": epoch,
                    "ref_unet_state_dict": ref_unet.module.state_dict(),
                    "denoising_unet_state_dict": denoising_unet.module.state_dict(),
                    "gaussian_diffusion_state_dict": gaussian_diffusion.module.state_dict(),
                    "psc_model_state_dict": PSC_MODEL_WORKER.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(checkpoint_dict, last_checkpoint_file)
                if avg_loss == min(Loss):
                    print("[INFO] Saving new best checkpoint...")
                    torch.save(checkpoint_dict, best_checkpoint_file)

                if (epoch + 1) % N_refresh_log == 0:
                    torch.save(checkpoint_dict, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth.tar"))
    except KeyboardInterrupt:
        if rank == 0:
            print("[INFO] Training interrupted. Saving loss history before exiting...")
            save_loss_history(Loss, checkpoint_dir)
    finally:
        torch.distributed.destroy_process_group()

    return ref_unet, denoising_unet, gaussian_diffusion, PSC_MODEL_WORKER, Loss

"""
import json
import matplotlib.pyplot as plt

loss_file = "path/to/checkpoint_dir/loss_history.json"
with open(loss_file, "r") as f:
    loss_data = json.load(f)

plt.plot(loss_data)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Evolution During Training")
plt.show()
"""