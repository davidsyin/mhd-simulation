import torch
from tqdm import tqdm
import time


def train_epoch_diffusion(model, train_loader, optimizer, device):
    """Train diffusion model for one epoch

    Args:
        model: Diffusion model
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        device: Device to run training on

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0

    # Performance tracking
    forward_time = 0
    backward_time = 0
    data_time = 0

    pbar = tqdm(train_loader, desc="Training")
    for l_fidel, h_fidel in pbar:
        data_start = time.time()
        l_fidel = l_fidel.to(device, non_blocking=True)
        h_fidel = h_fidel.to(device, non_blocking=True)
        data_time += time.time() - data_start

        optimizer.zero_grad(set_to_none=True)

        forward_start = time.time()
        # Ensure inputs are contiguous for better performance
        l_fidel = l_fidel.contiguous()
        h_fidel = h_fidel.contiguous()

        # Diffusion models use the conditional input (l_fidel) and target (h_fidel)
        # to calculate the loss internally
        loss = model(h_fidel, l_fidel)
        forward_time += time.time() - forward_start

        backward_start = time.time()
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        backward_time += time.time() - backward_start

        total_loss += loss.item()
        pbar.set_description(f"Train Loss: {loss.item():.6f}")

    # Log performance statistics
    print("=" * 30)
    print(f"Train Loop Statistics:")
    print(f"- Data transfer: {data_time:.2f}s")
    print(f"- Forward pass: {forward_time:.2f}s")
    print(f"- Backward pass: {backward_time:.2f}s\n")
    print("=" * 30)

    return total_loss / len(train_loader)
