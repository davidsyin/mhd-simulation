import torch
from tqdm import tqdm
import time


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch on a single device with optimized performance

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
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

    # Check if we're using physics-informed loss
    use_physics_loss = (
        hasattr(criterion, "requires_input_data") and criterion.requires_input_data
    )

    pbar = tqdm(train_loader)
    for x, y in pbar:
        data_start = time.time()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        data_time += time.time() - data_start

        optimizer.zero_grad(set_to_none=True)

        forward_start = time.time()
        x = x.contiguous()  # Ensure input is contiguous for better performance
        pred = model(x)

        if use_physics_loss:
            loss = criterion(pred, y, x)
        else:
            loss = criterion(pred, y)

        forward_time += time.time() - forward_start

        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time += time.time() - backward_start

        total_loss += loss.item()
        pbar.set_description(f"Train Loss: {loss.item():.6f}")

    print("=" * 30)
    print(f"Train Loop Statistics:")
    print(f"- Data transfer: {data_time:.2f}s")
    print(f"- Forward pass: {forward_time:.2f}s")
    print(f"- Backward pass: {backward_time:.2f}s\n")
    print("=" * 30)

    return total_loss / len(train_loader)
