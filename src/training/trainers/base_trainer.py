import torch
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch on a single device

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

    pbar = tqdm(train_loader)
    for x, y in pbar:
        # Move to device
        x, y = x.to(device), y.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred = model(x)

        # Calculate loss
        loss = criterion(pred, y)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Log loss
        total_loss += loss.item()
        pbar.set_description(f"Train Loss: {loss.item():.6f}")

    return total_loss / len(train_loader)
