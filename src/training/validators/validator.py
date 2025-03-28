import torch


def validate(model, val_loader, criterion, device):
    """Validate model

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            # Move to device
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)

            # Calculate loss
            loss = criterion(pred, y)

            # Log loss
            total_loss += loss.item()

    return total_loss / len(val_loader)


def validate_size_mismatch_fix(model, val_loader, criterion, device):
    """Validate model with dimension adjustment

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            # Move to device
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)

            # Handle dimension mismatch if needed
            if pred.shape[2] == 1 and y.shape[2] > 1:
                y_adjusted = y[:, :, -1:, :, :]
                loss = criterion(pred, y_adjusted)
            else:
                loss = criterion(pred, y)

            # Log loss
            total_loss += loss.item()

    return total_loss / len(val_loader)
