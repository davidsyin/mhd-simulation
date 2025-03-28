import torch
from tqdm import tqdm

from ...models.model_factory import create_model


def train_epoch_distributed(
    model, config, train_loader, optimizer, criterion, num_gpus
):
    """Train for one epoch using multiple GPUs with data parallelism.

    Args:
        model: Main model on the primary GPU
        config: Configuration dictionary
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        num_gpus: Number of GPUs to use

    Returns:
        float: Average training loss for the epoch
    """

    model.train()
    total_loss = 0

    # Create model copies for each GPU in advance
    models = [model]  # First GPU already has the model
    for i in range(1, num_gpus):
        temp_model = create_model(config).to(f"cuda:{i}")
        # Copy weights properly without metadata
        temp_model.load_state_dict(
            {k: v for k, v in model.state_dict().items() if k != "_metadata"}
        )
        models.append(temp_model)

    pbar = tqdm(train_loader)
    for x, y in pbar:
        optimizer.zero_grad()

        # Split batch across GPUs
        batch_size = x.size(0)
        chunk_size = batch_size // num_gpus
        chunks_x = torch.chunk(x, num_gpus)
        chunks_y = torch.chunk(y, num_gpus)

        # Process each chunk on a different GPU
        all_losses = []

        for i in range(num_gpus):
            # Send this chunk to the corresponding GPU
            x_chunk = chunks_x[i].to(f"cuda:{i}")
            y_chunk = chunks_y[i].to(f"cuda:{i}")

            # Use the model on this GPU
            outputs = models[i](x_chunk)

            # Calculate loss
            loss = criterion(outputs, y_chunk)
            loss.backward()

            all_losses.append(loss.item())

        # Average the gradients across all models by copying them to the main model
        for param in model.parameters():
            param.grad = param.grad / num_gpus

        # Update parameters
        optimizer.step()

        # Update all other models with the new weights from the main model
        for i in range(1, num_gpus):
            models[i].load_state_dict(
                {k: v for k, v in model.state_dict().items() if k != "_metadata"}
            )

        # Calculate average loss for logging
        avg_loss = sum(all_losses) / len(all_losses)
        total_loss += avg_loss
        pbar.set_description(f"Train Loss: {avg_loss:.6f}")

        # Free memory
        torch.cuda.empty_cache()

    return total_loss / len(train_loader)


def train_epoch_distributed_size_mismatch_fix(
    model, config, train_loader, optimizer, criterion, num_gpus
):
    """Train for one epoch using multiple GPUs with handling for size mismatches.

    This version handles cases where the model output and target dimensions don't match.

    Args:
        model: Main model on the primary GPU
        config: Configuration dictionary
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        num_gpus: Number of GPUs to use

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0

    # Create model copies for each GPU in advance
    models = [model]  # First GPU already has the model
    for i in range(1, num_gpus):
        temp_model = create_model(config).to(f"cuda:{i}")
        # Copy weights properly without metadata
        temp_model.load_state_dict(
            {k: v for k, v in model.state_dict().items() if k != "_metadata"}
        )
        models.append(temp_model)

    pbar = tqdm(train_loader)
    for x, y in pbar:
        optimizer.zero_grad()

        # Split batch across GPUs
        batch_size = x.size(0)
        chunk_size = batch_size // num_gpus
        chunks_x = torch.chunk(x, num_gpus)
        chunks_y = torch.chunk(y, num_gpus)

        # Process each chunk on a different GPU
        all_losses = []

        for i in range(num_gpus):
            # Send this chunk to the corresponding GPU
            x_chunk = chunks_x[i].to(f"cuda:{i}")
            y_chunk = chunks_y[i].to(f"cuda:{i}")

            # Use the model on this GPU
            outputs = models[i](x_chunk)

            # Check output shape and adjust target if needed
            if outputs.shape[2] == 1 and y_chunk.shape[2] > 1:
                # If model outputs a single timestep but target has multiple,
                # use only the last timestep of the target
                # Assuming the last time step is the one we want to predict
                y_chunk_adjusted = y_chunk[:, :, -1:, :, :]
                print(
                    f"Adjusted target shape from {y_chunk.shape} to {y_chunk_adjusted.shape}"
                )
                loss = criterion(outputs, y_chunk_adjusted)
            else:
                loss = criterion(outputs, y_chunk)

            loss.backward()
            all_losses.append(loss.item())

        # Average the gradients across all models by scaling them
        for param in model.parameters():
            if param.grad is not None:
                param.grad = param.grad / num_gpus

        # Update parameters
        optimizer.step()

        # Update all other models with the new weights from the main model
        for i in range(1, num_gpus):
            models[i].load_state_dict(
                {k: v for k, v in model.state_dict().items() if k != "_metadata"}
            )

        # Calculate average loss for logging
        avg_loss = sum(all_losses) / len(all_losses)
        total_loss += avg_loss
        pbar.set_description(f"Train Loss: {avg_loss:.6f}")

    return total_loss / len(train_loader)
