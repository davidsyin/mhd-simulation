import torch
from tqdm import tqdm
import time

from ...models.model_factory import create_model


def train_epoch_distributed(
    model, config, train_loader, optimizer, criterion, num_gpus
):
    """Train for one epoch using multiple GPUs with data parallelism.

    Optimized for better performance with timing and memory management.

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

    forward_time = 0
    backward_time = 0
    data_time = 0

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
        optimizer.zero_grad(set_to_none=True)  # More efficient gradient zeroing

        # Split batch across GPUs
        batch_size = x.size(0)
        chunk_size = batch_size // num_gpus
        chunks_x = torch.chunk(x, num_gpus)
        chunks_y = torch.chunk(y, num_gpus)

        # Process each chunk on a different GPU
        all_losses = []

        for i in range(num_gpus):
            # Send this chunk to the corresponding GPU with non-blocking transfer
            data_start = time.time()
            x_chunk = chunks_x[i].to(f"cuda:{i}", non_blocking=True)
            y_chunk = chunks_y[i].to(f"cuda:{i}", non_blocking=True)
            data_time += (time.time() - data_start) / num_gpus

            # Use the model on this GPU
            forward_start = time.time()
            x_chunk = x_chunk.contiguous()  # Ensure input is contiguous
            outputs = models[i](x_chunk)
            forward_time += (time.time() - forward_start) / num_gpus

            loss = criterion(outputs, y_chunk)

            backward_start = time.time()
            loss.backward()
            backward_time += (time.time() - backward_start) / num_gpus

            all_losses.append(loss.item())

        # Average the gradients across all models by copying them to the main model
        for param in model.parameters():
            param.grad = param.grad / num_gpus

        optimizer.step()

        # Update all other models with the new weights from the main model
        for i in range(1, num_gpus):
            models[i].load_state_dict(
                {k: v for k, v in model.state_dict().items() if k != "_metadata"}
            )

        avg_loss = sum(all_losses) / len(all_losses)
        total_loss += avg_loss
        pbar.set_description(f"Train Loss: {avg_loss:.6f}")

    print(f"Epoch Statistics:")
    print(f"- Data transfer: {data_time:.2f}s")
    print(f"- Forward pass: {forward_time:.2f}s")
    print(f"- Backward pass: {backward_time:.2f}s")

    torch.cuda.empty_cache()

    return total_loss / len(train_loader)
