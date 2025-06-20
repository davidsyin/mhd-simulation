import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import datetime

from ..utils.config import load_config
from ..diffusion import create_diffusion_model
from ..data import get_diffusion_dataloaders
from .train_utils import save_checkpoint, load_checkpoint
from ..optimizers import create_optimizer, create_scheduler
from .trainers.diffusion_trainer import train_epoch_diffusion
from .validators import (
    validate_diffusion,
    validate_with_denorm_metrics_diffusion,
)
from ..visualization.diffusion_visualizer import (
    save_diffusion_samples,
    save_diffusion_comparison_grid,
)
from ..visualization.enhanced_diffusion_visualizer import save_enhanced_diffusion_plots


def train(config_path, data_path=None):
    """Training routine for diffusion models

    Args:
        config_path: Path to configuration file
        data_path: Path to dataset file (optional, overrides config)
    """
    # Set CUDA benchmark for better performance
    torch.backends.cudnn.benchmark = True

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data paths from config if not provided
    if data_path:
        if "dataset_params" not in config:
            config["dataset_params"] = {}

        config["dataset_params"]["train_path"] = os.path.join(data_path, "train.npy")
        config["dataset_params"]["val_path"] = os.path.join(data_path, "val.npy")
        config["dataset_params"]["test_path"] = os.path.join(data_path, "test.npy")

    log_dir = config.get("log_dir", "./diffusion_logs")
    os.makedirs(log_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    print(
        f"\nTensorBoard logs will be saved to {os.path.join(log_dir, 'tensorboard')}\n"
    )

    print("Creating dataloaders...")
    start_time = time.time()
    train_loader, val_loader, test_loader = get_diffusion_dataloaders(
        config, num_workers=config.get("train_loader_params", {}).get("num_workers", 4)
    )
    print(f"\nDataloaders created in {time.time() - start_time:.2f} seconds\n")

    # Print data shapes from first batch for debugging
    if train_loader:
        x_batch, y_batch = next(iter(train_loader))
        print(f"Input batch shape: {x_batch.shape}")
        print(f"Output batch shape: {y_batch.shape}\n")

    print("Creating diffusion model...")
    model = create_diffusion_model(config)
    model = model.to(device)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters\n"
    )

    optimizer = create_optimizer(model.parameters(), config)
    print(f"Using {config['optimizer_params']['optimizer_type']} optimizer\n")

    scheduler = create_scheduler(optimizer, config)
    scheduler_type = config["optimizer_params"].get("scheduler_type", "dummy")
    print(f"Using {scheduler_type} scheduler\n")

    train_params = config["train_params"]
    num_epochs = train_params.get("epochs", 1000)

    start_epoch, best_val_loss = 0, float("inf")
    best_denorm_val_rel_l2 = float("inf")
    best_denorm_val_mse = float("inf")

    if train_params.get("load_ckpt") and os.path.exists(train_params["load_ckpt"]):
        print(f"Loading checkpoint from {train_params['load_ckpt']}")
        checkpoint_data = load_checkpoint(
            train_params["load_ckpt"], model, optimizer, scheduler
        )

        if not train_params.get("is_finetune", False):
            start_epoch = checkpoint_data["epoch"]
            best_val_loss = checkpoint_data["loss"]
            best_denorm_val_rel_l2 = checkpoint_data.get(
                "denorm_loss_rel_l2", float("inf")
            )
            best_denorm_val_mse = checkpoint_data.get("denorm_loss_mse", float("inf"))
            start_epoch += 1  # Start from the next epoch

    os.makedirs(os.path.dirname(train_params["ckpt_path"]), exist_ok=True)

    print("\nStarting diffusion training loop...")
    torch.cuda.empty_cache()  # Clear memory before starting

    for epoch in range(start_epoch, num_epochs):
        torch.backends.cudnn.benchmark = True
        epoch_start_time = time.time()

        train_loss = train_epoch_diffusion(model, train_loader, optimizer, device)

        if (
            epoch != 0
            and epoch % train_params.get("val_interval", 10) == 0
            and val_loader is not None
        ):
            val_loss, denorm_val_rel_l2, denorm_val_mse = (
                validate_with_denorm_metrics_diffusion(model, val_loader, device)
            )

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Save the model if validation loss is improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    scheduler,
                    train_params["ckpt_path"],
                    denorm_loss_rel_l2=denorm_val_rel_l2,
                    denorm_loss_mse=denorm_val_mse,
                )

                print(
                    f"\nSaved best model at epoch {epoch} with val_loss {val_loss:.6f} (denorm: {denorm_val_mse:.6f}) under {train_params['ckpt_path']}\n"
                )

                # Track best denormalized metrics
                if denorm_val_rel_l2 < best_denorm_val_rel_l2:
                    best_denorm_val_rel_l2 = denorm_val_rel_l2
                    print(
                        f"New best denormalized relative L2 loss at epoch {epoch}: {denorm_val_rel_l2:.6f} (norm: {val_loss:.6f})\n"
                    )

                if denorm_val_mse < best_denorm_val_mse:
                    best_denorm_val_mse = denorm_val_mse
                    print(
                        f"New best denormalized MSE loss at epoch {epoch}: {denorm_val_mse:.6f} (norm: {val_loss:.6f})\n"
                    )

            # Log validation metrics to tensorboard
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Loss/val_denorm_rel_l2", denorm_val_rel_l2, epoch)
            tb_writer.add_scalar("Loss/val_denorm_mse", denorm_val_mse, epoch)

            epoch_time = time.time() - epoch_start_time
            print(
                f"\nEpoch {epoch} ({epoch_time:.2f}s): Train Loss: {train_loss:.6f}, "
                f"Val Loss (Model): {val_loss:.6f}, Val Loss (Denorm Rel L2): {denorm_val_rel_l2:.6f}, Val Loss (Denorm MSE): {denorm_val_mse:.6f}, "
                f"LR: {scheduler.get_last_lr()[0] if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else optimizer.param_groups[0]['lr']}\n"
            )
        else:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            epoch_time = time.time() - epoch_start_time
            print(
                f"\nEpoch {epoch} ({epoch_time:.2f}s): Train Loss: {train_loss:.6f}, "
                f"LR: {scheduler.get_last_lr()[0] if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else optimizer.param_groups[0]['lr']}\n"
            )

        # Log training metrics
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar(
            "LearningRate",
            (
                scheduler.get_last_lr()[0]
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                else optimizer.param_groups[0]["lr"]
            ),
            epoch,
        )

        # Generate and save samples periodically
        wandb_config = config.get("wandb_params", {})
        if (
            epoch % wandb_config.get("wandb_plot_freq", 50) == 0
            or epoch == 0
            or epoch == num_epochs - 1
        ) and val_loader is not None:
            # Save traditional diffusion samples
            comparison = save_diffusion_samples(
                model,
                val_loader,
                device,
                epoch,
                log_dir,
                num_samples=wandb_config.get("wandb_num_plots", 4),
                normalize=train_params.get("norm_torchvision", True),
            )

            # Save enhanced diffusion plots
            save_enhanced_diffusion_plots(
                model,
                val_loader,
                device,
                epoch,
                log_dir,
                num_samples=min(4, wandb_config.get("wandb_num_plots", 4)),
                denormalize=train_params.get("denorm_enhanced_vis", True),
            )

            # Log to tensorboard
            tb_writer.add_images("Samples_Comparison", comparison, epoch)

        print("=" * 110)
        print("\n")
        # Explicitly free up memory
        torch.cuda.empty_cache()

    print("Training finished.")
    print(f"Training Time: {time.time() - t0:.1f}s")

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    load_checkpoint(train_params["ckpt_path"], model, optimizer, scheduler)

    # Final evaluation on test set
    if test_loader:
        print("\nEvaluating on test set...")
        test_loss, denorm_test_rel_l2, denorm_test_mse = (
            validate_with_denorm_metrics_diffusion(model, test_loader, device)
        )

        print("=" * 40)
        print(f"\nFinal test loss (normalized): {test_loss:.6f}")
        print(f"Final test loss (denormalized): {denorm_test_mse:.6f}")
        print("=" * 40)

        # Log final test metrics
        tb_writer.add_scalar("Loss/test_normalized", test_loss, num_epochs)
        tb_writer.add_scalar("Loss/test_denorm_rel_l2", denorm_test_rel_l2, num_epochs)
        tb_writer.add_scalar("Loss/test_denorm_mse", denorm_test_mse, num_epochs)

        print("\nGenerating final test samples...")
        l_fidel_samples, h_fidel_samples = next(iter(test_loader))
        l_fidel_samples = l_fidel_samples[:8].to(device)
        h_fidel_samples = h_fidel_samples[:8].to(device)

        with torch.no_grad():
            samples = model.sample(l_fidel_samples)

            # Save final comparison
            save_diffusion_comparison_grid(
                l_fidel_samples,
                h_fidel_samples,
                samples,
                os.path.join(log_dir, "final_test_samples.png"),
                normalize=train_params.get("norm_torchvision", True),
            )

            # Also save enhanced plots for final results
            save_enhanced_diffusion_plots(
                model,
                test_loader,
                device,
                "final",
                log_dir,
                num_samples=4,
                normalize=train_params.get("denorm_enhanced_vis", True),
            )

    # Close tensorboard writer
    tb_writer.close()

    return model
