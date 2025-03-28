"""
Early stopping utility to prevent overfitting during training.

This implements a callback that monitors a validation metric and stops training
when the metric stops improving for a certain number of epochs.
"""

import numpy as np


class EarlyStopping:
    """Early stopping to stop training when validation performance stops improving.

    Args:
        patience (int): Number of epochs to wait after last improvement
        min_delta (float): Minimum change to qualify as an improvement
        mode (str): 'min' for loss, 'max' for metrics like accuracy
    """

    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Determine if a new score is better than the best score
        self.is_better = self._get_comparator()

    def __call__(self, current_score):
        """Check if training should be stopped.

        Args:
            current_score (float): Current validation metric value

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _get_comparator(self):
        if self.mode == "min":
            return lambda x, y: (y - x) > self.min_delta
        elif self.mode == "max":
            return lambda x, y: (x - y) > self.min_delta
        else:
            raise ValueError(f"Mode {self.mode} not supported. Use 'min' or 'max'")


# Example usage in a training loop:
"""
early_stopping = EarlyStopping(patience=10, mode='min')

for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)
    
    if early_stopping(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break
"""
