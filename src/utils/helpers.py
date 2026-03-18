"""Saving utilities."""

from pathlib import Path
import matplotlib.pyplot as plt


def save_or_create(plt: plt.Figure, save_path: str):
    """Save a plot to a file, creating the directory if it does not exist.

    Args:
        plt (plt.Figure): The matplotlib figure to save.
        save_path (str): The path where the plot will be saved.

    Returns:
        None
    """
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_p)
