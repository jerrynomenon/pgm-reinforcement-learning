import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class ExperimentTracker:
    """Tracks and saves all experiment data and visualizations"""
    def __init__(self, base_dir="results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"experiment_{self.timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        
        self.metrics = {
            "training_rewards": [],
            "routes": [],
            "vehicle_loads": [],
            "action_probs": [],
            "parameters": {}
        }

    def save_plot(self, fig, name):
        """Save matplotlib figure"""
        if fig is not None:
            fig.savefig(os.path.join(self.exp_dir, "plots", f"{name}.png"))
            plt.close(fig)
        else:
            print(f"Warning: Attempted to save None figure for {name}")

    def save_metrics(self):
        """Save all tracked metrics with proper numpy array handling"""
        metrics_copy = {}
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                metrics_copy[key] = value
            elif isinstance(value, list):
                # Convert any numpy arrays within lists to regular lists
                metrics_copy[key] = [
                    x.tolist() if isinstance(x, np.ndarray) else x 
                    for x in value
                ]
            elif isinstance(value, np.ndarray):
                metrics_copy[key] = value.tolist()
            else:
                metrics_copy[key] = value

        with open(os.path.join(self.exp_dir, "metrics.json"), "w") as f:
            json.dump(metrics_copy, f, indent=2)