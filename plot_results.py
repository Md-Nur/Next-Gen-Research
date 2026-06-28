import os
import matplotlib.pyplot as plt
import numpy as np

# Create directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Data from your execution runs
categories = ["Solar (Plant A)", "Solar (Plant B)", "Biogas (Plant A)", "Biogas (Plant B)"]
lr_r2 = [0.9555, 0.9555, -0.3666, -0.5013]
xgb_r2 = [0.9404, 0.9404, -0.8750, -1.5220]
bilstm_r2 = [0.3505, 0.2475, 0.4343, 0.3840]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, lr_r2, width, label="Linear Regression", color="#6c757d")
ax.bar(x, xgb_r2, width, label="XGBoost", color="#dc3545")
ax.bar(x + width, bilstm_r2, width, label="STA-BiLSTM (Ours)", color="#0d6efd")

ax.set_ylabel("R² Score")
ax.set_title("Model Performance Comparison across Subsystems")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.grid(axis="y", linestyle=":", alpha=0.6)

# Save the plot
plot_path = "plots/model_comparison_r2.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Plot successfully saved to {plot_path}")