from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "forest_plots"

METRICS = {
    "F1": {"color": "#009E73", "marker": "^", "offset": -0.18},
    "AUROC": {"color": "#0072B2", "marker": "o", "offset": 0.00},
    "AUPR": {"color": "#D55E00", "marker": "s", "offset": 0.18},
}

DRUG_EMBEDDING = [
    {
        "label": "ChemBERTa-100M",
        "F1": (0.853, 0.001),
        "AUROC": (0.924, 0.003),
        "AUPR": (0.924, 0.003),
    },
    {
        "label": "UniMol2-164M",
        "F1": (0.865, 0.004),
        "AUROC": (0.932, 0.003),
        "AUPR": (0.931, 0.002),
    },
    {
        "label": "MolFormer",
        "F1": (0.867, 0.003),
        "AUROC": (0.935, 0.004),
        "AUPR": (0.934, 0.005),
    },
]

PROTEIN_EMBEDDING = [
    {
        "label": "Ankh-large",
        "F1": (0.876, 0.005),
        "AUROC": (0.942, 0.003),
        "AUPR": (0.941, 0.002),
    },
    {
        "label": "ESM2-650M",
        "F1": (0.867, 0.003),
        "AUROC": (0.935, 0.004),
        "AUPR": (0.934, 0.005),
    },
    {
        "label": "ESM3-SM",
        "F1": (0.866, 0.003),
        "AUROC": (0.934, 0.004),
        "AUPR": (0.932, 0.008),
    },
    {
        "label": "ESMC-300M",
        "F1": (0.865, 0.004),
        "AUROC": (0.934, 0.003),
        "AUPR": (0.934, 0.004),
    },
    {
        "label": "ESMC-600M",
        "F1": (0.861, 0.005),
        "AUROC": (0.931, 0.004),
        "AUPR": (0.930, 0.004),
    },
    {
        "label": "ProtBert",
        "F1": (0.862, 0.004),
        "AUROC": (0.932, 0.004),
        "AUPR": (0.930, 0.005),
    },
    {
        "label": "ProtBert-BFD",
        "F1": (0.863, 0.006),
        "AUROC": (0.934, 0.005),
        "AUPR": (0.934, 0.004),
    },
    {
        "label": "ProtT5",
        "F1": (0.864, 0.005),
        "AUROC": (0.935, 0.004),
        "AUPR": (0.936, 0.003),
    },
]

LEARNING_RATE = [
    {
        "label": "5e-5",
        "F1": (0.876, 0.005),
        "AUROC": (0.942, 0.003),
        "AUPR": (0.941, 0.002),
    },
    {
        "label": "1e-4",
        "F1": (0.871, 0.006),
        "AUROC": (0.938, 0.003),
        "AUPR": (0.936, 0.002),
    },
    {
        "label": "2e-4",
        "F1": (0.870, 0.004),
        "AUROC": (0.938, 0.004),
        "AUPR": (0.938, 0.003),
    },
]


def plot_forest_panel(data, title, selected_label, output_name, xlim=(0.84, 0.95)):
    """Plot one publication-style forest panel and save it as PDF and PNG."""
    labels = [row["label"] for row in data]
    base_y = np.arange(len(data))[::-1]
    
    figure_height = max(2.8, 0.55 * len(data) + 1.5)
    plt.figure(figsize=(7, figure_height))

    if selected_label in labels:
        selected_index = labels.index(selected_label)
        selected_y = base_y[selected_index]
        plt.axhspan(
            selected_y - 0.42,
            selected_y + 0.42,
            color="#F0E442",
            alpha=0.22,
            linewidth=0,
            zorder=0,
        )

    for metric, style in METRICS.items():
        means = np.array([row[metric][0] for row in data])
        stds = np.array([row[metric][1] for row in data])
        y = base_y + style["offset"]

        plt.errorbar(
            means,
            y,
            xerr=stds,
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=1.0,
            elinewidth=2.0,
            capsize=5.0,
            capthick=2.0,
            label=metric,
            zorder=3,)

    _, ytick_labels = plt.yticks(base_y, labels)

    plt.xlim(*xlim)
    plt.ylim(-0.65, len(data) - 0.35)
    plt.xlabel("Performance (mean ± std)", fontsize=15, labelpad=20)
    # plt.ylabel("Models", fontsize=18, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    # plt.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    plt.box(True)
    # plt.tick_params(axis="y", length=0, pad=20)
  
    plt.legend(fontsize=14, frameon=True)
       
    for tick_label in ytick_labels:
        if tick_label.get_text() == selected_label:
            tick_label.set_fontweight("bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
   
    plt.savefig(OUTPUT_DIR / f"{output_name}.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{output_name}.png", bbox_inches="tight", dpi=600)
    plt.close()


def main():
    # plt.rcParams.update(
    #     {
    #         "font.family": "sans-serif",
    #         "font.sans-serif": ["Arial", "DejaVu Sans"],
    #         "font.size": 10,
    #         "axes.labelsize": 10,
    #         "axes.titlesize": 11,
    #         "legend.fontsize": 9,
    #         "xtick.labelsize": 9,
    #         "ytick.labelsize": 9,
    #         "pdf.fonttype": 42,
    #         "ps.fonttype": 42,
    #     }
    # )

    plot_forest_panel(
        DRUG_EMBEDDING,
        title="Drug embedding \n (protein embedding: ESM2-650M)",
        selected_label="MolFormer",
        output_name="forest_drug_embedding",)
    
    plot_forest_panel(
        PROTEIN_EMBEDDING,
        title="Protein embedding \n (drug embedding: MolFormer)",
        selected_label="Ankh-large",
        output_name="forest_protein_embedding",)
    
    plot_forest_panel(
        LEARNING_RATE,
        title="Learning rate \n (MolFormer + Ankh)",
        selected_label="5e-5",
        output_name="forest_learning_rate",)

    print("Forest plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
