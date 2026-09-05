import csv
import matplotlib.pyplot as plt

RESULTS_PATH = "results/ablation_results.csv"
PLOTS_DIR = "results/plots"


def load_results():
    with open(RESULTS_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_accuracy_vs_flops(results):
    flops = [float(r["flops_M"]) for r in results]
    acc = [float(r["accuracy"]) * 100 for r in results]
    names = [r["name"] for r in results]

    plt.figure(figsize=(8, 6))
    colors = ["#d62728" if "Teacher" in n else "#1f77b4" for n in names]
    plt.scatter(flops, acc, c=colors, s=120, zorder=3)

    for i, name in enumerate(names):
        label = name.replace("Student (keep_ratio=", "kr=").replace(")", "").replace("Teacher (no pruning)", "Teacher")
        plt.annotate(label, (flops[i], acc[i]), textcoords="offset points", xytext=(8, 8), fontsize=9)

    plt.xlabel("FLOPs (Millions)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Computational Cost (FLOPs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/accuracy_vs_flops.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/accuracy_vs_flops.png")


def plot_latency_comparison(results):
    names = [r["name"].replace("Student (keep_ratio=", "kr=").replace(")", "").replace("Teacher (no pruning)", "Teacher") for r in results]
    latency = [float(r["latency_ms"]) for r in results]
    colors = ["#d62728" if "Teacher" in n else "#1f77b4" for n in names]

    plt.figure(figsize=(8, 6))
    plt.bar(names, latency, color=colors)
    plt.ylabel("Latency (ms per image)")
    plt.title("Inference Latency Comparison")
    plt.xticks(rotation=20)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/latency_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/latency_comparison.png")


def plot_params_vs_accuracy(results):
    params = [float(r["params_M"]) for r in results]
    acc = [float(r["accuracy"]) * 100 for r in results]
    names = [r["name"].replace("Student (keep_ratio=", "kr=").replace(")", "").replace("Teacher (no pruning)", "Teacher") for r in results]
    colors = ["#d62728" if "Teacher" in n else "#1f77b4" for n in names]

    plt.figure(figsize=(8, 6))
    plt.scatter(params, acc, c=colors, s=120, zorder=3)
    for i, name in enumerate(names):
        plt.annotate(name, (params[i], acc[i]), textcoords="offset points", xytext=(8, 8), fontsize=9)

    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Model Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/params_vs_accuracy.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/params_vs_accuracy.png")


if __name__ == "__main__":
    results = load_results()
    plot_accuracy_vs_flops(results)
    plot_latency_comparison(results)
    plot_params_vs_accuracy(results)
    print("\nAll plots generated successfully.")