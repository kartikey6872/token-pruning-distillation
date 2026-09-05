import torch
import time
import csv
from thop import profile
from model import SimpleViT
from distillation import build_teacher_student
from train import get_dataloaders, evaluate as compute_accuracy

DEVICE = "cpu"


def measure_flops(model, prune_after_layer=None, keep_ratio=0.7, img_size=32):
    dummy_input = torch.randn(1, 3, img_size, img_size)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)

    wrapped = Wrapper(model)
    flops, params = profile(wrapped, inputs=(dummy_input,), verbose=False)
    return flops, params


def measure_latency(model, prune_after_layer=None, keep_ratio=0.7, img_size=32, num_runs=20):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        for _ in range(3):
            model(dummy_input, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(dummy_input, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)
    end = time.time()

    return ((end - start) / num_runs) * 1000


def measure_memory(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 ** 2)


def run_full_evaluation(model, name, test_loader, prune_after_layer=None, keep_ratio=0.7):
    print(f"\n--- Evaluating: {name} ---")
    acc = compute_accuracy(model, test_loader, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)
    flops, params = measure_flops(model, prune_after_layer, keep_ratio)
    latency = measure_latency(model, prune_after_layer, keep_ratio)
    memory = measure_memory(model)

    result = {
        "name": name,
        "keep_ratio": keep_ratio if keep_ratio else 1.0,
        "accuracy": round(acc, 4),
        "flops_M": round(flops / 1e6, 2),
        "params_M": round(params / 1e6, 3),
        "latency_ms": round(latency, 2),
        "memory_MB": round(memory, 2),
    }

    print(f"Accuracy: {result['accuracy']}")
    print(f"FLOPs: {result['flops_M']}M")
    print(f"Params: {result['params_M']}M")
    print(f"Latency: {result['latency_ms']} ms/image")
    print(f"Memory: {result['memory_MB']} MB")

    return result


if __name__ == "__main__":
    _, test_loader = get_dataloaders(subset_size=5000, batch_size=8)

    all_results = []

    # Evaluate teacher
    teacher, _ = build_teacher_student()
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth", map_location=DEVICE))
    teacher_result = run_full_evaluation(teacher, "Teacher (no pruning)", test_loader,
                                          prune_after_layer=None, keep_ratio=None)
    all_results.append(teacher_result)

    # Evaluate all student configs
    keep_ratios = [0.9, 0.7, 0.5, 0.3]
    for kr in keep_ratios:
        _, student = build_teacher_student()
        checkpoint_path = f"checkpoints/student_prune2_keep{kr}.pth"
        student.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        result = run_full_evaluation(student, f"Student (keep_ratio={kr})", test_loader,
                                      prune_after_layer=2, keep_ratio=kr)
        all_results.append(result)

    # Save to CSV
    with open("results/ablation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print("\n=== All results saved to results/ablation_results.csv ===")
    for r in all_results:
        print(r)