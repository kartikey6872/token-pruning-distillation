import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SimpleViT


def build_teacher_student():
    """
    Teacher: larger, no pruning — the accuracy reference.
    Student: smaller, with pruning enabled during forward pass.
    """
    teacher = SimpleViT(
        img_size=32, patch_size=4, num_classes=10,
        embed_dim=192, depth=8, num_heads=6  # bigger
    )
    student = SimpleViT(
        img_size=32, patch_size=4, num_classes=10,
        embed_dim=128, depth=6, num_heads=4  # smaller, matches our earlier tests
    )
    return teacher, student


def distillation_loss(student_logits, teacher_logits, true_labels, temperature=4.0, alpha=0.5):
    """
    Combines:
    - Hard loss: standard cross-entropy against true labels
    - Soft loss: KL divergence between student and teacher's softened outputs

    alpha: weight balance between hard and soft loss (0.5 = equal weight)
    temperature: softens the probability distributions so the student learns
                 from the teacher's relative confidences, not just the top class
    """
    hard_loss = F.cross_entropy(student_logits, true_labels)

    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, log_target=True, reduction="batchmean") * (temperature ** 2)

    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss, hard_loss, soft_loss


if __name__ == "__main__":
    # sanity check with dummy data
    teacher, student = build_teacher_student()
    teacher.eval()  # teacher doesn't train, just provides targets

    dummy_images = torch.randn(4, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (4,))

    with torch.no_grad():
        teacher_logits = teacher(dummy_images)  # no pruning for teacher

    student_logits = student(dummy_images, prune_after_layer=2, keep_ratio=0.5)  # pruning for student

    total_loss, hard_loss, soft_loss = distillation_loss(student_logits, teacher_logits, dummy_labels)

    print("Teacher output shape:", teacher_logits.shape)
    print("Student output shape:", student_logits.shape)
    print("Total loss:", total_loss.item())
    print("Hard loss:", hard_loss.item())
    print("Soft loss:", soft_loss.item())