import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os

from model import SimpleViT
from distillation import build_teacher_student, distillation_loss

DEVICE = "cpu"
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_dataloaders(subset_size=5000, batch_size=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_subset = Subset(full_train, range(subset_size))
    test_subset = Subset(full_test, range(min(1500, len(full_test))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def evaluate(model, loader, prune_after_layer=None, keep_ratio=0.7):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_teacher(teacher, train_loader, test_loader, epochs=20, lr=1e-3):
    print("\n=== Training Teacher ===")
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = teacher(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(teacher, test_loader)
        print(f"Teacher Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} — Test Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(teacher.state_dict(), os.path.join(CHECKPOINT_DIR, "teacher.pth"))

    print(f"Teacher checkpoint saved. Best accuracy: {best_acc:.4f}")


def train_student(student, teacher, train_loader, test_loader, epochs=20, lr=1e-3,
                   prune_after_layer=2, keep_ratio=0.7):
    print(f"\n=== Training Student (prune_after_layer={prune_after_layer}, keep_ratio={keep_ratio}) ===")
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    teacher.eval()

    best_acc = 0.0
    checkpoint_name = f"student_prune{prune_after_layer}_keep{keep_ratio}.pth"

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)
            loss, hard_loss, soft_loss = distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(student, test_loader, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)
        print(f"Student Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} — Test Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), os.path.join(CHECKPOINT_DIR, checkpoint_name))

    print(f"Student checkpoint saved as {checkpoint_name}. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(subset_size=5000, batch_size=8)

    # Train teacher once
    teacher, _ = build_teacher_student()
    train_teacher(teacher, train_loader, test_loader, epochs=20)

    # Ablation: train multiple students with different keep_ratio values
    keep_ratios_to_try = [0.9, 0.5, 0.3]  # 0.7 already done in a previous run

    for keep_ratio in keep_ratios_to_try:
        _, student = build_teacher_student()  # fresh student each time
        train_student(student, teacher, train_loader, test_loader, epochs=20,
                      prune_after_layer=2, keep_ratio=keep_ratio)