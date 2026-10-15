import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CHECKPOINT_DIR, BEST_MODEL_PATH, N_QUBITS
)
from dataset import get_dataloaders, get_test_loader
from model_modified import HybridQCModel, count_parameters

def compute_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(labels, preds)
    return acc, f1, auc, cm

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    avg_loss         = total_loss / len(loader)
    acc, f1, auc, cm = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, acc, f1, auc, cm

def train():
    print("="*60)
    print("  Hybrid Quantum-Classical Pleural Effusion Classifier")
    print("  (Modified: Partial Unfreeze + Residual + Separate LR)")
    print("="*60)
    print(f"  Device : {DEVICE}  |  Epochs : {NUM_EPOCHS}")
    print("="*60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader, val_loader = get_dataloaders()
    model = HybridQCModel().to(DEVICE)
    count_parameters(model)

    criterion = nn.BCELoss()

    # CHANGE 4: Separate learning rates for each component
    # Quantum layer gets 10x higher LR to compensate for small parameter-shift gradients
    # denseblock4 gets a low LR (fine-tuning, not training from scratch)
    optimizer = optim.Adam([
        {
            'params': [p for n, p in model.classical_encoder.features.named_parameters()
                       if p.requires_grad],
            'lr': 1e-5,           # very low LR for unfrozen backbone (fine-tuning)
            'name': 'backbone_unfrozen'
        },
        {
            'params': model.classical_encoder.projector.parameters(),
            'lr': LEARNING_RATE,  # 1e-4 standard
            'name': 'projector'
        },
        {
            'params': model.quantum_layer.parameters(),
            'lr': LEARNING_RATE * 10,  # 1e-3 — quantum needs higher LR
            'name': 'quantum'
        },
        {
            'params': model.classifier.parameters(),
            'lr': LEARNING_RATE,  # 1e-4 standard
            'name': 'classifier'
        },
    ], weight_decay=WEIGHT_DECAY)

    # Print LR setup
    print("\n[Optimizer] Learning rates:")
    for group in optimizer.param_groups:
        print(f"  {group['name']:<25} LR = {group['lr']}")

    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    best_val_auc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'─'*60}\n  Epoch {epoch}/{NUM_EPOCHS}\n{'─'*60}")
        model.train()
        train_loss, train_labels, train_preds = 0.0, [], []
        start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (outputs.detach().cpu().numpy() >= 0.5).astype(int)
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}]  "
                      f"Loss: {loss.item():.4f}  Time: {time.time()-start:.1f}s")

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = accuracy_score(train_labels, train_preds)
        val_loss, val_acc, val_f1, val_auc, val_cm = evaluate(model, val_loader, criterion)
        scheduler.step(val_auc)

        print(f"\n  Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss  : {val_loss:.4f}  Val Acc  : {val_acc:.4f}")
        print(f"  Val AUC   : {val_auc:.4f}  Val F1   : {val_f1:.4f}")
        print(f"  Confusion Matrix:\n{val_cm}")

        torch.save(
            {'epoch': epoch, 'model_state': model.state_dict(),
             'optim_state': optimizer.state_dict(), 'val_auc': val_auc},
            os.path.join(CHECKPOINT_DIR, f'epoch_{epoch:02d}.pth')
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✅ Best model saved (AUC: {best_val_auc:.4f})")

    print(f"\n{'='*60}\n  Training Complete! Best AUC: {best_val_auc:.4f}\n{'='*60}")

def test():
    print("\n[Test] Loading best model...")
    model = HybridQCModel().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    criterion   = nn.BCELoss()
    test_loader = get_test_loader()
    test_loss, test_acc, test_f1, test_auc, test_cm = evaluate(model, test_loader, criterion)
    print(f"\n{'='*60}\n  Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  "
          f"AUC: {test_auc:.4f}  F1: {test_f1:.4f}")
    print(f"  Confusion Matrix:\n{test_cm}\n{'='*60}")

if __name__ == '__main__':
    train()
    test()
