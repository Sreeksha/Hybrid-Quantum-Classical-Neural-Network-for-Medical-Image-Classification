import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml
from pennylane.qnn import TorchLayer
from config import N_QUBITS, N_LAYERS, CLASSICAL_OUT, FREEZE_BACKBONE, DEVICE

# ── Quantum Device ─────────────────────────────────────────────────────────────
dev = qml.device('default.qubit', wires=N_QUBITS)

# ── Quantum Circuit ────────────────────────────────────────────────────────────
@qml.qnode(dev, diff_method='parameter-shift', interface='torch')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ── Quantum Layer via TorchLayer ───────────────────────────────────────────────
def create_quantum_layer():
    weight_shapes = {
        "weights": qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    }
    return TorchLayer(quantum_circuit, weight_shapes)

# ── Classical Encoder (DenseNet-121) ──────────────────────────────────────────
class ClassicalEncoder(nn.Module):
    def __init__(self):
        super(ClassicalEncoder, self).__init__()
        densenet       = models.densenet121(pretrained=True)
        self.features  = densenet.features
        self.relu      = nn.ReLU(inplace=True)
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(1024, 256),   # CHANGE 1: wider intermediate (was 64)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 32),     # CHANGE 1: gentler compression (new layer)
            nn.ReLU(),
            nn.Linear(32, CLASSICAL_OUT)
        )

        if FREEZE_BACKBONE:
            # CHANGE 2: Partial unfreezing — freeze all, then unfreeze denseblock4
            for param in self.features.parameters():
                param.requires_grad = False

            # Unfreeze last dense block and final norm layer
            for name, param in self.features.named_parameters():
                if 'denseblock4' in name or 'norm5' in name:
                    param.requires_grad = True

            frozen   = sum(p.numel() for p in self.features.parameters() if not p.requires_grad)
            unfrozen = sum(p.numel() for p in self.features.parameters() if p.requires_grad)
            print(f"[Model] DenseNet-121 backbone partially frozen.")
            print(f"[Model] Frozen params     : {frozen:,}")
            print(f"[Model] Unfrozen params   : {unfrozen:,}  (denseblock4 + norm5)")

    def forward(self, x):
        f = self.features(x)
        f = self.relu(f)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        return self.projector(f)             # (B, 4)

# ── Hybrid Model with Residual Connection ────────────────────────────────────
class HybridQCModel(nn.Module):
    def __init__(self):
        super(HybridQCModel, self).__init__()
        self.classical_encoder = ClassicalEncoder()
        self.quantum_layer     = create_quantum_layer()
        self.classifier        = nn.Sequential(
            nn.Linear(N_QUBITS, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = self.classical_encoder(x)              # (B, 4)
        c_scaled = torch.tanh(c) * 3.14159265      # scale to [-π, π]
        q = self.quantum_layer(c_scaled)            # (B, 4) quantum output

        # CHANGE 3: Residual connection — quantum refines classical features
        # Instead of passing only quantum output, add classical + quantum
        combined = c + q                            # (B, 4) residual

        return self.classifier(combined).squeeze(1) # (B,)

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters     : {total:,}")
    print(f"[Model] Trainable parameters : {trainable:,}")

if __name__ == '__main__':
    model = HybridQCModel().to(DEVICE)
    count_parameters(model)
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    out   = model(dummy)
    print(f"[Model] Output shape  : {out.shape}")
    print(f"[Model] Output values : {out}")
