import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from classification.classifier import FakeImageClassifier

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load cached features
data = torch.load("cached_features/train_features.pt")

features = data["features"]
labels = data["labels"]

print("Loaded features:", features.shape)
print("Loaded labels:", labels.shape)

# -------------------------------
# Train / Validation Split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Move to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

# -------------------------------
# Model
# -------------------------------
model = FakeImageClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 30
BATCH_SIZE = 64

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    batch_count = 0

    # Shuffle training data
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    for i in range(0, len(X_train), BATCH_SIZE):
        x = X_train[i:i+BATCH_SIZE]
        y = y_train[i:i+BATCH_SIZE]

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

# -------------------------------
# Evaluation
# -------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(X_val), BATCH_SIZE):
        x = X_val[i:i+BATCH_SIZE]
        y = y_val[i:i+BATCH_SIZE]

        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Classification Report
print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

# Confusion Matrix (optional)
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# -------------------------------
# Save Model
# -------------------------------
torch.save(model.state_dict(), "fake_image_classifier.pth")
print("\n✅ Classifier trained and saved")
