"""
Fashion Classifier Using Fashion-MNIST

Goal:
Train a neural network to classify clothing images into 10 categories.

Each image:
- 28 x 28 pixels (grayscale)
- Flattened to 784 values

Pipeline:
image → tensor → flatten → model → output scores → loss → backprop → update → prediction
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1. LABEL NAMES
# -----------------------------
fashion_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# -----------------------------
# 2. TRANSFORMS (preprocessing)
# -----------------------------
# ToTensor():
#   converts image to tensor and scales pixels to [0, 1]
#
# Normalize((0.5,), (0.5,)):
#   rescales values roughly to [-1, 1]
#
# Lambda(...view(-1)):
#   flattens (1, 28, 28) → (784,)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))
])

# -----------------------------
# 3. OPTIONAL SOFTMAX
# -----------------------------
# Used only for readable probabilities at the end.
# Not needed during training with CrossEntropyLoss.
def softmax(x):
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

# -----------------------------
# 4. LOAD DATA
# -----------------------------
trainset = datasets.FashionMNIST(
    root='./data',
    download=True,
    train=True,
    transform=transform
)

testset = datasets.FashionMNIST(
    root='./data',
    download=True,
    train=False,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# -----------------------------
# 5. DEFINE MODEL
# -----------------------------
# Architecture:
# 784 → 128 → 64 → 10
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# -----------------------------
# 6. LOSS + OPTIMIZER
# -----------------------------
# CrossEntropyLoss expects raw logits (no softmax in model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# -----------------------------
# 7. TRAINING + VALIDATION
# -----------------------------
epochs = 3

for e in range(epochs):
    running_loss = 0

    # ----- TRAIN -----
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()              # clear old gradients
        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()                    # backpropagation
        optimizer.step()                   # update weights
        running_loss += loss.item()

    train_loss = running_loss / len(trainloader)

    # ----- EVALUATE ON TEST SET -----
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(testloader)
    test_accuracy = correct / total

    print(
        f"Epoch {e+1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: {test_accuracy:.4f}"
    )

# -----------------------------
# 8. VISUALIZE ONE RANDOM IMAGE
# -----------------------------
index = random.randint(0, len(testset) - 1)
image, label = testset[index]

print("\nRandom test example")
print("Index:", index)
print("True label:", fashion_labels[label])

# image is normalized and flattened, so reshape to 28x28 for display
image_2d = image.view(28, 28)

plt.imshow(image_2d, cmap='gray')
plt.title(f"True: {fashion_labels[label]}")
plt.axis("off")
plt.show()

# -----------------------------
# 9. PREDICT ON THAT IMAGE
# -----------------------------
model.eval()

# add batch dimension: (784,) → (1, 784)
image_batch = image.unsqueeze(0)

with torch.no_grad():
    output = model(image_batch)
    probs = softmax(output)

_, predicted = torch.max(probs, dim=1)

print("Predicted:", fashion_labels[predicted.item()])
print("\nClass probabilities:")
for i, p in enumerate(probs[0]):
    print(f"{fashion_labels[i]:>12}: {p.item():.4f}")