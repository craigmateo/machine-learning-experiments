# Build a neural network to identify handwritten digits in an image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1. TRANSFORMS (preprocessing)
# -----------------------------
# Convert image → tensor (values 0–1)
# Then flatten 28x28 → 784 so it matches nn.Linear input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # (1,28,28) → (784,)
])

# -----------------------------
# 2. LOAD DATA
# -----------------------------
# MNIST = dataset of handwritten digits
# train=True → training data
# train=False → test data

trainset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
testset  = datasets.MNIST(root='./data', download=True, train=False, transform=transform)

# DataLoader = feeds data in batches (mini-batch gradient descent)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# -----------------------------
# 3. DEFINE MODEL
# -----------------------------
# Architecture: 784 → 128 → 10
# 784 inputs (pixels)
# 128 hidden units (learn features)
# 10 outputs (digits 0–9)

model = nn.Sequential(
    nn.Linear(784, 128),  # weighted sum (like dot product)
    nn.ReLU(),            # nonlinearity
    nn.Linear(128, 10)    # output scores (logits)
)

# -----------------------------
# 4. LOSS + OPTIMIZER
# -----------------------------
# CrossEntropyLoss:
# - combines softmax + log loss internally
# - expects raw outputs (no softmax applied)

criterion = nn.CrossEntropyLoss()

# SGD = gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
# Core loop:
# input → model → loss → backward → update weights

epochs = 1

for e in range(epochs):
    running_loss = 0

    for images, labels in trainloader:
        optimizer.zero_grad()              # reset gradients (important)

        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)  # compute error

        loss.backward()                    # compute gradients (backprop)
        optimizer.step()                   # update weights

        running_loss += loss.item()

    print("Training loss:", running_loss / len(trainloader))

# -----------------------------
# 6. TEST ACCURACY
# -----------------------------
# Turn off gradient tracking (faster, no training)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)

        # Pick class with highest score
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test accuracy:", correct / total)

# -----------------------------
# 7. VISUALIZE ONE IMAGE
# -----------------------------
# Load one example from dataset

image, label = trainset[0]

print("True label:", label)

# reshape back to 28x28 for display
image_2d = image.view(28, 28)

plt.imshow(image_2d, cmap='gray')
plt.title(f"Label: {label}")
plt.show()

# -----------------------------
# 8. PREDICT ON ONE IMAGE
# -----------------------------
# Model expects batch dimension → (1, 784)

image = image.unsqueeze(0)

output = model(image)

# pick highest score = predicted digit
_, predicted = torch.max(output, dim=1)

print("Predicted:", predicted.item())