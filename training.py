
import numpy as np
import matplotlib.pyplot as plt
# DATASET
X = np.array([
    [0.2, 0.8, 0.1],
    [0.9, 0.2, 0.7],
    [0.4, 0.6, 0.3],
    [0.7, 0.1, 0.9]
], dtype=np.float64)

Y = np.array([
    [0],
    [1],
    [0],
    [1]
], dtype=np.float64)

# INIZIALIZZAZIONE
rng = np.random.default_rng(42)

W1 = rng.standard_normal((3, 4)) * np.sqrt(2 / 3)
b1 = np.zeros((1, 4))

W2 = rng.standard_normal((4, 1)) * np.sqrt(1 / 4)
b2 = np.zeros((1, 1))

# FUNZIONI DI ATTIVAZIONE
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# LOSS FUNCTION
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return np.mean(loss)

# TRAINING
learning_rate = 0.1
epochs = 1000

loss_history = []

for epoch in range(epochs):

    # 1. FORWARD PROPAGATION
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    Y_pred = sigmoid(Z2)

    # 2. CALCOLO DELLA LOSS
    loss = binary_cross_entropy(Y, Y_pred)
    loss_history.append(loss)

    # 3. BACKPROPAGATION
    m = X.shape[0]

    dZ2 = (Y_pred - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # 4. GRADIENT DESCENT
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # 5. MONITORAGGIO
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

# PREVISIONI FINALI CON I PESI AGGIORNATI
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
Y_pred = sigmoid(Z2)

print("\nLoss finale:", binary_cross_entropy(Y, Y_pred))
print("\nProbabilità finali:")
print(Y_pred)

print("\nValori reali:")
print(Y)

plt.figure(figsize=(9, 5))

plt.plot(loss_history)

plt.xlabel("Epoche")
plt.ylabel("Binary Cross-Entropy")
plt.title("Andamento della loss durante il training")

plt.grid(True)
plt.show()
