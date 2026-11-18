import numpy as np

# Dataset di esempio: 4 clienti, 3 caratteristiche
X = np.array([
    [0.2, 0.8, 0.1],
    [0.9, 0.2, 0.7],
    [0.4, 0.6, 0.3],
    [0.7, 0.1, 0.9]
], dtype=np.float64)

# Etichette: 0 = cliente rimane, 1 = cliente abbandona
Y = np.array([
    [0],
    [1],
    [0],
    [1]
], dtype=np.float64) #valori normalizzati da 0 a 1

#shape ci fa sapere ordini delle matrici
print("\nDimensione X:", X.shape)
print("Dimensione Y:", Y.shape)

print("Dataset X:")
print(X)







# Numero di caratteristiche in input
n_input = 3

# Numero di neuroni nel primo hidden layer
n_hidden = 4

# Generatore di numeri casuali riproducibili
rng = np.random.default_rng(42)

# Matrice dei pesi
W1 = rng.standard_normal((n_input, n_hidden))

# Vettore dei bias
b1 = np.zeros((1, n_hidden))

print("\nMatrice dei pesi:")
print(W1)#matrice dei pesi

print("\nBias:")
print(b1)

print("\nDimensioni W1:", W1.shape)
print("Dimensioni b1:", b1.shape)








# FORWARD PROPAGATION - HIDDEN LAYER 1

# Calcolo del predittore lineare
Z1 = X @ W1 + b1

# Funzione di attivazione ReLU
A1 = np.maximum(0, Z1)

# Visualizzazione dei risultati
print("\nPredittori lineari Z1:")
print(Z1)

print("\nAttivazioni ReLU A1:")
print(A1)

print("\nDimensioni Z1:", Z1.shape)
print("Dimensioni A1:", A1.shape)




print("\nPrimo cliente - prima di ReLU:")
print(Z1[0])

print("\nPrimo cliente - dopo ReLU:")
print(A1[0])




# OUTPUT LAYER

# 4 input provenienti dall'hidden layer
# 1 neurone di output
W2 = rng.standard_normal((4, 1)) * np.sqrt(1 / 4)

# Un bias per il neurone di output
b2 = np.zeros((1, 1))

# Predittore lineare
Z2 = A1 @ W2 + b2

# Funzione di attivazione Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Probabilità previste
Y_pred = sigmoid(Z2)

print("\nProbabilità di abbandono:")
print(Y_pred)

print("\nDimensioni:", Y_pred.shape)
