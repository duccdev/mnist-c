import numpy as np

INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 0.005


def cce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    cross_entropy = -np.sum(y_true * np.log(y_pred), dtype=np.float64)
    return np.mean(cross_entropy)


def d_cce(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return y_pred - y_true


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def one_hot(x: int, labels: int) -> np.ndarray:
    y = np.zeros(labels)
    y[x] = 1
    return y


with np.load("mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"] / 255.0, f["y_train"]
    x_test, y_test = f["x_test"] / 255.0, f["y_test"]

weights = np.random.rand(OUTPUT_SIZE, INPUT_SIZE) - 0.5
bias = np.zeros(OUTPUT_SIZE)


def predict(x: np.ndarray) -> np.ndarray:
    return softmax(weights.dot(x) + bias)


for epoch in range(EPOCHS):
    loss = 0.0

    for x, y in zip(x_train, y_train):
        x_flat = x.flatten()
        y_true = one_hot(y, OUTPUT_SIZE)

        y_pred = predict(x_flat)
        loss += cce(y_true, y_pred)

        error = y_pred - y_true
        grad_w = np.outer(error, x_flat)
        grad_b = error

        weights -= LEARNING_RATE * grad_w
        bias -= LEARNING_RATE * grad_b

    print(f"Epoch {epoch + 1} | Loss: {loss / len(x_train)}")

while True:
    sample = int(input("sample="))

    x_flat = x_test[sample].flatten()
    y_pred = predict(x_flat)
    label_pred = np.argmax(y_pred)
    y_true = y_test[sample]

    print(f"{y_pred=}")
    print(f"{label_pred=}")
    print(f"{y_true=}")
    print(f"confidence={y_pred[label_pred]:.2f}")
