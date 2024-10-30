import numpy as np

with np.load("mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]


with open("mnist.bin", "wb") as f:
    f.write(x_train.flatten().astype(np.uint8).tobytes())
    f.write(y_train.flatten().astype(np.uint8).tobytes())
    f.write(x_test.flatten().astype(np.uint8).tobytes())
    f.write(y_test.flatten().astype(np.uint8).tobytes())
