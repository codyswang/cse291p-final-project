import contextlib
import ctypes
import io
import compiler
import numpy as np


def target_function(x):
    return (np.sin(3.0 * x) + 0.3 * x) > 0.0


def init_array(rng, size, scale):
    values = rng.normal(0.0, scale, size).astype(np.float32)
    return (ctypes.c_float * size)(*values)


def average_loss(lib, model, x_data, y_data):
    total = 0.0
    for x, y in zip(x_data, y_data):
        total += lib.bce_loss(float(x), float(y), model)
    return total / len(x_data)


def accuracy(lib, model, x_data, y_data):
    correct = 0
    for x, y in zip(x_data, y_data):
        pred = lib.mlp_forward(float(x), model)
        if (pred >= 0.5) == (y >= 0.5):
            correct += 1
    return correct / len(x_data)


def main():
    rng = np.random.default_rng(7)

    train_x = rng.uniform(-2.0, 2.0, 128).astype(np.float32)
    train_y = target_function(train_x).astype(np.float32)
    test_x = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
    test_y = target_function(test_x).astype(np.float32)

    with open("mlp_loma.py") as f:
        loma_code = f.read()

    compile_log = io.StringIO()
    with contextlib.redirect_stdout(compile_log):
        structs, lib = compiler.compile(loma_code, target="c", output_filename="_code/mlp_example")

    MLP = structs["MLP"]
    model = MLP()
    model.W1 = init_array(rng, 16, 1.0)
    model.b1 = init_array(rng, 16, 0.2)
    model.W2 = init_array(rng, 16, 0.2)
    model.b2 = 0.0

    d_model = MLP()
    learning_rate = 0.03
    epochs = 400

    initial_loss = average_loss(lib, model, train_x, train_y)

    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for idx in order:
            x = float(train_x[idx])
            y = float(train_y[idx])
            lib.zero_mlp(ctypes.byref(d_model))
            lib.bce_loss_grad(x, y, model, ctypes.byref(d_model))
            lib.sgd_update(ctypes.byref(model), d_model, learning_rate)

    final_loss = average_loss(lib, model, train_x, train_y)
    train_acc = accuracy(lib, model, train_x, train_y)
    test_acc = accuracy(lib, model, test_x, test_y)

    print(f"Initial BCE loss: {initial_loss:.4f}")
    print(f"Final BCE loss: {final_loss:.4f}")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    if final_loss >= initial_loss:
        raise RuntimeError("Loss did not decrease")
    if test_acc < 0.75:
        raise RuntimeError("Test accuracy is too low")


if __name__ == "__main__":
    main()
