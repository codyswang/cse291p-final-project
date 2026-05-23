import ctypes

import numpy as np

rng = np.random.default_rng(42)

import compiler


def target_function(x):
    return np.sin(3.0 * x) + 0.3 * (x**2)


# Generate toy dataset
dataset_size = 100
X_data = rng.uniform(-2, 2, dataset_size)
Y_data = target_function(X_data)

with open("mlp_loma.py") as f:
    structs, lib = compiler.compile(f.read(), target="c", output_filename="_code/mlp")

MLP_struct = structs["MLP"]


def init_array(size):
    arr = rng.random(size)
    return (ctypes.c_float * size)(*arr)


model = MLP_struct()
model.W1 = init_array(16)
model.b1 = init_array(16)
model.W2 = init_array(16)

d_model = MLP_struct()

epochs = 10_000
learning_rate = 0.0001

for epoch in range(epochs):
    total_loss = 0.0

    for x, y in zip(X_data, Y_data):
        loss = lib.mse_loss(x, y, model)
        total_loss += loss

        dx = ctypes.c_float(0)
        dy = ctypes.c_float(0)

        lib.d_mse_loss(
            x,
            ctypes.byref(dx),
            y,
            ctypes.byref(dy),
            model,
            ctypes.byref(d_model),
            loss,
        )

        lib.sgd_update(model, d_model, learning_rate)

        # Zero gradients
        ctypes.memset(ctypes.byref(d_model), 0, ctypes.sizeof(d_model))

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss / dataset_size:.4f}")

print("Training complete. Testing model...")
test_x = rng.uniform(-2, 2)
pred_y = lib.mlp_forward(test_x, model)
print(
    f"Input: {test_x:.2f} | Target: {target_function(test_x):.4f} | Prediction: {pred_y:.4f}"
)
