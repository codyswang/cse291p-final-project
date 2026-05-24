import ctypes

import numpy as np

rng = np.random.default_rng(42)

import compiler


def target_function(x):
    return (np.sin(3.0 * x) + 0.3 * x) > 0.0


# Generate toy dataset
dataset_size = 100
X_data = rng.uniform(-2, 2, dataset_size)
Y_data = target_function(X_data).astype(np.float32)

with open("mlp_loma.py") as f:
    structs, lib = compiler.compile(f.read(), target="c", output_filename="_code/mlp")

MLP_struct = structs["MLP"]


def init_array(size):
    arr = rng.normal(0.0, 0.2, size)
    return (ctypes.c_float * size)(*arr)


model = MLP_struct()
model.W1 = init_array(16)
model.b1 = init_array(16)
model.W2 = init_array(16)

d_model = MLP_struct()

epochs = 10_000
learning_rate = 0.01

for epoch in range(epochs):
    total_loss = 0.0

    for x, y in zip(X_data, Y_data):
        loss = lib.bce_loss(x, y, model)
        total_loss += loss

        lib.zero_mlp(ctypes.byref(d_model))
        lib.bce_loss_grad(x, y, model, ctypes.byref(d_model))
        lib.sgd_update(ctypes.byref(model), d_model, learning_rate)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss / dataset_size:.4f}")

print("Training complete. Testing model...")
test_x = rng.uniform(-2, 2)
pred_y = lib.mlp_forward(test_x, model)
print(
    f"Input: {test_x:.2f} | Target: {float(target_function(test_x)):.0f} | Prediction: {pred_y:.4f}"
)
