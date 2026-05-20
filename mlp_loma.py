# mlp_loma.py


class MLP:
    W1: Array[float, 16]
    b1: Array[float, 16]
    W2: Array[float, 16]
    b2: float


def mlp_forward(x: In[float], model: In[MLP]) -> float:
    h: Array[float, 16]
    i: int = 0
    z: float = 0

    # Hidden layer with ReLU activation
    while (i < 16, max_iter := 20):
        z = model.W1[i] * x + model.b1[i]
        if z > 0.0:
            h[i] = z
        else:
            h[i] = 0.0
        i = i + 1

    # Output layer
    out: float = model.b2
    j: int = 0
    while (j < 16, max_iter := 20):
        out = out + model.W2[j] * h[j]
        j = j + 1

    return out


# Loss Function
def mse_loss(x: In[float], y: In[float], model: In[MLP]) -> float:
    pred: float = mlp_forward(x, model)
    diff: float = pred - y
    loss: float = diff * diff
    return loss


# 4. Reverse-mode AD
d_mse_loss = rev_diff(mse_loss)


def sgd_update(model: Out[MLP], d_model: In[MLP], lr: In[float]):
    i: int = 0

    while (i < 16, max_iter := 16):
        model.W1[i] = model.W1[i] - lr * d_model.W1[i]
        model.b1[i] = model.b1[i] - lr * d_model.b1[i]
        model.W2[i] = model.W2[i] - lr * d_model.W2[i]

        i = i + 1

    model.b2 = model.b2 - lr * d_model.b2
