# mlp_loma.py


class MLP:
    W1: Array[float, 16]
    b1: Array[float, 16]
    W2: Array[float, 16]
    b2: float


def sigmoid(x: In[float]) -> float:
    return 1.0 / (1.0 + exp(0.0 - x))


def mlp_logit(x: In[float], model: In[MLP]) -> float:
    h: Array[float, 16]
    i: int = 0

    while (i < 16, max_iter := 16):
        if model.W1[i] * x + model.b1[i] > 0.0:
            h[i] = model.W1[i] * x + model.b1[i]
        else:
            h[i] = 0.0
        i = i + 1

    out: float = model.b2
    j: int = 0
    while (j < 16, max_iter := 16):
        out = out + model.W2[j] * h[j]
        j = j + 1

    return out


def mlp_forward(x: In[float], model: In[MLP]) -> float:
    return sigmoid(mlp_logit(x, model))


def bce_loss(x: In[float], y: In[float], model: In[MLP]) -> float:
    logit: float = mlp_logit(x, model)
    loss: float

    if logit > 0.0:
        loss = logit - logit * y + log(1.0 + exp(0.0 - logit))
    else:
        loss = 0.0 - logit * y + log(1.0 + exp(logit))

    return loss


d_bce_loss = rev_diff(bce_loss)


def bce_loss_grad(x: In[float], y: In[float], model: In[MLP], d_model: Out[MLP]):
    dx: float
    dy: float
    d_bce_loss(x, dx, y, dy, model, d_model, 1.0)


def zero_mlp(model: Out[MLP]):
    i: int = 0

    while (i < 16, max_iter := 16):
        model.W1[i] = 0.0
        model.b1[i] = 0.0
        model.W2[i] = 0.0
        i = i + 1

    model.b2 = 0.0


def sgd_update(model: Out[MLP], d_model: In[MLP], lr: In[float]):
    i: int = 0

    while (i < 16, max_iter := 16):
        model.W1[i] = model.W1[i] - lr * d_model.W1[i]
        model.b1[i] = model.b1[i] - lr * d_model.b1[i]
        model.W2[i] = model.W2[i] - lr * d_model.W2[i]
        i = i + 1

    model.b2 = model.b2 - lr * d_model.b2
