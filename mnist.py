import matplotlib.pyplot as plt
import numpy as np
import struct
import os

# ------------------ 激活函数 ------------------


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(y_pred, y_true):
    m = y_pred.shape[0]
    return -np.sum(np.log(y_pred[np.arange(m), y_true] + 1e-9)) / m


def cross_entropy_grad(y_pred, y_true):
    grad = y_pred.copy()
    grad[np.arange(len(y_true)), y_true] -= 1
    return grad / len(y_true)

# ------------------ 数据加载 ------------------


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, 1, rows, cols).astype(np.float32) / 255.0
        return images


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# ------------------ 卷积/池化 ------------------


def conv2d(x, w, b, stride=1, padding=0):
    n, c, h, w_ = x.shape
    f, _, kh, kw = w.shape
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w_ + 2 * padding - kw) // stride + 1
    out = np.zeros((n, f, h_out, w_out))

    x_padded = np.pad(x, ((0,), (0,), (padding,), (padding,)), mode='constant')

    for i in range(h_out):
        for j in range(w_out):
            region = x_padded[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[:, :, i, j] = np.tensordot(
                region, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def conv2d_backward(dout, x, w, stride=1, padding=0):
    n, c, h, w_ = x.shape
    f, _, kh, kw = w.shape
    x_padded = np.pad(x, ((0,), (0,), (padding,), (padding,)), mode='constant')
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(w[:, 0, 0, 0])

    h_out, w_out = dout.shape[2], dout.shape[3]

    for i in range(h_out):
        for j in range(w_out):
            region = x_padded[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            for n_idx in range(n):
                for f_idx in range(f):
                    dw[f_idx] += region[n_idx] * dout[n_idx, f_idx, i, j]
                    dx_padded[n_idx, :, i*stride:i*stride+kh, j*stride:j *
                              stride+kw] += w[f_idx] * dout[n_idx, f_idx, i, j]
    db = np.sum(dout, axis=(0, 2, 3))
    dx = dx_padded[:, :, padding:h+padding, padding:w_+padding]
    return dx, dw, db


def avg_pool(x, size=2):
    n, c, h, w = x.shape
    out = x.reshape(n, c, h//size, size, w//size, size).mean(axis=(3, 5))
    return out


def avg_pool_backward(dout, x, size=2):
    n, c, h, w = x.shape
    dx = np.zeros_like(x)
    for i in range(h//size):
        for j in range(w//size):
            dx[:, :, i*size:(i+1)*size, j*size:(j+1) *
               size] += dout[:, :, i:i+1, j:j+1] / (size * size)
    return dx

# ------------------ LeNet 网络 ------------------


class LeNet:
    def __init__(self):
        self.conv1_w = np.random.randn(6, 1, 5, 5) * np.sqrt(2/25)
        self.conv1_b = np.zeros(6)
        self.conv2_w = np.random.randn(16, 6, 5, 5) * np.sqrt(2/150)
        self.conv2_b = np.zeros(16)

        self.fc1_w = np.random.randn(400, 120) * np.sqrt(2/400)
        self.fc1_b = np.zeros(120)
        self.fc2_w = np.random.randn(120, 84) * np.sqrt(2/120)
        self.fc2_b = np.zeros(84)
        self.fc3_w = np.random.randn(84, 10) * np.sqrt(2/84)
        self.fc3_b = np.zeros(10)

    def forward(self, x):
        self.x = x
        self.z1 = conv2d(x, self.conv1_w, self.conv1_b, padding=2)
        self.a1 = leaky_relu(self.z1)
        self.p1 = avg_pool(self.a1)

        self.z2 = conv2d(self.p1, self.conv2_w, self.conv2_b)
        self.a2 = leaky_relu(self.z2)
        self.p2 = avg_pool(self.a2)

        self.flat = self.p2.reshape(x.shape[0], -1)

        self.z3 = self.flat @ self.fc1_w + self.fc1_b
        self.a3 = leaky_relu(self.z3)
        self.z4 = self.a3 @ self.fc2_w + self.fc2_b
        self.a4 = leaky_relu(self.z4)
        self.z5 = self.a4 @ self.fc3_w + self.fc3_b
        self.out = softmax(self.z5)
        return self.out

    def backward(self, y_true, lr=0.01):
        dout = cross_entropy_grad(self.out, y_true)

        dz5 = dout
        d_fc3_w = self.a4.T @ dz5
        d_fc3_b = np.sum(dz5, axis=0)
        da4 = dz5 @ self.fc3_w.T

        dz4 = da4 * leaky_relu_deriv(self.z4)
        d_fc2_w = self.a3.T @ dz4
        d_fc2_b = np.sum(dz4, axis=0)
        da3 = dz4 @ self.fc2_w.T

        dz3 = da3 * leaky_relu_deriv(self.z3)
        d_fc1_w = self.flat.T @ dz3
        d_fc1_b = np.sum(dz3, axis=0)
        d_flat = dz3 @ self.fc1_w.T
        d_p2 = d_flat.reshape(self.p2.shape)

        d_a2 = avg_pool_backward(d_p2, self.a2)
        d_z2 = d_a2 * leaky_relu_deriv(self.z2)
        d_p1, d_conv2_w, d_conv2_b = conv2d_backward(
            d_z2, self.p1, self.conv2_w)

        d_a1 = avg_pool_backward(d_p1, self.a1)
        d_z1 = d_a1 * leaky_relu_deriv(self.z1)
        _, d_conv1_w, d_conv1_b = conv2d_backward(
            d_z1, self.x, self.conv1_w, padding=2)

        # 更新
        self.fc3_w -= lr * d_fc3_w
        self.fc3_b -= lr * d_fc3_b
        self.fc2_w -= lr * d_fc2_w
        self.fc2_b -= lr * d_fc2_b
        self.fc1_w -= lr * d_fc1_w
        self.fc1_b -= lr * d_fc1_b
        self.conv2_w -= lr * d_conv2_w
        self.conv2_b -= lr * d_conv2_b
        self.conv1_w -= lr * d_conv1_w
        self.conv1_b -= lr * d_conv1_b


# ------------------ 训练 ------------------


def train(model, X, y, batch_size=64, epochs=5, lr=0.01):
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        total_loss = 0
        total_correct = 0
        batches = 0

        for i in range(0, min(6400, len(X)), batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            out = model.forward(xb)
            loss = cross_entropy(out, yb)
            model.backward(yb, lr)

            pred = np.argmax(out, axis=1)
            total_correct += np.sum(pred == yb)
            total_loss += loss
            batches += 1

            correct = np.sum(pred == yb)
            acc = correct / len(yb)

        avg_loss = total_loss / batches
        acc = total_correct / min(6400, len(X))

        loss_history.append(avg_loss)
        acc_history.append(acc)

        print(
            f"[Epoch {epoch+1}]  Acc: {acc*100:.2f}%,  Loss: {avg_loss:.4f}")

    # 📊 画图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, marker='o')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot([x * 100 for x in acc_history], marker='s', color='green')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.show()


# ------------------ 运行主程序 ------------------
images = load_mnist_images('data/train-images-idx3-ubyte')[:6400]
labels = load_mnist_labels('data/train-labels-idx1-ubyte')[:6400]

model = LeNet()
train(model, images, labels, epochs=30)
