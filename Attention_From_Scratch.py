#!/usr/bin/env python3
# ============================================
# Build Attention From Scratch — 5-Lesson Kit
# Dependencies: only Python stdlib (math, random)
# ============================================

import math, random
random.seed(42)

# ------------------------------------------------------------
# Utility helpers (no external libs)
# ------------------------------------------------------------

def zeros_like(A):
    if isinstance(A[0], list):  # matrix
        return [[0.0 for _ in row] for row in A]
    return [0.0 for _ in A]

def mat_shape(A):
    if isinstance(A[0], list): return (len(A), len(A[0]))
    return (len(A), )

def matmul(A, B):
    """Matrix multiply A (m x n) by B (n x p) -> (m x p)."""
    m, n = len(A), len(A[0])
    assert len(B) == n, f"shape mismatch {mat_shape(A)} @ {mat_shape(B)}"
    p = len(B[0])
    out = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            for j in range(p):
                out[i][j] += aik * B[k][j]
    return out

def transpose(A):
    return [list(row) for row in zip(*A)]

def add(A, B):
    if isinstance(A[0], list):
        return [[a+b for a,b in zip(ra, rb)] for ra, rb in zip(A,B)]
    return [a+b for a,b in zip(A,B)]

def sub(A, B):
    if isinstance(A[0], list):
        return [[a-b for a,b in zip(ra, rb)] for ra, rb in zip(A,B)]
    return [a-b for a,b in zip(A,B)]

def scalar_mul(A, s):
    if isinstance(A[0], list):
        return [[a*s for a in row] for row in A]
    return [a*s for a in A]

def dot(u, v):
    return sum(ui*vi for ui,vi in zip(u,v))

def norm(v):
    return math.sqrt(max(1e-12, dot(v,v)))

def cosine_similarity(u, v):
    return dot(u, v) / (norm(u) * norm(v))

def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])

def one_hot(idx, n):
    v = [0.0]*n
    v[idx] = 1.0
    return v

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    Z = sum(exps)
    return [e / Z for e in exps]

def softmax_rows(M):
    """Row-wise softmax for a 2D list."""
    return [softmax(row) for row in M]

def cross_entropy(p, q):
    """p,q are probability vectors"""
    eps = 1e-12
    return -sum(pi * math.log(max(qi, eps)) for pi, qi in zip(p,q))

def cross_entropy_rows(P, Q):
    """Row-wise CE over batches"""
    return sum(cross_entropy(p, q) for p, q in zip(P, Q)) / len(P)

def gelu(x):
    # Approximate GELU
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0/math.pi)*(x + 0.044715*(x**3))))

def relu(x):
    return x if x > 0 else 0.0

def relu_vec(v):
    return [relu(x) for x in v]

def apply_rowwise(A, fn):
    return [[fn(x) for x in row] for row in A]

def randn_matrix(m, n, std=0.02):
    return [[random.gauss(0.0, std) for _ in range(n)] for _ in range(m)]

# ============================================================
# LESSON 1 — Linear Algebra Foundations
# ============================================================

def lesson1_demo():
    print("\n--- LESSON 1: Linear Algebra Foundations ---")
    # Example vectors
    u = [1, 2, 3]
    v = [2, 1, 0]

    print("dot(u,v) =", dot(u,v))
    print("norm(u) =", norm(u))
    print("cosine_similarity(u,v) =", cosine_similarity(u,v))

    # Example matrices
    A = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]  # shape (3x2)
    B = [
        [7, 8, 9],
        [10, 11, 12],
    ]  # shape (2x3)

    C = matmul(A,B)  # (3x3)
    print("A @ B =", C)
    print("transpose(C) =", transpose(C))

    # TODO: Try projecting a vector onto another and verify with dot-product identities.

# ============================================================
# LESSON 2 — Probability & Softmax
# ============================================================

def lesson2_demo():
    print("\n--- LESSON 2: Probability & Softmax ---")
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    target = one_hot(0, 3)

    print("logits:", logits)
    print("softmax(logits):", probs)
    print("cross_entropy(target, probs):", cross_entropy(target, probs))

    # Row-wise demo
    batch_logits = [
        [1.0, -1.0, 0.0],
        [0.1, 0.2, 0.3],
    ]
    batch_probs = softmax_rows(batch_logits)
    batch_targets = [one_hot(2,3), one_hot(1,3)]
    print("batch softmax:", batch_probs)
    print("batch CE:", cross_entropy_rows(batch_targets, batch_probs))

    # TODO: Implement KL divergence and compare CE with true/empirical distributions.

# ============================================================
# LESSON 3 — Calculus & Optimization (Backprop Basics)
# ============================================================

def gradient_descent_1d(f, df, x0, lr=0.1, steps=25):
    x = x0
    for t in range(steps):
        grad = df(x)
        x = x - lr * grad
        # print(f"step {t:02d}: x={x:.4f}, f(x)={f(x):.4f}")
    return x

def numeric_grad(f, x, eps=1e-5):
    return (f(x+eps) - f(x-eps)) / (2*eps)

def lesson3_demo():
    print("\n--- LESSON 3: Calculus & Optimization ---")
    # Minimize f(x) = (x - 3)^2
    f  = lambda x: (x-3)**2
    df = lambda x: 2*(x-3)

    x_star = gradient_descent_1d(f, df, x0=0.0, lr=0.2, steps=30)
    print("argmin x* ≈", round(x_star, 4), "f(x*)=", round(f(x_star), 8))
    print("numeric_grad at x*=3 ->", numeric_grad(f, 3.0))

    # TODO: Extend to 2D: gradient descent on f(w) = ||Aw - b||^2 with your matmul.

# ============================================================
# LESSON 4 — Tiny Neural Net From Scratch (XOR)
# ============================================================

def init_layer(in_dim, out_dim, std=0.5):
    W = randn_matrix(out_dim, in_dim, std=std)  # rows=out_dim
    b = [[0.0] for _ in range(out_dim)]
    return W, b

def lin_forward(X, W, b):
    # X: (N x D), W: (O x D), b: (O x 1)
    out = add(matmul(X, transpose(W)), transpose(b))  # (N x O)
    return out

def relu_forward(X):
    return apply_rowwise(X, relu)

def softmax_forward(X):
    return softmax_rows(X)

def mse_rows(P, T):
    N = len(P); D = len(P[0])
    return sum(sum((pi - ti)**2 for pi,ti in zip(p,t)) for p,t in zip(P,T)) / (N*D)

def xor_dataset():
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    Y = [
        [1.0, 0.0],  # class 0
        [0.0, 1.0],  # class 1
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    return X, Y

def lesson4_train_xor(epochs=2000, lr=0.1, hidden=4):
    print("\n--- LESSON 4: Tiny NN From Scratch (XOR) ---")
    X, T = xor_dataset()      # X: (4x2), T: (4x2 one-hot)
    D_in, D_h, D_out = 2, hidden, 2

    # Init layers
    W1, b1 = init_layer(D_in, D_h, std=0.8)  # (h x 2), (h x 1)
    W2, b2 = init_layer(D_h, D_out, std=0.8) # (2 x h), (2 x 1)

    for epoch in range(epochs):
        # Forward
        Z1 = lin_forward(X, W1, b1)        # (4 x h)
        A1 = relu_forward(Z1)
        Z2 = lin_forward(A1, W2, b2)       # (4 x 2)
        Y  = softmax_forward(Z2)           # (4 x 2)

        # Loss (Cross-Entropy)
        loss = cross_entropy_rows(T, Y)

        # Backprop (manual, small network)
        # dL/dZ2 = Y - T   (from softmax + CE)
        dZ2 = [[(y - t) for y,t in zip(yrow, trow)] for yrow, trow in zip(Y, T)]  # (4 x 2)

        # Grad W2 = A1^T @ dZ2
        dW2 = matmul(transpose(A1), dZ2)   # (h x 2)
        db2 = [[sum(col)] for col in transpose(dZ2)]  # (2 x 1)

        # Back to layer1: dA1 = dZ2 @ W2^T
        dA1 = matmul(dZ2, W2)              # (4 x h)
        # ReLU backprop: dZ1 = dA1 * (Z1 > 0)
        dZ1 = []
        for i in range(len(Z1)):
            dZ1.append([dA1[i][j] * (1.0 if Z1[i][j] > 0 else 0.0) for j in range(len(Z1[0]))])

        # Grad W1 = X^T @ dZ1
        dW1 = matmul(transpose(X), dZ1)    # (2 x h) -> need (h x 2), so transpose
        dW1 = transpose(dW1)               # (h x 2)
        db1 = [[sum(col)] for col in transpose(dZ1)]  # (h x 1)

        # Gradient step
        W2 = sub(W2, scalar_mul(dW2, lr))
        b2 = sub(b2, scalar_mul(db2, lr))
        W1 = sub(W1, scalar_mul(dW1, lr))
        b1 = sub(b1, scalar_mul(db1, lr))

        if (epoch+1) % 500 == 0:
            preds = [argmax(y) for y in Y]
            acc = sum(int(p == argmax(t)) for p,t in zip(preds, T)) / len(T)
            print(f"epoch {epoch+1:4d} | loss {loss:.4f} | acc {acc:.2f}")

    # Final sanity check
    print("Predictions after training:")
    for x, y in zip(X, Y):
        print(f" X={x} -> P={ [round(v,3) for v in y] } -> class={argmax(y)}")
    # TODO: Try sigmoid/tanh and observe optimization differences.

# ============================================================
# LESSON 5 — Scaled Dot-Product Attention
# ============================================================

def causal_mask(n):
    """Lower-triangular mask (n x n) with 0 above diagonal, 1 elsewhere."""
    return [[1.0 if j <= i else 0.0 for j in range(n)] for i in range(n)]

def masked_fill(M, mask, fill_value=-1e9):
    out = []
    for i in range(len(M)):
        row = []
        for j in range(len(M[0])):
            row.append(M[i][j] if mask[i][j] > 0.5 else fill_value)
        out.append(row)
    return out

def attention(Q, K, V, causal=False):
    """
    Q: (T x d), K: (T x d), V: (T x dv)
    returns: (T x dv)
    """
    d = len(Q[0])
    scores = matmul(Q, transpose(K))               # (T x T)
    scores = [[s / math.sqrt(d) for s in row] for row in scores]
    if causal:
        M = causal_mask(len(scores))
        scores = masked_fill(scores, M, fill_value=-1e9)
    weights = softmax_rows(scores)                 # (T x T)
    out = matmul(weights, V)                       # (T x dv)
    return out, weights

def linear_project(X, W):
    # X: (T x d_in), W: (d_in x d_out)  -> (T x d_out)
    return matmul(X, W)

def lesson5_demo_attention(T=3, d=4, dv=4, causal=True):
    print("\n--- LESSON 5: Scaled Dot-Product Attention ---")
    # Toy token embeddings (sequence length T, dim d)
    X = [[random.gauss(0.0, 1.0) for _ in range(d)] for _ in range(T)]

    # Learnable projections (random init for demo)
    Wq = randn_matrix(d, d, std=0.4)
    Wk = randn_matrix(d, d, std=0.4)
    Wv = randn_matrix(d, dv, std=0.4)

    Q = linear_project(X, Wq)   # (T x d)
    K = linear_project(X, Wk)   # (T x d)
    V = linear_project(X, Wv)   # (T x dv)

    Y, W = attention(Q, K, V, causal=causal)

    print("X embeddings (first row):", [round(x,3) for x in X[0]])
    print("Q.K^T / sqrt(d) (scores):")
    for row in matmul(Q, transpose(K)):
        print([round(s/math.sqrt(d),3) for s in row])
    print("Attention weights (rows sum to 1):")
    for row in W:
        print([round(w,3) for w in row])
    print("Output (context) vector for each token:")
    for row in Y:
        print([round(y,3) for y in row])

    # TODO: Try without causal masking. Observe how token t attends to future tokens.

# ============================================================
# RUN ALL DEMOS
# ============================================================

if __name__ == "__main__":
    lesson1_demo()
    lesson2_demo()
    lesson3_demo()
    lesson4_train_xor(epochs=2000, lr=0.1, hidden=4)
    lesson5_demo_attention(T=4, d=6, dv=6, causal=True)
