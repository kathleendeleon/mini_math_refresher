import math, random, time, csv, io
import streamlit as st

random.seed(42)
st.set_page_config(page_title="Attention From Scratch — 5 Lessons", page_icon="🧠", layout="wide")

# ======================== Shared Utilities (Safe) ========================

def _assert_rectangular(M, name="matrix"):
    if not isinstance(M, list) or len(M) == 0 or not isinstance(M[0], list):
        raise ValueError(f"{name} must be a non-empty 2D list")
    cols = len(M[0])
    for r, row in enumerate(M):
        if not isinstance(row, list):
            raise ValueError(f"{name} row {r} is not a list")
        if len(row) != cols:
            raise ValueError(f"{name} is ragged at row {r}: expected {cols} columns, got {len(row)}")
    return len(M), cols  # (rows, cols)

def _shape(M):
    if isinstance(M, list):
        if len(M) == 0: return (0, 0)
        if isinstance(M[0], list): return (len(M), len(M[0]))
        return (len(M),)
    return ()

def matmul(A, B):
    """Safe matmul with rectangular checks and clear shape errors."""
    m, nA = _assert_rectangular(A, "A")
    nB_rows, p = _assert_rectangular(B, "B")
    if nA != nB_rows:
        raise ValueError(f"matmul shape mismatch: A is {m}x{nA}, B is {nB_rows}x{p} (nA must equal nB_rows)")
    out = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(nA):
            aik = A[i][k]
            rowB = B[k]
            for j in range(p):
                out[i][j] += aik * rowB[j]
    return out

def transpose(A): return [list(row) for row in zip(*A)]
def add(A, B): return [[a+b for a,b in zip(ra, rb)] for ra, rb in zip(A, B)]
def sub(A, B): return [[a-b for a,b in zip(ra, rb)] for ra, rb in zip(A, B)]
def scalar_mul(A, s): return [[a*s for a in row] for row in A]
def dot(u, v): return sum(ui*vi for ui,vi in zip(u,v))
def norm(v): return math.sqrt(max(1e-12, dot(v, v)))
def cosine_similarity(u, v): return dot(u, v) / (norm(u) * norm(v))
def argmax(xs): return max(range(len(xs)), key=lambda i: xs[i])
def one_hot(idx, n): v = [0.0]*n; v[idx] = 1.0; return v
def softmax(xs): m = max(xs); exps = [math.exp(x - m) for x in xs]; Z = sum(exps); return [e / Z for e in exps]
def softmax_rows(M): return [softmax(row) for row in M]
def cross_entropy(p, q): eps = 1e-12; return -sum(pi * math.log(max(qi, eps)) for pi, qi in zip(p, q))
def cross_entropy_rows(P, Q): return sum(cross_entropy(p, q) for p, q in zip(P, Q)) / len(P)
def relu(x): return x if x > 0 else 0.0
def apply_rowwise(A, fn): return [[fn(x) for x in row] for row in A]
def randn_matrix(m, n, std=0.02): return [[random.gauss(0.0, std) for _ in range(n)] for _ in range(m)]

# (We keep ensure_2d for other lessons if needed, but Lesson 4 won't use it to avoid warnings.)
def ensure_2d(M, rows, cols, name="matrix"):
    """Coerce M to (rows x cols) if possible; used sparingly outside Lesson 4 sandbox."""
    # Already rectangular?
    try:
        r, c = _assert_rectangular(M, name)
        if r == rows and c == cols:
            return M
        if r == cols and c == rows:
            return transpose(M)
    except Exception:
        pass

    # 1D vector → row/column
    if isinstance(M, list) and len(M) > 0 and not isinstance(M[0], list):
        if rows == 1 and len(M) == cols: return [M[:]]
        if cols == 1 and len(M) == rows: return [[x] for x in M]

    # Special case: [[x0,...,x_{rows-1}]] -> (rows x 1)
    if (isinstance(M, list) and len(M) == 1 and
        isinstance(M[0], list) and len(M[0]) == rows and cols == 1):
        return [[M[0][i]] for i in range(rows)]

    # Last resort: flatten & fill silently (to avoid yellow spam)
    vals = []
    if isinstance(M, list):
        for row in M:
            if isinstance(row, list): vals.extend(row)
            else: vals.append(row)
    else:
        vals = [M]
    out = [[0.0] * cols for _ in range(rows)]
    k = 0
    for i in range(rows):
        for j in range(cols):
            if k < len(vals):
                out[i][j] = vals[k]; k += 1
    return out

# =================== Session state for quizzes & metrics ==================

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}   # key: quiz_id, value: selected option index
if "quiz_scores" not in st.session_state:
    st.session_state.quiz_scores = {}    # key: quiz_id, value: bool correct
if "session_id" not in st.session_state:
    st.session_state.session_id = f"S{int(time.time())}-{random.randint(1000,9999)}"

def ask_mcq(quiz_id, question, options, correct_idx, help_text=None):
    """Render MCQ and store answer."""
    if help_text: st.caption(help_text)
    default_index = st.session_state.quiz_answers.get(quiz_id, 0)
    selected = st.radio(question, options, key=f"q_{quiz_id}", index=default_index)
    sel_idx = options.index(selected)
    st.session_state.quiz_answers[quiz_id] = sel_idx
    is_correct = (sel_idx == correct_idx)
    st.session_state.quiz_scores[quiz_id] = is_correct
    if is_correct: st.success("✅ Correct!")
    else:          st.info(f"ℹ️ Selected: {options[sel_idx]}")
    return sel_idx, is_correct

# ======================== Introduction (single column) ===================

def tab_introduction():
    st.subheader("What is Attention? (Executive & Technical View)")

    st.markdown("#### For Non-Technical Leaders")
    st.write("""
    **Attention is a smart prioritization mechanism.**  
    When the model reads a sentence or analyzes data, it asks:
    *“Which parts matter most for the current decision?”*  
    It **weights important pieces higher** and down-weights the rest—
    like giving more weight to key voices in a leadership meeting.
    """)
    st.markdown("**Business value:** filters noise, surfaces signal, adapts on the fly.")
    st.markdown("**Why it changed the game:** enables models to keep track of long context and nuance.")

    st.markdown("#### Real-World Applications")
    st.write("""
    - Search & Q/A, chat assistants, document summarization
    - Code completion, SQL generation
    - Customer support triage, CRM note summarization
    - Fraud/anomaly detection
    - Product recommendations
    - Vision & speech: captioning, recognition
    """)
    
    with st.expander("Limitations / Risks"):
        st.write("- **Quadratic compute** in standard attention with context length.")
        st.write("- **Hallucinations** if prompts/data are insufficient.")
        st.write("- **Bias** mirrors training data; needs evaluation.")

    st.markdown("#### Technical Summary")
    st.code("Attention(Q, K, V) = softmax(Q Kᵀ / √d) V", language="text")
    st.write("""
    - Inputs: sequence **X** ∈ ℝ^{T×d}. Project to **Q = X W_q**, **K = X W_k**, **V = X W_v**  
    - **Scores**: `S = Q Kᵀ / √d` (T×T)  
    - **Weights**: `W = softmax(S)` row-wise  
    - **Output**: `Y = W V`  
    - **Causal mask** blocks future positions for generation.
    """)
    with st.expander("Glossary"):
        st.write("- **Q (Query)**: what the current token is looking for.")
        st.write("- **K (Key)**: what each token offers as a tag/index.")
        st.write("- **V (Value)**: the content carried by each token.")
        st.write("- **Scaling (√d)**: keeps dot-products in a stable range.")

# ======================== Intro Quiz (separate tab) ======================

def tab_intro_quiz():
    st.subheader("Quick Intro Quiz")
    ask_mcq(
        "intro_1",
        "At a high level, attention lets the model…",
        ["Treat all inputs equally to be fair",
         "Focus more on relevant parts and less on irrelevant ones",
         "Memorize the entire internet"],
        correct_idx=1
    )
    ask_mcq(
        "intro_2",
        "Which of the following is **NOT** a typical application powered by attention?",
        ["Document summarization", "Machine translation", "Sorting a list with quicksort"],
        correct_idx=2
    )

# ======================== Lesson Formula Blocks ==========================

def formulas_lesson1():
    st.markdown("**Key formulas & shapes**")
    st.latex(r"\text{Dot: } \langle u,v\rangle=\sum_i u_i v_i \quad;\quad \|u\|=\sqrt{\sum_i u_i^2}")
    st.latex(r"\text{Cosine similarity: } \cos\theta=\frac{\langle u,v\rangle}{\|u\|\|v\|}")
    st.latex(r"\text{Matmul: } C=AB,\; A\in\mathbb{R}^{m\times n},\; B\in\mathbb{R}^{n\times p}\Rightarrow C\in\mathbb{R}^{m\times p}")
    st.latex(r"\text{Transpose: } (A^\top)_{ij}=A_{ji}")

def formulas_lesson2():
    st.markdown("**Key formulas**")
    st.latex(r"\text{Softmax: } p_i=\frac{e^{z_i}}{\sum_j e^{z_j}} \;=\;\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}")
    st.latex(r"\text{Cross-entropy: } H(y,p)=-\sum_i y_i\log p_i \quad (\text{with one-hot }y)")
    st.latex(r"\text{Softmax + CE gradient: } \frac{\partial H}{\partial z_i}=p_i-y_i")

def formulas_lesson3():
    st.markdown("**Key formulas**")
    st.latex(r"\text{Objective: } f(x)=(x-a)^2,\;\; f'(x)=2(x-a)")
    st.latex(r"\text{GD update: } x^{(t+1)}=x^{(t)}-\eta\,\nabla f(x^{(t)})")
    st.latex(r"\text{Smaller }\eta \text{ → stable; too large }\eta \text{ → divergence.}")

def formulas_lesson4():
    st.markdown("**2-layer NN for XOR (shapes)**")
    st.latex(r"X\in\mathbb{R}^{N\times D},\; W_1\in\mathbb{R}^{H\times D},\; b_1\in\mathbb{R}^{H\times 1},\; W_2\in\mathbb{R}^{C\times H},\; b_2\in\mathbb{R}^{C\times 1}")
    st.latex(r"Z_1=X W_1^\top + \mathbf{1} b_1^\top,\;\; A_1=\text{ReLU}(Z_1)")
    st.latex(r"Z_2=A_1 W_2^\top + \mathbf{1} b_2^\top,\;\; P=\text{softmax}(Z_2)")
    st.markdown("**Backprop (softmax+CE)**")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial Z_2}=P-Y")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial W_2}=(A_1)^\top (P-Y),\;\; \frac{\partial \mathcal{L}}{\partial b_2}=\sum_{n}(P-Y)_n")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial Z_1}=\left((P-Y)W_2\right)\odot \mathbb{1}[Z_1>0]")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial W_1}=X^\top \frac{\partial \mathcal{L}}{\partial Z_1},\;\; \frac{\partial \mathcal{L}}{\partial b_1}=\sum_{n}\left(\frac{\partial \mathcal{L}}{\partial Z_1}\right)_n")

def formulas_lesson5():
    st.markdown("**Scaled Dot-Product Attention**")
    st.latex(r"Q=XW_q,\; K=XW_k,\; V=XW_v")
    st.latex(r"S=\frac{QK^\top}{\sqrt{d}},\quad W=\text{softmax}(S)\;(\text{row-wise}),\quad Y=WV")
    st.latex(r"\text{Causal mask: } S_{ij}\leftarrow \begin{cases}S_{ij} & j\le i\\ -\infty & j>i\end{cases}")

# ======================== Lessons 1–3 =========================

def lesson1():
    st.subheader("Lesson 1 — Linear Algebra Foundations")
    
    st.write("""
    - **Concepts**: vectors, dot products, norms, angles, matrix multiplication, transpose, cosine similarity.
    - **Why it matters**: Every neural net layer is basically `y = xW + b`. Attention uses `Q @ K^T`. Understanding these basics makes the rest feel natural.
    """)
    
    with st.expander("Vector demo: dot, norm, cosine"):
        u = [float(x) for x in st.text_input("u", "1,2,3").split(",")]
        v = [float(x) for x in st.text_input("v", "2,1,0").split(",")]
        st.write("dot(u,v) =", dot(u, v))
        st.write("norm(u) =", norm(u))
        st.write("cosine_similarity =", round(cosine_similarity(u, v), 4))
    with st.expander("Matrix demo: A @ B"):
        A = randn_matrix(3, 2, 0.3); B = randn_matrix(2, 3, 0.3)
        try:
            C = matmul(A, B)
            st.write("A:", A); st.write("B:", B); st.write("C=A@B:", C)
        except Exception as e:
            st.error(f"Matrix multiply failed: {e}")

    show_f = st.checkbox("Show formulas (LaTeX)", key="f_l1")
    if show_f: formulas_lesson1()

    st.markdown("---")
    st.markdown("### Lesson 1 Quiz")
    ask_mcq(
        "l1_1",
        "Which operation is the backbone of neural network layers?",
        ["Matrix multiplication", "Element-wise max", "Sorting"],
        correct_idx=0,
        help_text="Recall: y = xW + b."
    )
    ask_mcq(
        "l1_2",
        "Cosine similarity measures similarity based on…",
        ["Magnitude only", "Angle between vectors", "Random chance"],
        correct_idx=1
    )

def lesson2():
    st.subheader("Lesson 2 — Probability & Softmax")

    st.write("""
    - **Concepts**: turning scores (logits) into probabilities (softmax), comparing distributions (cross‑entropy).
    - **Why it matters**: Classifiers output logits; training minimizes cross‑entropy with the target distribution.
    """)
    
    logits = [float(x) for x in st.text_input("Logits", "2.0,1.0,0.1").split(",")]
    true_idx = st.number_input("True class", 0, len(logits)-1, 0)
    probs = softmax(logits); target = one_hot(true_idx, len(probs))
    st.write("softmax:", [round(p, 4) for p in probs])
    st.write("cross-entropy:", round(cross_entropy(target, probs), 6))

    show_f = st.checkbox("Show formulas (LaTeX)", key="f_l2")
    if show_f: formulas_lesson2()

    st.markdown("---")
    st.markdown("### Lesson 2 Quiz")
    ask_mcq(
        "l2_1",
        "Softmax converts logits into…",
        ["Binary labels", "Probabilities that sum to 1", "Raw scores"],
        correct_idx=1
    )
    ask_mcq(
        "l2_2",
        "Cross-entropy is low when…",
        ["The predicted distribution matches the target", "The logits are large", "We pick the wrong class confidently"],
        correct_idx=0
    )

def lesson3():
    st.subheader("Lesson 3 — Gradient Descent Basics")

    st.write("""
    - **Concepts**: gradient, numerical vs. analytical derivatives, gradient descent.
    - **Why it matters**: Training = minimizing a loss by following the gradient downhill.
    """)
    
    a = st.number_input("a (target)", -10.0, 10.0, 3.0)
    x0 = st.number_input("x0 (start)", -10.0, 10.0, 0.0)
    lr = st.slider("Learning rate", 0.01, 1.0, 0.2)
    steps = st.slider("Steps", 5, 100, 20)
    f  = lambda x: (x - a)**2
    df = lambda x: 2*(x - a)
    x = x0; history = []
    for t in range(steps):
        grad = df(x); x -= lr * grad; history.append((t, x, f(x)))
    st.line_chart([fx for (_, _, fx) in history])
    st.write("Final x≈", round(x, 4), "f(x)≈", round(f(x), 6))

    show_f = st.checkbox("Show formulas (LaTeX)", key="f_l3")
    if show_f: formulas_lesson3()

    st.markdown("---")
    st.markdown("### Lesson 3 Quiz")
    ask_mcq(
        "l3_1",
        "Gradient descent updates parameters by moving…",
        ["In the direction of the gradient", "Opposite the gradient", "Randomly"],
        correct_idx=1
    )
    ask_mcq(
        "l3_2",
        "Too large a learning rate typically causes…",
        ["Faster convergence always", "Divergence or oscillation", "No effect"],
        correct_idx=1
    )

# ======================== XOR Helpers (Lesson 4 Sandbox) ==================

def xor_dataset(): 
    return [[0,0],[0,1],[1,0],[1,1]], [[1,0],[0,1],[0,1],[1,0]]  # X (4x2), T one-hot (4x2)

def lin_forward(X, W, b): return add(matmul(X, transpose(W)), transpose(b))
def relu_forward(X): return apply_rowwise(X, relu)
def softmax_forward(X): return softmax_rows(X)
def init_layer(inp, out, std=0.5): return randn_matrix(out, inp, std), [[0.0] for _ in range(out)]

def nn_init(H=2, std=0.8):
    """Initialize a stable 2-layer net for XOR: D=2 -> H -> C=2."""
    W1 = randn_matrix(H, 2, std)         # (H x 2)
    b1 = [[0.0] for _ in range(H)]       # (H x 1)
    W2 = randn_matrix(2, H, std)         # (2 x H)
    b2 = [[0.0] for _ in range(2)]       # (2 x 1)
    return W1, b1, W2, b2

def nn_forward(X, W1, b1, W2, b2):
    """Forward pass only."""
    Z1 = lin_forward(X, W1, b1)          # (N x H)
    A1 = relu_forward(Z1)                 # (N x H)
    Z2 = lin_forward(A1, W2, b2)         # (N x C)
    Y  = softmax_forward(Z2)             # (N x C)
    return Z1, A1, Z2, Y

def nn_backward_step(X, T, params, lr=0.1):
    """
    Do exactly one SGD step and return (new_params, loss, Y).
    params = (W1,b1,W2,b2)
    """
    W1, b1, W2, b2 = params
    Z1, A1, Z2, Y = nn_forward(X, W1, b1, W2, b2)

    loss = cross_entropy_rows(T, Y)

    # Gradients
    dZ2 = [[(y - t) for y, t in zip(yrow, trow)] for yrow, trow in zip(Y, T)]  # (N x C)
    dW2 = matmul(transpose(A1), dZ2)                    # (H x C)
    db2 = [[sum(col)] for col in transpose(dZ2)]        # (C x 1)

    dA1 = matmul(dZ2, W2)                               # (N x H)
    dZ1 = [[dA1[i][j] * (1.0 if Z1[i][j] > 0 else 0.0) for j in range(len(Z1[0]))]
           for i in range(len(Z1))]                     # (N x H)
    dW1 = transpose(matmul(transpose(X), dZ1))          # (H x D)
    db1 = [[sum(col)] for col in transpose(dZ1)]        # (H x 1)

    # SGD update
    W2 = sub(W2, scalar_mul(dW2, lr))
    b2 = sub(b2, scalar_mul(db2, lr))
    W1 = sub(W1, scalar_mul(dW1, lr))
    b1 = sub(b1, scalar_mul(db1, lr))

    return (W1, b1, W2, b2), loss, Y

# ======================== Lesson 4 (Interactive Sandbox) ==================

def lesson4():
    st.subheader("Lesson 4 — Tiny Neural Net (XOR) — Interactive Sandbox (H=2)")

    st.write("""
    - **Concepts**: linear layers, activations, softmax + cross‑entropy, manual backprop.
    - **Why it matters**: Before Transformers, all deep nets are compositions of linear maps and nonlinearities.
    """)

    X, T = xor_dataset()  # X: (4x2), T: (4x2) one-hot
    lr = st.slider("Learning rate", 0.001, 1.0, 0.1, 0.001)
    steps = st.number_input("Steps (for 'Train N steps')", min_value=1, max_value=5000, value=50, step=10)

    # Keep parameters in session_state (stable across reruns)
    if "nn_params" not in st.session_state:
        st.session_state.nn_params = nn_init(H=2)

    colA, colB, colC = st.columns([1,1,2])
    with colA:
        if st.button("🔄 Re-initialize parameters"):
            st.session_state.nn_params = nn_init(H=2)
    with colB:
        if st.button("🧭 Train 1 step"):
            st.session_state.nn_params, loss, Y = nn_backward_step(X, T, st.session_state.nn_params, lr=lr)
            st.toast(f"Trained 1 step — loss: {loss:.4f}")
    with colC:
        if st.button("🚀 Train N steps"):
            loss = None
            Y = None
            for _ in range(int(steps)):
                st.session_state.nn_params, loss, Y = nn_backward_step(X, T, st.session_state.nn_params, lr=lr)
            st.success(f"Trained {steps} steps — loss: {loss:.4f}")

    # Always show current forward pass
    W1, b1, W2, b2 = st.session_state.nn_params
    _, _, _, Y_now = nn_forward(X, W1, b1, W2, b2)
    loss_now = cross_entropy_rows(T, Y_now)

    rows = [{
        "X": x,
        "P": [round(v, 3) for v in y],
        "pred": int(argmax(y)),
        "true": int(argmax(t))
    } for x, y, t in zip(X, Y_now, T)]
    st.write(f"Current loss: **{loss_now:.4f}**")
    st.dataframe(rows, use_container_width=True)

    with st.expander("Show current parameters (shapes)"):
        st.write(f"W1 (H×D = 2×2): {W1}")
        st.write(f"b1 (H×1 = 2×1): {b1}")
        st.write(f"W2 (C×H = 2×2): {W2}")
        st.write(f"b2 (C×1 = 2×1): {b2}")

    show_f = st.checkbox("Show formulas (LaTeX)", key="f_l4")
    if show_f: formulas_lesson4()

    st.markdown("---")
    st.markdown("### Lesson 4 Quiz")
    ask_mcq(
        "l4_1",
        "Backpropagation relies primarily on…",
        ["Chain rule of calculus", "Random search", "Sorting gradients"],
        correct_idx=0
    )
    ask_mcq(
        "l4_2",
        "Why do we need a nonlinearity like ReLU between linear layers?",
        ["We don't; linear+linear is universal", "To model non-linear decision boundaries", "To speed up matmul"],
        correct_idx=1
    )

# ======================== Lesson 5 (Attention) ===========================

def causal_mask(n): return [[1 if j <= i else 0 for j in range(n)] for i in range(n)]
def masked_fill(M, mask, v=-1e9): return [[M[i][j] if mask[i][j] else v for j in range(len(M[0]))] for i in range(len(M))]

def attention(Q, K, V, causal=False):
    d = len(Q[0])
    scores = matmul(Q, transpose(K))
    scores = [[s / math.sqrt(d) for s in row] for row in scores]
    if causal:
        scores = masked_fill(scores, causal_mask(len(scores)))
    W = softmax_rows(scores)
    return matmul(W, V), W, scores

def lesson5():
    st.subheader("Lesson 5 — Scaled Dot-Product Attention")

    st.write("""
    - **Concepts**: projections (Q, K, V), similarity via dot products, scaling by √d, softmax over scores, causal masks.
    - **Why it matters**: This operation is the heart of GPT‑style Transformers.
    """)
    
    # Guarded sliders (avoid 0)
    T = st.slider("Seq length (T)", 1, 32, 4, 1)
    d = st.slider("Model dim (d)", 1, 64, 6, 1)
    dv = st.slider("Value dim (dv)", 1, 64, 6, 1)
    causal = st.checkbox("Causal mask", True)

    # Build shapes
    X  = [[random.gauss(0, 1) for _ in range(d)] for _ in range(T)]  # (T x d)
    Wq = randn_matrix(d, d, 0.4)   # (d x d)
    Wk = randn_matrix(d, d, 0.4)   # (d x d)
    Wv = randn_matrix(d, dv, 0.4)  # (d x dv)

    try:
        Q  = matmul(X, Wq)    # (T x d)
        K  = matmul(X, Wk)    # (T x d)
        V  = matmul(X, Wv)    # (T x dv)
        Y, W, S = attention(Q, K, V, causal)
        st.write("Scores (QKᵀ/√d):"); st.dataframe(S, use_container_width=True)
        st.write("Weights (softmax rows):"); st.dataframe(W, use_container_width=True)
        st.write("Output Y = W V:"); st.dataframe(Y, use_container_width=True)
    except Exception as e:
        st.error(f"Attention computation failed: {e}")
        st.code(
            f"Debug shapes -> X:{_shape(X)}, Wq:{_shape(Wq)}, Wk:{_shape(Wk)}, Wv:{_shape(Wv)}",
            language="text"
        )

    show_f = st.checkbox("Show formulas (LaTeX)", key="f_l5")
    if show_f: formulas_lesson5()

    st.markdown("---")
    st.markdown("### Lesson 5 Quiz")
    ask_mcq(
        "l5_1",
        "In attention, scaling by √d is used to…",
        ["Make outputs larger", "Keep dot-products in a stable range", "Reduce matrix size"],
        correct_idx=1
    )
    ask_mcq(
        "l5_2",
        "Causal masking ensures that…",
        ["Tokens can see the future", "Each token only attends to past and current tokens", "We reduce memory usage"],
        correct_idx=1
    )

# ========================== Sidebar Theory ============================

st.sidebar.title("📘 Theory Notes")
lesson_pick = st.sidebar.radio(
    "Pick a lesson",
    ["Intro", "Intro Quiz", "1. Linear Algebra","2. Probability","3. Gradient Descent","4. Neural Nets","5. Attention", "Metrics"]
)
if lesson_pick == "Intro":
    st.sidebar.markdown("**Attention = Focus.** Assign bigger weights to the most relevant parts of the input.")
elif lesson_pick == "Intro Quiz":
    st.sidebar.markdown("Test key ideas from the introduction.")
elif lesson_pick.startswith("1"): st.sidebar.markdown("**Linear Algebra**: vectors, dot, norms, cosine, matrix multiply—backbone of embeddings & attention.")
elif lesson_pick.startswith("2"): st.sidebar.markdown("**Probability**: softmax makes a distribution; cross-entropy trains classifiers.")
elif lesson_pick.startswith("3"): st.sidebar.markdown("**Optimization**: follow the gradient downhill; tuning LR matters.")
elif lesson_pick.startswith("4"): st.sidebar.markdown("**Neural Nets**: stacked linear + nonlinear layers; backprop is chain rule.")
elif lesson_pick.startswith("5"): st.sidebar.markdown("**Attention**: softmax(QKᵀ/√d)V with optional causal masking.")
elif lesson_pick == "Metrics":
    st.sidebar.markdown("View scores & download a CSV of your current session (anonymized id).")

st.write("**Goal**: Understand the math under the hood of modern AI (especially Transformers) by coding everything from first principles—no NumPy, no PyTorch.")
st.write("**Why no heavy libraries?** To remove the “black box” and make the math tangible. When you write `matmul`, `softmax`, or `attention` yourself, you feel why these operations matter.")

# ============================== Metrics Tab ===========================

def tab_metrics():
    st.subheader("📊 Quiz Metrics (This Session)")
    sid = st.session_state.session_id
    st.write(f"Anonymized Session ID: **{sid}**")

    rows = []
    total = 0; correct = 0
    for qid, sel_idx in st.session_state.quiz_answers.items():
        ok = bool(st.session_state.quiz_scores.get(qid, False))
        rows.append({"quiz_id": qid, "selected_idx": sel_idx, "correct": ok})
        total += 1; correct += int(ok)

    rows = sorted(rows, key=lambda r: r["quiz_id"])
    st.write(f"Score: **{correct} / {total}**" if total else "No answers yet.")
    st.dataframe(rows, use_container_width=True)

    # CSV download
    if rows:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=["session_id", "quiz_id", "selected_idx", "correct"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"session_id": sid, **r})
        st.download_button("Download CSV", data=csv_buf.getvalue(), file_name=f"quiz_results_{sid}.csv", mime="text/csv")

    # Reset controls
    st.markdown("---")
    if st.button("🔄 Reset all quiz answers"):
        st.session_state.quiz_answers = {}
        st.session_state.quiz_scores = {}
        st.success("Cleared! Go back to the tabs and try again.")

# ============================== Main UI ===============================

st.title("Attention From Scratch — Introduction + 5 Lessons")
tabs = st.tabs(["Introduction","Intro Quiz","Lesson 1","Lesson 2","Lesson 3","Lesson 4","Lesson 5","Metrics"])
with tabs[0]: tab_introduction()
with tabs[1]: tab_intro_quiz()
with tabs[2]: lesson1()
with tabs[3]: lesson2()
with tabs[4]: lesson3()
with tabs[5]: lesson4()
with tabs[6]: lesson5()
with tabs[7]: tab_metrics()
