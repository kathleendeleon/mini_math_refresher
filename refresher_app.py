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

# =================== Session state for quizzes & metrics ==================

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}   # key: quiz_id, value: selected option index
if "quiz_scores" not in st.session_state:
    st.session_state.quiz_scores = {}    # key: quiz_id, value: bool correct
if "session_id" not in st.session_state:
    st.session_state.session_id = f"S{int(time.time())}-{random.randint(1000,9999)}"

def ask_mcq(quiz_id, question, options, correct_idx, help_text=None):
    """
    Render a multiple-choice question and store the answer in session_state.
    Returns (selected_idx, is_correct).
    """
    if help_text:
        st.caption(help_text)
    # Default selection (0) unless previously answered
    default_index = st.session_state.quiz_answers.get(quiz_id, 0)
    selected = st.radio(question, options, key=f"q_{quiz_id}", index=default_index)
    sel_idx = options.index(selected)
    st.session_state.quiz_answers[quiz_id] = sel_idx
    is_correct = (sel_idx == correct_idx)
    st.session_state.quiz_scores[quiz_id] = is_correct
    if is_correct:
        st.success("✅ Correct!")
    else:
        st.info(f"ℹ️ Selected: {options[sel_idx]}")
    return sel_idx, is_correct

# ======================== Introduction Tab =========================

def tab_introduction():
    st.subheader("What is Attention? (Executive & Technical View)")

    col1, col2 = st.columns(2)
    with col1:
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
        st.write("- Search & Q/A, chat assistants, document summarization")
        st.write("- Code completion, SQL generation")
        st.write("- Customer support triage, CRM note summarization")
        st.write("- Fraud/anomaly detection")
        st.write("- Product recommendations")
        st.write("- Vision & speech: captioning, recognition")

        with st.expander("Limitations / Risks"):
            st.write("- **Quadratic compute** in standard attention with context length.")
            st.write("- **Hallucinations** if prompts/data are insufficient.")
            st.write("- **Bias** mirrors training data; needs evaluation.")

    with col2:
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

    st.markdown("---")
    st.markdown("### Quick Intro Quiz")
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

# ======================== Lessons 1–5 =========================

def lesson1():
    st.subheader("Lesson 1 — Linear Algebra Foundations")
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
    logits = [float(x) for x in st.text_input("Logits", "2.0,1.0,0.1").split(",")]
    true_idx = st.number_input("True class", 0, len(logits)-1, 0)
    probs = softmax(logits); target = one_hot(true_idx, len(probs))
    st.write("softmax:", [round(p, 4) for p in probs])
    st.write("cross-entropy:", round(cross_entropy(target, probs), 6))

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

def xor_dataset(): return [[0,0],[0,1],[1,0],[1,1]], [[1,0],[0,1],[0,1],[1,0]]
def lin_forward(X, W, b): return add(matmul(X, transpose(W)), transpose(b))
def relu_forward(X): return apply_rowwise(X, relu)
def softmax_forward(X): return softmax_rows(X)
def init_layer(inp, out, std=0.5): return randn_matrix(out, inp, std), [[0.0] for _ in range(out)]

def train_xor(epochs=2000, lr=0.1, hidden=4, debug=False):
    # Ensure hidden >= 1 to avoid zero-width layers
    hidden = max(1, int(hidden))
    X, T = xor_dataset()  # X:(4x2), T:(4x2)
    D_in, D_h, D_out = 2, hidden, 2

    W1, b1 = init_layer(D_in, D_h, std=0.8)  # W1:(h x 2), b1:(h x 1)
    W2, b2 = init_layer(D_h, D_out, std=0.8) # W2:(2 x h), b2:(2 x 1)

    for _ in range(epochs):
        Z1 = lin_forward(X, W1, b1)        # (4 x h)
        A1 = relu_forward(Z1)
        Z2 = lin_forward(A1, W2, b2)       # (4 x 2)
        Y  = softmax_forward(Z2)           # (4 x 2)

        # dL/dZ2 = Y - T
        dZ2 = [[(y - t) for y, t in zip(yrow, trow)] for yrow, trow in zip(Y, T)]  # (4 x 2)

        # dW2 = A1^T @ dZ2
        dW2 = matmul(transpose(A1), dZ2)                    # (h x 2)
        db2 = [[sum(col)] for col in transpose(dZ2)]        # (2 x 1)

        # dA1 = dZ2 @ W2
        dA1 = matmul(dZ2, W2)                               # (4 x h)
        # dZ1 = dA1 * ReLU'(Z1)
        dZ1 = []
        for i in range(len(Z1)):
            dZ1.append([dA1[i][j] * (1.0 if Z1[i][j] > 0 else 0.0)
                        for j in range(len(Z1[0]))])        # (4 x h)

        # dW1 = X^T @ dZ1 -> (2 x h); transpose to (h x 2)
        dW1 = transpose(matmul(transpose(X), dZ1))          # (h x 2)
        db1 = [[sum(col)] for col in transpose(dZ1)]        # (h x 1)

        # SGD step
        W2 = sub(W2, scalar_mul(dW2, lr))
        b2 = sub(b2, scalar_mul(db2, lr))
        W1 = sub(W1, scalar_mul(dW1, lr))
        b1 = sub(b1, scalar_mul(db1, lr))

    if debug:
        st.caption(
            f"Shapes — X:{_shape(X)}, T:{_shape(T)}, W1:{_shape(W1)}, b1:{_shape(b1)}, W2:{_shape(W2)}, b2:{_shape(b2)}"
        )

    # Final forward for reporting
    Z1 = lin_forward(X, W1, b1); A1 = relu_forward(Z1)
    Z2 = lin_forward(A1, W2, b2); Y = softmax_forward(Z2)
    return Y

def lesson4():
    st.subheader("Lesson 4 — Tiny Neural Net (XOR)")
    epochs = st.slider("Epochs", 100, 5000, 2000, 100)
    lr     = st.number_input("Learning rate", 0.0001, 1.0, 0.1, step=0.05, format="%.4f")
    hidden = st.slider("Hidden units", 1, 16, 4, 1)  # min 1 to avoid zero-width layers
    debug  = st.checkbox("Show debug shapes", False)

    try:
        Y = train_xor(epochs=epochs, lr=lr, hidden=hidden, debug=debug)
        X, T = xor_dataset()
        rows = [{
            "X": x,
            "P": [round(v, 3) for v in y],
            "pred": int(argmax(y)),
            "true": int(argmax(t))
        } for x, y, t in zip(X, Y, T)]
        st.dataframe(rows, use_container_width=True)
    except Exception as e:
        st.error(f"Lesson 4 failed: {e}")
        st.stop()

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
    T = st.slider("Seq length", 1, 8, 4); d = st.slider("Model dim", 1, 16, 6); dv = st.slider("Value dim", 1, 16, 6)
    causal = st.checkbox("Causal mask", True)
    X  = [[random.gauss(0, 1) for _ in range(d)] for _ in range(T)]
    Wq = randn_matrix(d, d, 0.4); Wk = randn_matrix(d, d, 0.4); Wv = randn_matrix(d, dv, 0.4)
    Q  = matmul(X, Wq); K = matmul(X, Wk); V = matmul(X, Wv)
    try:
        Y, W, S = attention(Q, K, V, causal)
        st.write("Scores (QKᵀ/√d):"); st.dataframe(S, use_container_width=True)
        st.write("Weights (softmax rows):"); st.dataframe(W, use_container_width=True)
        st.write("Output Y = W V:"); st.dataframe(Y, use_container_width=True)
    except Exception as e:
        st.error(f"Attention computation failed: {e}")

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
    ["Intro", "1. Linear Algebra","2. Probability","3. Gradient Descent","4. Neural Nets","5. Attention", "Metrics"]
)
if lesson_pick == "Intro":
    st.sidebar.markdown("**Attention = Focus.** Assign bigger weights to the most relevant parts of the input.")
elif lesson_pick.startswith("1"): st.sidebar.markdown("**Linear Algebra**: vectors, dot, norms, cosine, matrix multiply—backbone of embeddings & attention.")
elif lesson_pick.startswith("2"): st.sidebar.markdown("**Probability**: softmax makes a distribution; cross-entropy trains classifiers.")
elif lesson_pick.startswith("3"): st.sidebar.markdown("**Optimization**: follow the gradient downhill; tuning LR matters.")
elif lesson_pick.startswith("4"): st.sidebar.markdown("**Neural Nets**: stacked linear + nonlinear layers; backprop is chain rule.")
elif lesson_pick.startswith("5"): st.sidebar.markdown("**Attention**: softmax(QKᵀ/√d)V with optional causal masking.")
elif lesson_pick == "Metrics":
    st.sidebar.markdown("View scores & download a CSV of your current session (anonymized id).")

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
tabs = st.tabs(["Introduction","Lesson 1","Lesson 2","Lesson 3","Lesson 4","Lesson 5","Metrics"])
with tabs[0]: tab_introduction()
with tabs[1]: lesson1()
with tabs[2]: lesson2()
with tabs[3]: lesson3()
with tabs[4]: lesson4()
with tabs[5]: lesson5()
with tabs[6]: tab_metrics()
