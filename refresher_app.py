import math, random
import streamlit as st

random.seed(42)
st.set_page_config(page_title="Attention From Scratch — 5 Lessons", page_icon="🧠", layout="wide")

# ======================== Shared Utilities ========================
def banner(title): st.markdown(f"### {title}")

def matmul(A, B):
    m, n = len(A), len(A[0]); p = len(B[0])
    out = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            for j in range(p): out[i][j] += aik * B[k][j]
    return out

def transpose(A): return [list(row) for row in zip(*A)]
def add(A, B): return [[a+b for a,b in zip(ra, rb)] for ra, rb in zip(A,B)]
def sub(A, B): return [[a-b for a,b in zip(ra, rb)] for ra, rb in zip(A,B)]
def scalar_mul(A, s): return [[a*s for a in row] for row in A]
def dot(u, v): return sum(ui*vi for ui,vi in zip(u,v))
def norm(v): return math.sqrt(max(1e-12, dot(v,v)))
def cosine_similarity(u, v): return dot(u, v) / (norm(u) * norm(v))
def argmax(xs): return max(range(len(xs)), key=lambda i: xs[i])
def one_hot(idx, n): v=[0.0]*n; v[idx]=1.0; return v
def softmax(xs): m=max(xs); exps=[math.exp(x-m) for x in xs]; Z=sum(exps); return [e/Z for e in exps]
def softmax_rows(M): return [softmax(row) for row in M]
def cross_entropy(p, q): eps=1e-12; return -sum(pi*math.log(max(qi, eps)) for pi, qi in zip(p,q))
def cross_entropy_rows(P, Q): return sum(cross_entropy(p, q) for p, q in zip(P,Q)) / len(P)
def relu(x): return x if x>0 else 0.0
def apply_rowwise(A, fn): return [[fn(x) for x in row] for row in A]
def randn_matrix(m, n, std=0.02): return [[random.gauss(0.0, std) for _ in range(n)] for _ in range(m)]

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
        st.write("- Fraud/anomaly detection (focus on suspicious patterns)")
        st.write("- Product recommendations (focus on relevant behaviors)")
        st.write("- Vision & speech: image captioning, speech recognition")

        with st.expander("Limitations / Risks"):
            st.write("- **Compute cost** grows with context length (quadratic time/memory in standard attention).")
            st.write("- **Hallucinations** if training data lacks facts or prompts are unclear.")
            st.write("- **Bias** can appear if data has bias; needs monitoring & evaluation.")

    with col2:
        st.markdown("#### Technical Summary")
        st.code("Attention(Q, K, V) = softmax(Q Kᵀ / √d) V", language="text")
        st.write("""
        - Inputs: a sequence of token embeddings **X** ∈ ℝ^{T×d}  
          Project to **Q = X W_q**, **K = X W_k**, **V = X W_v**  
        - **Scores**: similarity matrix `S = Q Kᵀ / √d` (T×T)  
        - **Weights**: `W = softmax(S)` row-wise (each row sums to 1)  
        - **Output**: `Y = W V` (weighted mixture of values)
        - **Causal mask** (for generation): prevent attending to future tokens.
        """)
        with st.expander("Glossary"):
            st.write("- **Q (Query)**: what the current token is looking for.")
            st.write("- **K (Key)**: what each token offers as a tag/index.")
            st.write("- **V (Value)**: the content carried by each token.")
            st.write("- **Scaling (√d)**: keeps dot-products in a numerically stable range.")
        st.markdown("#### Why it Scales")
        st.write("Parallelizable across tokens; replaces recurrence with matrix multiplies; hardware-friendly.")

# ======================== Lesson Content =========================
def lesson1():
    st.subheader("Lesson 1 — Linear Algebra Foundations")
    with st.expander("Vector demo: dot, norm, cosine"):
        u = [float(x) for x in st.text_input("u", "1,2,3").split(",")]
        v = [float(x) for x in st.text_input("v", "2,1,0").split(",")]
        st.write("dot(u,v) =", dot(u,v))
        st.write("norm(u) =", norm(u))
        st.write("cosine_similarity =", round(cosine_similarity(u,v),4))
    with st.expander("Matrix demo: A @ B"):
        A = randn_matrix(3,2,0.3); B = randn_matrix(2,3,0.3)
        st.write("A:", A); st.write("B:", B); st.write("C=A@B:", matmul(A,B))

def lesson2():
    st.subheader("Lesson 2 — Probability & Softmax")
    logits = [float(x) for x in st.text_input("Logits", "2.0,1.0,0.1").split(",")]
    true_idx = st.number_input("True class", 0, len(logits)-1, 0)
    probs = softmax(logits); target = one_hot(true_idx, len(probs))
    st.write("softmax:", [round(p,4) for p in probs])
    st.write("cross-entropy:", round(cross_entropy(target, probs), 6))

def lesson3():
    st.subheader("Lesson 3 — Gradient Descent Basics")
    a = st.number_input("a (target)", -10.0, 10.0, 3.0)
    x0 = st.number_input("x0 (start)", -10.0, 10.0, 0.0)
    lr = st.slider("Learning rate", 0.01, 1.0, 0.2)
    steps = st.slider("Steps", 5, 100, 20)
    f  = lambda x: (x-a)**2; df=lambda x:2*(x-a)
    x=x0; history=[]
    for t in range(steps): grad=df(x); x-=lr*grad; history.append((t,x,f(x)))
    st.line_chart([fx for (_,_,fx) in history])
    st.write("Final x≈", round(x,4),"f(x)≈", round(f(x),6))

def xor_dataset(): return [[0,0],[0,1],[1,0],[1,1]], [[1,0],[0,1],[0,1],[1,0]]
def lin_forward(X,W,b): return add(matmul(X, transpose(W)), transpose(b))
def relu_forward(X): return apply_rowwise(X, relu)
def softmax_forward(X): return softmax_rows(X)
def init_layer(inp,out,std=0.5): return randn_matrix(out,inp,std), [[0.0] for _ in range(out)]

def train_xor(epochs=2000, lr=0.1, hidden=4):
    X,T=xor_dataset(); W1,b1=init_layer(2,hidden,0.8); W2,b2=init_layer(hidden,2,0.8)
    for _ in range(epochs):
        Z1=lin_forward(X,W1,b1); A1=relu_forward(Z1); Z2=lin_forward(A1,W2,b2); Y=softmax_forward(Z2)
        dZ2=[[y-t for y,t in zip(yrow,trow)] for yrow,trow in zip(Y,T)]
        dW2=matmul(transpose(A1),dZ2); db2=[[sum(col)] for col in transpose(dZ2)]
        dA1=matmul(dZ2,W2); dZ1=[[dA1[i][j]*(1 if Z1[i][j]>0 else 0) for j in range(len(Z1[0]))] for i in range(len(Z1))]
        dW1=transpose(matmul(transpose(X),dZ1)); db1=[[sum(col)] for col in transpose(dZ1)]
        W2=sub(W2,scalar_mul(dW2,lr)); b2=sub(b2,scalar_mul(db2,lr))
        W1=sub(W1,scalar_mul(dW1,lr)); b1=sub(b1,scalar_mul(db1,lr))
    return Y

def lesson4():
    st.subheader("Lesson 4 — Tiny Neural Net (XOR)")
    epochs=st.slider("Epochs",100,5000,2000); lr=st.slider("LR",0.01,1.0,0.1); hidden=st.slider("Hidden units",2,16,4)
    Y=train_xor(epochs,lr,hidden); X,T=xor_dataset()
    rows=[{"X":x,"pred":int(argmax(y)),"true":int(argmax(t)),"P":[round(v,3) for v in y]} for x,y,t in zip(X,Y,T)]
    st.dataframe(rows)

def causal_mask(n): return [[1 if j<=i else 0 for j in range(n)] for i in range(n)]
def masked_fill(M,mask,v=-1e9): return [[M[i][j] if mask[i][j] else v for j in range(len(M[0]))] for i in range(len(M))]
def attention(Q,K,V,causal=False):
    d=len(Q[0]); scores=matmul(Q,transpose(K)); scores=[[s/math.sqrt(d) for s in row] for row in scores]
    if causal: scores=masked_fill(scores,causal_mask(len(scores)))
    W=softmax_rows(scores); return matmul(W,V),W,scores

def lesson5():
    st.subheader("Lesson 5 — Scaled Dot-Product Attention")
    T=st.slider("Seq length",1,8,4); d=st.slider("Model dim",1,16,6); dv=st.slider("Value dim",1,16,6)
    causal=st.checkbox("Causal mask",True)
    X=[[random.gauss(0,1) for _ in range(d)] for _ in range(T)]
    Wq=randn_matrix(d,d,0.4); Wk=randn_matrix(d,d,0.4); Wv=randn_matrix(d,dv,0.4)
    Q=matmul(X,Wq); K=matmul(X,Wk); V=matmul(X,Wv)
    Y,W,S=attention(Q,K,V,causal)
    st.write("Scores (QKᵀ/√d):"); st.dataframe(S, use_container_width=True)
    st.write("Weights (softmax rows):"); st.dataframe(W, use_container_width=True)
    st.write("Output Y = W V:"); st.dataframe(Y, use_container_width=True)

# ========================== Sidebar Theory ============================
st.sidebar.title("📘 Theory Notes")
lesson_pick = st.sidebar.radio(
    "Pick a lesson",
    ["Intro", "1. Linear Algebra","2. Probability","3. Gradient Descent","4. Neural Nets","5. Attention"]
)
if lesson_pick == "Intro":
    st.sidebar.markdown("**Attention = Focus.** Assign bigger weights to the most relevant parts of the input.")
elif lesson_pick.startswith("1"): st.sidebar.markdown("**Linear Algebra**: vectors, dot, norms, cosine, matrix multiply—backbone of embeddings & attention.")
elif lesson_pick.startswith("2"): st.sidebar.markdown("**Probability**: softmax makes a distribution; cross-entropy trains classifiers.")
elif lesson_pick.startswith("3"): st.sidebar.markdown("**Optimization**: follow the gradient downhill; tuning LR matters.")
elif lesson_pick.startswith("4"): st.sidebar.markdown("**Neural Nets**: stacked linear + nonlinear layers; backprop is chain rule.")
elif lesson_pick.startswith("5"): st.sidebar.markdown("**Attention**: softmax(QKᵀ/√d)V with optional causal masking.")

# ============================== Main UI ===============================
st.title("🧠 Attention From Scratch — 5 Lessons + Introduction")
tabs = st.tabs(["Introduction","Lesson 1","Lesson 2","Lesson 3","Lesson 4","Lesson 5"])
with tabs[0]: tab_introduction()
with tabs[1]: lesson1()
with tabs[2]: lesson2()
with tabs[3]: lesson3()
with tabs[4]: lesson4()
with tabs[5]: lesson5()
