# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\mod-nn_feedforward.py

# streamlit_app_nn_batch_modes_profheesen_unique_keys.py
# Vollständige, didaktische Streamlit-App für Feedforward + Backprop
# - Eindeutige Keys für alle Widgets (kein DuplicateElementKey)
# - blobs fix auf 4 Klassen
# - Batching: Full-Batch / Mini-Batch / SGD (online)
# - Sample-Slider 30..1000 (Default 50)
# - Schritttraining (Einzelschritt), Epoche, Auto-Train
# - Loss: BCE / CrossEntropy / RMSE
# - Aktivierungen: ReLU, LeakyReLU, Sigmoid, Tanh (Hidden); Sigmoid/Softmax/Linear (Output)
# - Initialisierungen: Xavier / He / RandomShift
# - Optimizer: SGD / Momentum / Adam; LR-Step-Scheduler
# - Heatmaps, Gradienten-Histogramme, Rechengraph, Per-Sample-Tracing

import math
import json
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Optional: scikit-learn für Datensätze & Metriken
try:
    from sklearn.datasets import make_moons, make_circles, make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# =================== Aktivierungen ===================
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def dsigmoid(a): return a * (1.0 - a)          # a = sigmoid(z)

def tanh(z): return np.tanh(z)
def dtanh(a): return 1.0 - a**2                 # a = tanh(z)

def relu(z): return np.maximum(0.0, z)
def drelu(z): return (z > 0.0).astype(float)    # Ableitung in z

def leaky_relu(z, alpha=0.01): return np.where(z > 0.0, z, alpha * z)
def dleaky_relu(z, alpha=0.01):
    g = np.ones_like(z)
    g[z < 0.0] = alpha
    return g

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    ex = np.exp(z_shift)
    return ex / np.sum(ex, axis=1, keepdims=True)

# =================== Verluste ===================
def rmse_loss(y_true, y_pred):
    err = y_pred - y_true
    return np.sqrt(np.mean(err**2))

def bce_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_loss(y_true_onehot, y_pred_proba, eps=1e-12):
    y_pred_proba = np.clip(y_pred_proba, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_proba), axis=1))

# =================== Initialisierung ===================
def init_weights(in_dim, hid_dim, out_dim, init_type="Xavier", seed=42):
    rng = np.random.default_rng(seed)
    if init_type == "Xavier":
        lim1 = math.sqrt(6.0 / (in_dim + hid_dim))
        W1 = rng.uniform(-lim1, lim1, (in_dim, hid_dim)); b1 = np.zeros((1, hid_dim))
        lim2 = math.sqrt(6.0 / (hid_dim + out_dim))
        W2 = rng.uniform(-lim2, lim2, (hid_dim, out_dim)); b2 = np.zeros((1, out_dim))
    elif init_type == "He":
        W1 = rng.normal(0.0, math.sqrt(2.0 / in_dim), (in_dim, hid_dim)); b1 = np.zeros((1, hid_dim))
        W2 = rng.normal(0.0, math.sqrt(2.0 / hid_dim), (hid_dim, out_dim)); b2 = np.zeros((1, out_dim))
    else:
        W1 = rng.normal(0.0, 1.0, (in_dim, hid_dim)) - 0.5; b1 = np.zeros((1, hid_dim))
        W2 = rng.normal(0.0, 1.0, (hid_dim, out_dim)) - 0.5; b2 = np.zeros((1, out_dim))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# =================== Optimizer ===================
@dataclass
class OptimState:
    m: Dict[str, np.ndarray] = field(default_factory=dict)
    v: Dict[str, np.ndarray] = field(default_factory=dict)
    t: int = 0

def sgd_update(params, grads, lr):
    for k in params: params[k] -= lr * grads["d" + k]
    return params

def momentum_update(params, grads, state: OptimState, lr, beta=0.9):
    for k in params:
        if k not in state.m: state.m[k] = np.zeros_like(params[k])
        state.m[k] = beta * state.m[k] + (1 - beta) * grads["d" + k]
        params[k] -= lr * state.m[k]
    return params, state

def adam_update(params, grads, state: OptimState, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    state.t += 1
    for k in params:
        if k not in state.m:
            state.m[k] = np.zeros_like(params[k]); state.v[k] = np.zeros_like(params[k])
        state.m[k] = beta1 * state.m[k] + (1 - beta1) * grads["d" + k]
        state.v[k] = beta2 * state.v[k] + (1 - beta2) * (grads["d" + k] ** 2)
        m_hat = state.m[k] / (1 - beta1 ** state.t)
        v_hat = state.v[k] / (1 - beta2 ** state.t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, state

# =================== Forward & Backward ===================
def forward_pass(X, params, act_hidden="ReLU", act_out="Sigmoid"):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    if act_hidden == "Sigmoid":
        a1 = sigmoid(z1)
    elif act_hidden == "Tanh":
        a1 = tanh(z1)
    elif act_hidden == "LeakyReLU":
        a1 = leaky_relu(z1)
    else:
        a1 = relu(z1)

    z2 = a1 @ W2 + b2
    if act_out == "Softmax":
        a2 = softmax(z2)
    elif act_out == "Linear":
        a2 = z2
    else:
        a2 = sigmoid(z2)
    return a2, {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

def backward_pass(y_true, params, cache, act_hidden="ReLU", act_out="Sigmoid", loss="BCE"):
    X, z1, a1, z2, a2 = cache["X"], cache["z1"], cache["a1"], cache["z2"], cache["a2"]
    W2 = params["W2"]; n = X.shape[0]

    # Output-Delta
    if loss == "CrossEntropy" and act_out == "Softmax":
        delta2 = (a2 - y_true) / n
    elif loss == "BCE" and act_out == "Sigmoid":
        delta2 = (a2 - y_true) / n
    elif loss == "RMSE":
        if act_out == "Sigmoid":
            delta2 = (2.0 / n) * (a2 - y_true) * dsigmoid(a2)
        else:
            delta2 = (2.0 / n) * (a2 - y_true)
    else:
        delta2 = (a2 - y_true) / n

    dW2 = a1.T @ delta2
    db2 = np.sum(delta2, axis=0, keepdims=True)

    if act_hidden == "Sigmoid":
        delta1 = (delta2 @ W2.T) * dsigmoid(a1)
    elif act_hidden == "Tanh":
        delta1 = (delta2 @ W2.T) * dtanh(a1)
    elif act_hidden == "LeakyReLU":
        delta1 = (delta2 @ W2.T) * dleaky_relu(z1)
    else:
        delta1 = (delta2 @ W2.T) * drelu(z1)

    dW1 = X.T @ delta1
    db1 = np.sum(delta1, axis=0, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "delta1": delta1, "delta2": delta2}

# =================== Scheduler ===================
def step_lr(lr0, step_every, gamma, step_count):
    if step_every <= 0: return lr0
    steps = step_count // step_every
    return lr0 * (gamma ** steps)

# =================== Daten ===================
def gen_data(kind="moons", n=400, noise=0.15, seed=0, stdize=True, k_classes=4):
    rng = np.random.default_rng(seed)

    if kind == "eigene":
        # --- XOR-Beispiel ---
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        y = np.array([[0],
                      [1],
                      [1],
                      [0]])
        k = 2
        y_vec = y  # schon in der richtigen Form
        return X, y_vec, y.ravel(), k

    if SKLEARN_AVAILABLE:
        if kind == "moons":
            X, y = make_moons(n_samples=n, noise=noise, random_state=seed); k = 2
        elif kind == "circles":
            X, y = make_circles(n_samples=n, factor=0.5, noise=noise, random_state=seed); k = 2
        elif kind == "blobs":
            X, y = make_blobs(n_samples=n, centers=4, random_state=seed, cluster_std=1.5); k = 4
        elif kind == "linear":
            X = rng.normal(0, 1, (n, 2))
            w = np.array([1.5, -2.0])
            y = (X @ w + 0.1 * rng.normal(0, 1, n) > 0).astype(int); k = 2
        else:
            X = rng.normal(0, 1, (n, 2)); y = (X[:, 0] + X[:, 1] > 0).astype(int); k = 2

        if stdize: X = StandardScaler().fit_transform(X)

    else:
        X = rng.normal(0, 1, (n, 2)); y = (X[:, 0] + 0.5 * X[:, 1] + noise * rng.normal(0, 1, n) > 0).astype(int); k = 2

    y_vec = y.reshape(-1, 1) if k == 2 else np.eye(k)[y]
    return X, y_vec, y, k

# =================== Plots ===================
def plot_loss(history):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history, label="Loss")
    ax.set_xlabel("Schritt"); ax.set_ylabel("Loss"); ax.legend()
    st.pyplot(fig)

def heatmap_matrix(M, title="Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(M, aspect="auto"); ax.set_title(title)
    ax.set_xlabel("Spalte"); ax.set_ylabel("Zeile")
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)

def hist_values(vals, title="Histogramm"):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(vals.ravel(), bins=30); ax.set_title(title)
    st.pyplot(fig)

def plot_network_gradients(W1, W2, G1=None, G2=None, title="Rechengraph"):
    in_dim, hid_dim = W1.shape; out_dim = W2.shape[1]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.set_axis_off()
    xs_in = np.full(in_dim, 0.0); ys_in = np.linspace(0, 1, in_dim)
    xs_h  = np.full(hid_dim, 0.5); ys_h  = np.linspace(0, 1, hid_dim)
    xs_o  = np.full(out_dim, 1.0); ys_o  = np.linspace(0, 1, out_dim)
    if G1 is None: G1 = np.zeros_like(W1)
    if G2 is None: G2 = np.zeros_like(W2)
    maxg1 = max(np.max(np.abs(G1)), 1e-12); maxg2 = max(np.max(np.abs(G2)), 1e-12)
    for i in range(in_dim):
        for j in range(hid_dim):
            lw = 1.0 + 4.0 * (abs(G1[i, j]) / maxg1)
            ax.plot([xs_in[i], xs_h[j]], [ys_in[i], ys_h[j]], linewidth=lw, alpha=0.6)
    for j in range(hid_dim):
        for k in range(out_dim):
            lw = 1.0 + 4.0 * (abs(G2[j, k]) / maxg2)
            ax.plot([xs_h[j], xs_o[k]], [ys_h[j], ys_o[k]], linewidth=lw, alpha=0.6)
    ax.scatter(xs_in, ys_in, s=100); ax.scatter(xs_h, ys_h, s=120); ax.scatter(xs_o, ys_o, s=140)
    ax.set_title(title); st.pyplot(fig)

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="Feedforward Neuronale Netze", layout="wide")
st.title("Feedforward Neuronale Netze (NN)")
st.markdown(
    "Diese App illustriert die wesentlichen Prozessschritte eines einfachen Feedforward-Netzes: "
    "**Daten → Feedforward → Loss → Backprop → Parameter-Update**. Sie wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------
with st.sidebar:
    st.header("Daten & Problem")
    ds_kind   = st.selectbox("Datensatz", ["moons", "circles", "blobs", "linear", "eigene"], index=0, key="sidebar_ds_kind")
    n_samples = st.slider("Anzahl Samples", 30, 1000, 50, 10, key="sidebar_n_samples")
    noise     = st.slider("Rauschen", 0.0, 0.5, 0.2, 0.01, key="sidebar_noise")
    stdize    = st.checkbox("Standardisieren (z-Score)", True, key="sidebar_stdize")
    seed_ds   = st.number_input("Seed Daten", 0, 10_000, 42, 1, key="sidebar_seed_ds")

    st.header("Modell")
    hid_dim    = st.slider("Neuronen im Hidden Layer", 1, 100, 3, 1, key="sidebar_hid_dim")
    act_hidden = st.selectbox("Aktivierung: Hidden", ["ReLU", "LeakyReLU", "Sigmoid", "Tanh"], index=0, key="sidebar_act_hidden")
    st.caption("Bei 2 Klassen: Sigmoid-Output empfohlen; bei >2 Klassen: Softmax.")
    act_out    = st.selectbox("Aktivierung: Output", ["Sigmoid", "Softmax", "Linear"], index=0, key="sidebar_act_out")

    st.header("Training")
    loss_type  = st.selectbox("Loss", ["BCE", "CrossEntropy", "RMSE"], index=0, key="sidebar_loss_type")
    batch_mode = st.selectbox("Batching", ["Full-Batch", "Mini-Batch", "SGD (online)"], index=0, key="sidebar_batch_mode")
    batch_size = st.slider("Mini-Batch-Größe", 8, 256, 64, 8, key="sidebar_batch_size")
    lr0        = st.number_input("Lernrate (η)", 1e-5, 1.0, 0.05, format="%.5f", key="sidebar_lr0")
    init_type  = st.selectbox("Initialisierung", ["Xavier", "He", "RandomShift"], index=0, key="sidebar_init_type")
    seed_w     = st.number_input("Seed Gewichte", 0, 10_000, 7, 1, key="sidebar_seed_w")

    st.header("Optimizer & Scheduler")
    optimizer     = st.selectbox("Optimizer", ["SGD", "Momentum", "Adam"], index=2, key="sidebar_optimizer")
    momentum_beta = st.slider("β (Momentum)", 0.5, 0.99, 0.9, 0.01, key="sidebar_momentum_beta")
    step_every    = st.number_input("LR-Step alle N Schritte (0=aus)", 0, 10_000, 0, 1, key="sidebar_step_every")
    gamma         = st.number_input("LR-Multiplikator γ", 0.01, 1.0, 0.5, format="%.2f", key="sidebar_gamma")

    st.header("Steuerung")
    btn_reset   = st.button("Reset Parameter", key="btn_reset")
    step_1      = st.button("Einzelschritt (Batch/Sample)", key="btn_step_1")
    run_epoch   = st.button("Eine Epoche (alle Batches/Samples)", key="btn_run_epoch")
    auto_run    = st.checkbox("Auto-Train (Epoche, solange App aktiv)", False, key="chk_auto_run")

# Daten erzeugen
X, y_vec, y_labels, k = gen_data(kind=ds_kind, n=n_samples, noise=noise, seed=seed_ds, stdize=stdize, k_classes=4)

if k == 2 and act_out == "Softmax":
    st.info("Für 2 Klassen ist Sigmoid üblicher. Softmax funktioniert, ist aber redundant.")
if k > 2 and act_out != "Softmax":
    st.warning("Mehrklassen-Ziel erkannt – empfehle Softmax + CrossEntropy.")

# Session-State
if "params" not in st.session_state or btn_reset:
    in_dim = X.shape[1]; out_dim = 1 if (k == 2 and act_out != "Softmax") else k
    st.session_state.params = init_weights(in_dim, hid_dim, out_dim, init_type=init_type, seed=seed_w)
    st.session_state.opt_state = OptimState()
    st.session_state.loss_history = []
    st.session_state.step_count = 0
    st.session_state.last_grads = None
    st.session_state.sample_index = 0

# Output-Dimension anpassen, falls geändert
needed_out = 1 if (k == 2 and act_out != "Softmax") else k
if st.session_state.params["W2"].shape[1] != needed_out:
    st.session_state.params = init_weights(X.shape[1], hid_dim, needed_out, init_type=init_type, seed=seed_w)
    st.session_state.opt_state = OptimState()
    st.session_state.loss_history = []
    st.session_state.step_count = 0
    st.session_state.last_grads = None

# Tabs
tabs = st.tabs(["Daten", "Feedforward", "Loss", "Backprop", "Updates & Training"])

with tabs[0]:
    st.subheader("Datenüberblick")
    st.write(f"X-Shape: **{X.shape}**, y-Shape: **{y_vec.shape}**, Klassen: **{k}**")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(X[:, 0], X[:, 1], c=y_labels, s=15)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    st.pyplot(fig)
    st.markdown("**Feedforward-Formeln**")
    st.markdown(r"""
    $$
    \begin{aligned}
    \mathbf{z}^{(1)} &= \mathbf{X}\mathbf{W}_{xh} + \mathbf{b}_h \\
    \mathbf{a}^{(1)} &= f_h(\mathbf{z}^{(1)}) \\[4pt]
    \mathbf{z}^{(2)} &= \mathbf{a}^{(1)}\mathbf{W}_{hy} + \mathbf{b}_y \\
    \hat{\mathbf{y}} &= f_o(\mathbf{z}^{(2)})
    \end{aligned}
    $$
    """)

with tabs[1]:
    st.subheader("Feedforward (aktueller Zustand)")
    y_hat, cache = forward_pass(X, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
    st.write(f"a1 (Hidden) Shape: {cache['a1'].shape}, a2/ŷ (Output) Shape: {y_hat.shape}")
    st.session_state.sample_index = st.slider(
        "Sample-Index (Tracing)", 0, X.shape[0] - 1, st.session_state.sample_index, 1, key="slider_trace_sample"
    )
    i = st.session_state.sample_index
    st.markdown("**Per-Sample-Tracing**")
    st.write("X[i]:", X[i:i+1]); st.write("z1[i]:", cache["z1"][i:i+1])
    st.write("a1[i]:", cache["a1"][i:i+1]); st.write("z2[i]:", cache["z2"][i:i+1]); st.write("ŷ[i]:", y_hat[i:i+1])
    st.markdown("**Gewichte (Heatmaps)**")
    heatmap_matrix(st.session_state.params["W1"], "W_xh")
    heatmap_matrix(st.session_state.params["W2"], "W_hy")
    plot_network_gradients(st.session_state.params["W1"], st.session_state.params["W2"], title="Rechengraph (ohne Gradienten)")

with tabs[2]:
    st.subheader("Loss-Berechnung")
    y_hat, _ = forward_pass(X, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
    if k == 2 and y_vec.shape[1] == 1 and act_out == "Sigmoid" and loss_type == "BCE":
        loss_val = bce_loss(y_vec, y_hat)
        st.markdown(r"**BCE:** $-\frac{1}{n}\sum [y\log \hat y + (1-y)\log(1-\hat y)]$")
    elif k > 2 and act_out == "Softmax" and loss_type == "CrossEntropy":
        loss_val = cross_entropy_loss(y_vec, y_hat)
        st.markdown(r"**Cross-Entropy (Mehrklassen):** $-\frac{1}{n}\sum \sum y_k \log \hat y_k$")
    else:
        loss_val = rmse_loss(y_vec, y_hat)
        st.markdown(r"**RMSE:** $\sqrt{\frac{1}{n}\sum (\hat y - y)^2}$")
    st.metric("Aktueller Loss", f"{loss_val:.6f}")
    if len(st.session_state.loss_history) > 0:
        plot_loss(st.session_state.loss_history)

with tabs[3]:
    st.subheader("Backprop – Gradienten & Diagnostik")
    y_hat, cache = forward_pass(X, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
    grads = backward_pass(y_vec, st.session_state.params, cache, act_hidden=act_hidden, act_out=act_out, loss=loss_type)
    st.session_state.last_grads = grads
    st.write("Shapes — dW1:", grads["dW1"].shape, "db1:", grads["db1"].shape, "dW2:", grads["dW2"].shape, "db2:", grads["db2"].shape)
    st.markdown("**Heatmaps der Gradienten**")
    heatmap_matrix(grads["dW1"], "∂Loss/∂W_xh"); heatmap_matrix(grads["dW2"], "∂Loss/∂W_hy")
    st.markdown("**Gradienten-Histogramme**")
    hist_values(grads["dW1"], "Histogramm dW1"); hist_values(grads["dW2"], "Histogramm dW2")
    plot_network_gradients(st.session_state.params["W1"], st.session_state.params["W2"],
                           G1=grads["dW1"], G2=grads["dW2"], title="Rechengraph (Linienstärke ∝ |Gradient|)")

with tabs[4]:
    st.subheader("Updates & Training")

    def do_one_update():
        # Datenportion je nach Batchmodus
        if batch_mode == "Full-Batch":
            Xb, yb = X, y_vec
        elif batch_mode == "Mini-Batch":
            idx = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[idx], y_vec[idx]
        else:  # SGD (online)
            ii = np.random.randint(0, X.shape[0])
            Xb, yb = X[ii:ii+1], y_vec[ii:ii+1]

        y_hat_b, cache_b = forward_pass(Xb, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
        g = backward_pass(yb, st.session_state.params, cache_b, act_hidden=act_hidden, act_out=act_out, loss=loss_type)

        lr_curr = step_lr(lr0, step_every, gamma, st.session_state.step_count)

        if optimizer == "SGD":
            st.session_state.params = sgd_update(st.session_state.params, g, lr_curr)
        elif optimizer == "Momentum":
            st.session_state.params, st.session_state.opt_state = momentum_update(st.session_state.params, g, st.session_state.opt_state, lr_curr, beta=momentum_beta)
        else:
            st.session_state.params, st.session_state.opt_state = adam_update(st.session_state.params, g, st.session_state.opt_state, lr_curr)

        # Loss global (auf allen Daten) tracken
        y_hat2, _ = forward_pass(X, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
        if (k == 2 and y_vec.shape[1] == 1 and act_out == "Sigmoid" and loss_type == "BCE"):
            loss_now = bce_loss(y_vec, y_hat2)
        elif (k > 2 and act_out == "Softmax" and loss_type == "CrossEntropy"):
            loss_now = cross_entropy_loss(y_vec, y_hat2)
        else:
            loss_now = rmse_loss(y_vec, y_hat2)
        st.session_state.loss_history.append(loss_now)
        st.session_state.step_count += 1
        return g, loss_now

    if step_1:
        _, loss_now = do_one_update()
        st.success(f"Einzelschritt ausgeführt. Loss={loss_now:.6f}")

    if run_epoch:
        idx = np.arange(X.shape[0])
        if batch_mode == "Full-Batch":
            _, loss_now = do_one_update()
        elif batch_mode == "Mini-Batch":
            np.random.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                batch = idx[start:start+batch_size]
                Xb, yb = X[batch], y_vec[batch]
                y_hat_b, cache_b = forward_pass(Xb, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
                g = backward_pass(yb, st.session_state.params, cache_b, act_hidden=act_hidden, act_out=act_out, loss=loss_type)
                lr_curr = step_lr(lr0, step_every, gamma, st.session_state.step_count)
                if optimizer == "SGD":
                    st.session_state.params = sgd_update(st.session_state.params, g, lr_curr)
                elif optimizer == "Momentum":
                    st.session_state.params, st.session_state.opt_state = momentum_update(st.session_state.params, g, st.session_state.opt_state, lr_curr, beta=momentum_beta)
                else:
                    st.session_state.params, st.session_state.opt_state = adam_update(st.session_state.params, g, st.session_state.opt_state, lr_curr)
                y_hat2, _ = forward_pass(X, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
                loss_now = (bce_loss(y_vec, y_hat2) if (k == 2 and y_vec.shape[1] == 1 and act_out == "Sigmoid" and loss_type == "BCE")
                            else cross_entropy_loss(y_vec, y_hat2) if (k > 2 and act_out == "Softmax" and loss_type == "CrossEntropy")
                            else rmse_loss(y_vec, y_hat2))
                st.session_state.loss_history.append(loss_now); st.session_state.step_count += 1
        else:  # SGD
            for ii in np.random.permutation(len(idx)):
                Xi = X[ii:ii+1]; yi = y_vec[ii:ii+1]
                y_hat_i, cache_i = forward_pass(Xi, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
                g = backward_pass(yi, st.session_state.params, cache_i, act_hidden=act_hidden, act_out=act_out, loss=loss_type)
                lr_curr = step_lr(lr0, step_every, gamma, st.session_state.step_count)
                if optimizer == "SGD":
                    st.session_state.params = sgd_update(st.session_state.params, g, lr_curr)
                elif optimizer == "Momentum":
                    st.session_state.params, st.session_state.opt_state = momentum_update(st.session_state.params, g, st.session_state.opt_state, lr_curr, beta=momentum_beta)
                else:
                    st.session_state.params, st.session_state.opt_state = adam_update(st.session_state.params, g, st.session_state.opt_state, lr_curr)
                y_hat2, _ = forward_pass(X, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
                loss_now = (bce_loss(y_vec, y_hat2) if (k == 2 and y_vec.shape[1] == 1 and act_out == "Sigmoid" and loss_type == "BCE")
                            else cross_entropy_loss(y_vec, y_hat2) if (k > 2 and act_out == "Softmax" and loss_type == "CrossEntropy")
                            else rmse_loss(y_vec, y_hat2))
                st.session_state.loss_history.append(loss_now); st.session_state.step_count += 1
        st.success("Epoche abgeschlossen.")

    if auto_run:
        _, _ = do_one_update()

    st.markdown("**Aktuelle Lernrate (nach Scheduler):**")
    st.code(f"η_t = {step_lr(lr0, step_every, gamma, st.session_state.step_count):.6f}\n"
            f"Schritte = {st.session_state.step_count} (LR-Step alle {step_every}, γ={gamma})")

    st.markdown("**Metriken**")
    y_hat_all, _ = forward_pass(X, st.session_state.params, act_hidden=act_hidden, act_out=act_out)
    if k == 2:
        y_prob = y_hat_all.reshape(-1); y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_labels, y_pred) if SKLEARN_AVAILABLE else None
        if cm is not None: st.write("Confusion-Matrix"); st.write(cm)
        acc = accuracy_score(y_labels, y_pred) if SKLEARN_AVAILABLE else np.mean(y_pred == y_labels)
        st.write(f"Accuracy: {acc:.4f}")
        if SKLEARN_AVAILABLE:
            try: st.write(f"ROC-AUC: {roc_auc_score(y_labels, y_prob):.4f}")
            except Exception: pass
    else:
        y_pred = np.argmax(y_hat_all, axis=1); acc = np.mean(y_pred == y_labels)
        st.write(f"Accuracy: {acc:.4f}")

    st.markdown("**Download Artefakte**")
    params_json = json.dumps({k: v.tolist() for k, v in st.session_state.params.items()})
    st.download_button("Gewichte (JSON) herunterladen", params_json, file_name="nn_params.json", key="dl_params_json")
    hist_csv = "step,loss\n" + "\n".join(f"{i},{v}" for i, v in enumerate(st.session_state.loss_history))
    st.download_button("Loss-Verlauf (CSV) herunterladen", hist_csv, file_name="loss_history.csv", key="dl_loss_csv")
