# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\mod-svm.py

# streamlit_app_svm_explorer.py
# Interaktive, didaktische SVM-Exploration (ähnlich zur NN-App)
# - Datensätze: moons, circles, blobs (k=4), linear, XOR ("Eigene Beispieldaten")
# - Kernel: linear, poly, rbf (mit C, gamma, degree)
# - Visuals: Entscheidungsgrenze, Margin (linear), Support Vectors
# - Metriken: Confusion-Matrix, Accuracy, ROC-AUC (binär)
# - Hinge-Loss-Anschaulichkeit (linear)
# - Reproduzierbarkeit (Seed), Standardisierung (Pipeline)
# - Eindeutige Keys für alle Widgets

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="Support Vector Machines", layout="wide")
st.title("Support Vector Machines (SVM)")
st.markdown(
    "Diese App illustriert die wesentlichen Prozessschritte von SVM: "
    "Sie wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------
with st.sidebar:
    st.header("Daten & Problem")
    ds_kind = st.selectbox("Datensatz", ["moons", "circles", "blobs", "linear", "Eigene (XOR)"], index=3, key="ds_kind") # Index 3 entspricht "linear"
    n_samples = st.slider("Anzahl Samples", 30, 1000, 200, 10, key="n_samples")
    noise = st.slider("Rauschen", 0.0, 0.5, 0.2, 0.01, key="noise")
    stdize = st.checkbox("Standardisieren (z-Score)", True, key="stdize")
    seed = st.number_input("Seed", 0, 10000, 42, 1, key="seed")

    st.header("SVM-Kernel & Hyperparameter")
    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], index=0, key="kernel")   # Index 0 entspricht "linear"
    C = st.number_input("C (Regularisierung)", 0.001, 1e4, 1.0, format="%.3f", key="C")
    gamma = st.selectbox("gamma (rbf/poly)", ["scale", "auto", "manuell"], index=0, key="gamma_sel")
    gamma_val = st.number_input("γ (manuell)", 1e-6, 1e2, 0.5, format="%.6f", key="gamma_val") if gamma == "manuell" else None
    degree = st.slider("degree (poly)", 2, 8, 3, 1, key="degree") if kernel == "poly" else None

    st.header("Steuerung")
    show_margin = st.checkbox("Margin visualisieren (linear)", True, key="show_margin")
    show_scores = st.checkbox("Entscheidungsfunktion anzeigen (Heatmap)", True, key="show_scores")
    show_hinge = st.checkbox("Hinge-Loss (linear) zeigen", False, key="show_hinge")

# ---------------- Daten erzeugen ----------------
rng = np.random.default_rng(seed)
if ds_kind == "Eigene (XOR)":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,1,1,0], dtype=int)
elif ds_kind == "moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
elif ds_kind == "circles":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
elif ds_kind == "blobs":
    X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.5, random_state=seed)
else:  # linear
    X = rng.normal(0, 1, (n_samples, 2))
    w = np.array([1.5, -2.0])
    y = (X @ w + 0.2 * rng.normal(0, 1, n_samples) > 0).astype(int)

k = len(np.unique(y))

# ---------------- Modell bauen (Pipeline) ----------------
if kernel == "linear":
    # LinearSVC hat andere API bzgl. decision_function_shape; SVC(kernel='linear') ist hier anschaulicher
    svc = SVC(kernel="linear", C=C)
elif kernel == "rbf":
    svc = SVC(kernel="rbf", C=C, gamma=(gamma_val if gamma == "manuell" else gamma))
else:  # poly
    svc = SVC(kernel="poly", C=C, degree=degree, gamma=(gamma_val if gamma == "manuell" else gamma))

if stdize:
    model = Pipeline([("scaler", StandardScaler()), ("svc", svc)])
else:
    model = Pipeline([("svc", svc)])

model.fit(X, y)

# ---------------- Tabs ----------------
tabs = st.tabs(["Daten", "Entscheidungsfläche", "Metriken", "Hinge-Loss (linear)"])

# ---------------- Plot Helfer ----------------
def plot_decision_regions(X, y, model, title="", show_scores=True, show_margin=False, is_linear=False):
    x_min, x_max = X[:,0].min()-1.0, X[:,0].max()+1.0
    y_min, y_max = X[:,1].min()-1.0, X[:,1].max()+1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    XY = np.c_[xx.ravel(), yy.ravel()]

    # decision_function bevorzugen; fallback: predict_proba (falls verfügbar)
    try:
        Z = model.decision_function(XY)
        if Z.ndim > 1:
            Z_plot = Z.max(axis=1)  # für One-vs-Rest-Mehrklassen
        else:
            Z_plot = Z
    except Exception:
        Z_plot = model.predict_proba(XY).max(axis=1)

    Zc = model.predict(XY)

    fig, ax = plt.subplots(figsize=(6, 4))
    # Klassengebiet
    ax.contourf(xx, yy, Zc.reshape(xx.shape), alpha=0.25, levels=np.arange(-0.5, k+0.5, 1), cmap="viridis")

    # Heatmap der Scores
    if show_scores:
        im = ax.imshow(Z_plot.reshape(xx.shape), origin="lower",
                       extent=(x_min, x_max, y_min, y_max), alpha=0.35, aspect="auto", cmap="RdBu")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Entscheidungsfunktion/Score")

    # Margin/Trennlinie nur sinnvoll im linearen, binären Fall
    if show_margin and is_linear and k == 2:
        # Für linearen Kernel: Trennlinie f(x)=0 und Margins f(x)=±1
        # Erzeuge dichte Samples auf Gitter und zeichne Niveaulinien
        cs = ax.contour(xx, yy, Z_plot.reshape(xx.shape), levels=[-1, 0, 1],
                        colors=["k","k","k"], linestyles=["--","-","--"], linewidths=[1,2,1])
        # Legende improvisieren
        fmt = {cs.levels[0]: '-1 (Margin)', cs.levels[1]: '0 (Grenze)', cs.levels[2]: '+1 (Margin)'}
        ax.clabel(cs, inline=True, fontsize=8, fmt=fmt)

    # Datenpunkte
    scatter = ax.scatter(X[:,0], X[:,1], c=y, s=20, edgecolor="k", cmap="viridis", label="Daten")
    ax.set_title(title)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    # Support Vectors markieren (nur bei SVC verfügbar)
    try:
        svc_est = model.named_steps["svc"]
        if hasattr(svc_est, "support_") and stdize is False:
            sv = svc_est.support_
            ax.scatter(X[sv,0], X[sv,1], s=80, facecolors="none", edgecolors="red", linewidths=1.5, label="Support Vectors")
        elif hasattr(svc_est, "support_") and stdize is True:
            # Wenn standardisiert, dann liegen SVs im transformierten Raum; wir markieren die Trainingspunkte, die Support-Indices haben
            sv = svc_est.support_
            ax.scatter(X[sv,0], X[sv,1], s=80, facecolors="none", edgecolors="red", linewidths=1.5, label="Support Vectors")
    except Exception:
        pass

    ax.legend(loc="best")
    st.pyplot(fig)

with tabs[0]:
    st.subheader("Datenüberblick")
    st.write(f"X-Shape: **{X.shape}**, Klassen: **{k}**, Kernel: **{kernel}**, C={C}, "
             + (f"γ={gamma_val}" if gamma == "manuell" and kernel in ["rbf","poly"] else f"γ={gamma}")
             + (f", degree={degree}" if kernel=="poly" else ""))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X[:,0], X[:,1], c=y, s=25, cmap="viridis", edgecolor="k")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title("Rohdaten (2D)")
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Entscheidungsfläche, Margin & Support Vectors")
    is_linear = (kernel == "linear")
    plot_decision_regions(
        X, y, model,
        title="SVM-Entscheidungsfläche",
        show_scores=show_scores,
        show_margin=show_margin,
        is_linear=is_linear
    )

with tabs[2]:
    st.subheader("Metriken (Trainingsdaten)")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    st.write(f"Accuracy: **{acc:.4f}**")

    if k == 2:
        # ROC-AUC (binär) – decision_function bevorzugt
        try:
            scores = model.decision_function(X)
            if scores.ndim > 1: scores = scores[:,1]
        except Exception:
            # Fallback proba
            try:
                scores = model.predict_proba(X)[:,1]
            except Exception:
                scores = (y_pred == 1).astype(float)
        try:
            auc = roc_auc_score(y, scores)
            fpr, tpr, _ = roc_curve(y, scores)
            st.write(f"ROC-AUC: **{auc:.4f}**")
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
            ax.plot([0,1],[0,1], "k--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
            st.pyplot(fig)
        except Exception:
            pass

    cm = confusion_matrix(y, y_pred)
    st.write("Confusion-Matrix")
    st.write(cm)

with tabs[3]:
    st.subheader("Hinge-Loss-Anschaulichkeit (nur linear, binär)")
    st.markdown("""
Die (weiche) **Hinge-Loss** für ein Sample \\((x_i, y_i)\\) (mit Labels \\(y_i \\in \\{-1, +1\\}\\)) lautet:
\\[
L_i = \\max(0, 1 - y_i \\cdot f(x_i)),
\\]
wobei \\(f(x)=w^\\top x + b\\) die **Entscheidungsfunktion** ist.  
- **Margin-Bedingung**: korrekt & sicher klassifiziert, wenn \\(y_i f(x_i) \\ge 1\\)  \\(\\Rightarrow L_i=0\\)  
- **Strafterm**: falls \\(y_i f(x_i) < 1\\)  \\(\\Rightarrow L_i>0\\)
""")
    if kernel != "linear" or k != 2:
        st.info("Hinge-Loss-Demo setzt **linearen, binären** SVM voraus.")
    else:
        # decision function auf Trainingspunkten
        try:
            scores = model.decision_function(X)
            # y in {-1, +1}
            y_pm = np.where(y==1, 1, -1)
            hinge = np.maximum(0.0, 1.0 - y_pm * scores)
            st.write(f"Durchschnittliche Hinge-Loss: **{hinge.mean():.4f}**")

            fig, ax = plt.subplots(figsize=(6,4))
            # Punkte einfärben nach hinge-Loss-Größe
            sc = ax.scatter(X[:,0], X[:,1], c=hinge, s=30, cmap="magma", edgecolor="k")
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label("Hinge-Loss pro Sample")
            ax.set_title("Hinge-Loss (größer = näher/auf falscher Seite der Grenze)")
            ax.set_xlabel("x1"); ax.set_ylabel("x2")
            st.pyplot(fig)
        except Exception:
            st.warning("Entscheidungsfunktion nicht verfügbar.")

# ---------------- Didaktische Hinweise (kurz) ----------------
st.markdown("""
**Didaktische Kurzfassung (für Business-Anwendung):**
- **C (Regularisierung)** steuert den Trade-off zwischen **großer Margin** (starker Regularisierung, kleines C) und **wenigen Fehlklassifikationen** (großes C).
- **γ (RBF/poly)** steuert die **Radii** der Gaussian-Basisfunktionen bzw. die **Welligkeit** der Entscheidungsgrenze:
  kleine γ → glattere Grenze; große γ → sehr flexible Grenze (Overfitting-Gefahr).
- **linear vs. RBF/poly**: Lineare SVM ist robust und gut interpretierbar (Gewichte ≈ Feature-Bedeutung).
  RBF/poly fangen **nichtlineare Strukturen** ein (z. B. Kreise, XOR).
- **Support Vectors** bestimmen die Lage der Trennlinie wesentlich; weniger SVs → kompakteres Modell.
""")
