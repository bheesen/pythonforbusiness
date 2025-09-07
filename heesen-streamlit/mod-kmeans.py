# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\mod-kmeans.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances_argmin

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="K-Means Clustering", layout="centered")
st.title("K-Means Clustering Demo")
st.markdown(
    "Diese App illustriert die wesentlichen Prozessschritte des K-Means Clustering. "
    "Sie wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------
with st.sidebar:
    st.title("Parameter")
    iteration = st.slider("Iteration", 0, 2, 0)

# --- Daten vorbereiten ---
np.random.seed(42)

# Daten bewusst so gewählt, dass drei Iterationen nötig sind
cluster_1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.15, 0.05], [0.25, 0.15], [0.05, 0.25]])
cluster_2 = np.array([[2.0, 0.0], [2.1, 0.2], [1.9, -0.1], [2.2, 0.1], [1.95, 0.05]])
cluster_3 = np.array([[1.0, 2.5], [1.1, 2.6], [0.9, 2.4], [1.2, 2.7], [1.05, 2.55]])
X = np.vstack([cluster_1, cluster_2, cluster_3])

# Startzentren weit entfernt von echten Clustern
initial_centers = np.array([[2.5, 2.5], [0.0, 2.0], [2.0, -0.5]])
centers = initial_centers.copy()
centers_history = [centers.copy()]
labels_history = []
label_changes = []

# Durchführung bis Konvergenz oder max. 10 Iterationen
max_iter = 10
for i in range(max_iter):
    labels = pairwise_distances_argmin(X, centers)
    labels_history.append(labels)
    if i > 0:
        change_count = np.sum(labels != labels_history[-2])
        label_changes.append(change_count)
        if change_count == 0:
            break
    else:
        label_changes.append(len(X))  # beim ersten Schritt alle als initiale Veränderung zählen

    new_centers = np.array([X[labels == j].mean(axis=0) for j in range(3)])
    centers = new_centers
    centers_history.append(centers.copy())

# Farben
colors = ['blue', 'orange', 'green']

# Plotly-Grafik
fig = go.Figure()
centers = centers_history[iteration]
labels = pairwise_distances_argmin(X, centers)

for j in range(3):
    cluster_points = X[labels == j]
    fig.add_trace(go.Scatter(
        x=cluster_points[:, 0], y=cluster_points[:, 1],
        mode='markers',
        marker=dict(size=12, color=colors[j], line=dict(width=1, color='black')),
        name=f'Cluster {j+1} ({centers[j][0]:.2f}, {centers[j][1]:.2f})'
    ))

    if len(cluster_points) > 0:
        radius = np.max(np.linalg.norm(cluster_points - centers[j], axis=1))
        angle = np.linspace(0, 2*np.pi, 100)
        circle_x = centers[j][0] + radius * np.cos(angle)
        circle_y = centers[j][1] + radius * np.sin(angle)
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines',
            line=dict(color=colors[j], dash='dot'),
            showlegend=False
        ))

# Zentren
fig.add_trace(go.Scatter(
    x=centers[:, 0], y=centers[:, 1],
    mode='markers', marker=dict(symbol='x', size=16, color='black'),
    name='Zentren'
))

fig.update_layout(
    width=700, height=600,
    xaxis_title="Feature 1", yaxis_title="Feature 2",
    title=f"K-Means Iteration {iteration}",
    plot_bgcolor="white",   # Hintergrund der Plotfläche
    paper_bgcolor="white"   # Hintergrund um die Plotfläche herum
)

# Achsen anpassen (direkt nach dem Layout)
fig.update_xaxes(showline=True, linewidth=1, linecolor="black", gridcolor="lightgray")
fig.update_yaxes(showline=True, linewidth=1, linecolor="black", gridcolor="lightgray")

st.plotly_chart(fig)

# Anzeige der Anzahl der veränderten Zuweisungen
if iteration > 0:
    st.info(f"Anzahl der veränderten Clusterzuweisungen gegenüber vorheriger Iteration: {label_changes[iteration]}")
else:
    st.info("Initiale Clusterzuweisung (keine Voriteration zum Vergleichen)")

st.markdown("""
**Hinweis:** Die Startzentren und die Datenpunkte wurden so gewählt, dass der K-Means-Algorithmus genau drei Iterationen benötigt,
bevor sich die Clusterzuweisungen nicht mehr ändern.
""")

