# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\mod-kmeans-business.py

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

# Warnung verhindern
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="K-Means Clustering – Interaktive Simulation", layout="centered")
st.title("K-Means Clustering – Interaktive Simulation")
st.markdown(
    "Diese App illustriert die wesentlichen Prozessschritte des K-Means Clustering anhand eines Business-Beispiels. "
    "Sie wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------
with st.sidebar:
    st.title("Kundenanzahl")
    n_kunden = st.slider("Anzahl der Kunden", 30, 200, 80)
    st.title("Clusteranzahl")
    k_selected = st.slider("Anzahl der Cluster k:", 2, 10, 3)

# -----------------------------
# 1. Daten generieren
# -----------------------------
st.subheader("1️⃣ Datensatz-Generierung")

np.random.seed(42)
einkommen = np.random.randint(20000, 100000, n_kunden)
kaufhaeufigkeit = np.random.choice([1, 2, 3, 4, 5, 10, 12], size=n_kunden,
                                   p=[0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1])
jahresumsatz = (einkommen * np.random.uniform(0.05, 0.2, size=n_kunden) * (kaufhaeufigkeit / 5)
                + np.random.randint(-2000, 2000, n_kunden)).astype(int)
jahresumsatz = np.clip(jahresumsatz, 500, None)

df = pd.DataFrame({'Einkommen': einkommen, 'Jahresumsatz': jahresumsatz})
st.dataframe(df.head())

# -----------------------------
# 2. Normalisierung
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Einkommen', 'Jahresumsatz']])

# -----------------------------
# 3. Elbow und Silhouette-Methode
# -----------------------------
st.subheader("2️⃣ Bestimmung der optimalen Clusteranzahl")

range_k = range(2, 11)
inertias = []
silhouette_scores = []

for k in range_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plots
fig1, ax1 = plt.subplots()
ax1.plot(range_k, inertias, marker='o')
ax1.set_title("Elbow-Methode (Trägheit)")
ax1.set_xlabel("Clusteranzahl k")
ax1.set_ylabel("Inertia")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(range_k, silhouette_scores, marker='o', color='green')
ax2.set_title("Silhouette-Score")
ax2.set_xlabel("Clusteranzahl k")
ax2.set_ylabel("Score")
st.pyplot(fig2)

st.markdown("""
Die **Elbow-Methode** zeigt, ab wann zusätzliche Cluster nur noch wenig Verbesserung bringen (Knickpunkt).
Der **Silhouette-Score** bewertet, wie klar die Cluster voneinander getrennt sind (höher = besser).
""")

# -----------------------------
# 4. Plot-Funktion für Iterationen
# -----------------------------
def plot_iteration_voronoi(X, labels, zentren, iteration, best_k):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    kmeans_temp = KMeans(n_clusters=best_k, init=zentren, n_init=1, max_iter=1, random_state=42)
    kmeans_temp.fit(X)
    Z = kmeans_temp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFC0CB', '#E6E6FA', '#FFD580', '#90EE90', '#FFB6C1']
    cmap_bg = ListedColormap(colors[:best_k])
    cmap_pts = ListedColormap(['#FF0000', '#00AA00', '#0000FF', '#999900', '#FF69B4', '#8A2BE2', '#FFD700', '#008080', '#DC143C'])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap_pts, s=80, edgecolors='k')
    ax.scatter(zentren[:, 0], zentren[:, 1], c='black', marker='X', s=200, label='Zentren')
    ax.set_title(f"K-Means – Iteration {iteration} (k={best_k})")
    ax.set_xlabel("Einkommen (skaliert)")
    ax.set_ylabel("Jahresumsatz (skaliert)")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# 5. Iterative K-Means-Ausführung
# -----------------------------
if st.button("Clustering starten"):
    kmeans = KMeans(n_clusters=k_selected, init='k-means++', max_iter=1, n_init=1, random_state=42)
    kmeans.fit(X_scaled)
    zentren = kmeans.cluster_centers_
    labels = kmeans.labels_
    plot_iteration_voronoi(X_scaled, labels, zentren, 0, k_selected)

    iteration = 1
    max_iterationen = 50
    while iteration <= max_iterationen:
        prev_labels = labels.copy()
        kmeans = KMeans(n_clusters=k_selected, init=zentren, max_iter=1, n_init=1, random_state=42)
        kmeans.fit(X_scaled)
        zentren = kmeans.cluster_centers_
        labels = kmeans.labels_
        plot_iteration_voronoi(X_scaled, labels, zentren, iteration, k_selected)
        if np.array_equal(labels, prev_labels):
            st.success(f"Stabile Clusterzuweisungen nach {iteration} Iterationen erreicht.")
            break
        iteration += 1
