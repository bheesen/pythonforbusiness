# Aufruf in Anaconda Prompt mit:
# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\mod-qlearning.py
#
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="Q-Learning", layout="centered")
st.title("Q-Learning: Kundenbindung im Einzelhandel")
st.markdown(
    "Diese App illustriert die wesentlichen Prozessschritte des Algorithums Q-Learning anhand eines Business-Beispiels zur Kundenbindung im Einzelhandel, das entscheidet, welche Marketingmaßnahme bei unterschiedlichem Kundenstatus den größten langfristigen Nutzen bringt. "
    "Sie wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------
with st.sidebar:
    st.header("Parameter")
    alpha = st.slider("Lernrate (α)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.slider("Diskontfaktor (γ)", 0.1, 1.0, 0.9, 0.05)
    epsilon = st.slider("Explorationsrate (ε)", 0.0, 1.0, 0.2, 0.05)
    episodes = st.slider("Anzahl Trainings-Episoden", 100, 5000, 1000, 100)

# Zustände und Aktionen
states = [0, 1, 2]
actions = [0, 1, 2]
state_labels = ["aktiv\n ", "inaktiv\nkurz", "inaktiv\nlang"]
action_labels = ["Keine Aktion", "Gutschein", "Treuepunkte"]

# Q-Tabelle initialisieren
q_table = np.zeros((len(states), len(actions)))

# Belohnungsfunktion
def simulate_customer_response(state, action):
    if state == 0:
        return 0
    elif state == 1:
        return 10 if action in [1, 2] else 0
    elif state == 2:
        return 10 if action == 2 else 0
    return 0

# Q-Learning-Training
for _ in range(episodes):
    state = np.random.choice(states)
    if np.random.rand() < epsilon:
        action = np.random.choice(actions)
    else:
        action = np.argmax(q_table[state])
    reward = simulate_customer_response(state, action)
    next_state = np.random.choice(states)
    next_max = np.max(q_table[next_state])
    q_table[state, action] += alpha * (reward + gamma * next_max - q_table[state, action])

# Ergebnisse anzeigen
df_q = pd.DataFrame(q_table, columns=action_labels, index=state_labels)
st.subheader("Ergebnis: Q-Werte der Aktionen in jedem Kundenstatus")
st.dataframe(df_q.style.format("{:.0f}"))

# Visualisierung als Heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df_q, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Q-Wert"})
ax.set_xlabel("Aktion")
ax.set_ylabel("Kundenstatus")
st.pyplot(fig)
