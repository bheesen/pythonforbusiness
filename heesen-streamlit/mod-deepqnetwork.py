# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\mod-deepqnetwork.py

# Szenario: 
# - Lagerbestand: max. 20 Einheiten
# - Nachfrage: zufällig zwischen 0 und 10
# - Aktionen: Bestellung von 0 bis 5 Einheiten
# Ziel: optimale Bestellpolitik, um Fehlmengen und Lagerkosten zu minimieren (min Überbestand)   
# Die Q-Werte zeigen, wie der Agent den erwarteten Nutzen (Reward) für jede Kombination aus Lagerbestand und möglicher Bestellmenge bewertet

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import random
import seaborn as sns

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="DeepQ-Networks", layout="centered")
st.title("DeepQ-Networks: Lagerbestandsmanagement")
st.markdown(
    "Diese App illustriert die wesentlichen Prozessschritte des Algorithums DeepQ-Networks anhand eines Business-Beispiels zur Lagerbestandsoptimierung. "
    "Sie wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------
with st.sidebar:
    st.header("Parameter")
    EPISODES = st.sidebar.slider("Anzahl Episoden", 10, 500, 50, step=10)
    GAMMA = st.sidebar.slider("Diskontfaktor (Gamma)", 0.80, 0.99, 0.95)
    EPSILON_DECAY = st.sidebar.slider("Epsilon-Decay", 0.90, 0.999, 0.995)
    LEARNING_RATE = st.sidebar.select_slider("Lernrate", options=[0.01, 0.005, 0.001, 0.0005], value=0.001)

# Parameter
MAX_STOCK = 20
ACTIONS = [0, 1, 2, 3, 4, 5]
STATE_SPACE = MAX_STOCK + 1
ACTION_SPACE = len(ACTIONS)

# Environment
def get_demand():
    return np.random.randint(0, 11)

def step(state, action):
    order = ACTIONS[action]
    stock = min(state + order, MAX_STOCK)
    demand = get_demand()
    sold = min(stock, demand)
    stock -= sold
    holding_cost = stock * 0.1
    lost_sales_penalty = (demand - sold) * 2
    reward = -holding_cost - lost_sales_penalty
    next_state = stock
    return next_state, reward, False

# Modell
def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(ACTION_SPACE, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Training
model = create_model()
target_model = create_model()
target_model.set_weights(model.get_weights())

EPSILON = 1.0
EPSILON_MIN = 0.1
BATCH_SIZE = 32
MEMORY = []
rewards_per_episode = []

for episode in range(EPISODES):
    state = np.random.randint(0, MAX_STOCK + 1)
    total_reward = 0
    for t in range(10):
        if np.random.rand() < EPSILON:
            action = np.random.randint(ACTION_SPACE)
        else:
            q_values = model.predict(np.array([[state]]), verbose=0)
            action = np.argmax(q_values[0])
        next_state, reward, done = step(state, action)
        MEMORY.append((state, action, reward, next_state, done))
        if len(MEMORY) > 2000:
            MEMORY.pop(0)
        state = next_state
        total_reward += reward
        if len(MEMORY) >= BATCH_SIZE:
            batch = random.sample(MEMORY, BATCH_SIZE)
            states, targets = [], []
            for s, a, r, s_next, d in batch:
                q = model.predict(np.array([[s]]), verbose=0)[0]
                q_next = target_model.predict(np.array([[s_next]]), verbose=0)[0]
                q[a] = r + (0 if d else GAMMA * np.max(q_next))
                states.append([s])
                targets.append(q)
            model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
    rewards_per_episode.append(total_reward)
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

# Plot: Reward Verlauf
st.subheader("Gesamtreward pro Episode")
fig1, ax1 = plt.subplots()
ax1.plot(rewards_per_episode)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.set_title("DQN – Lagerbestandsmanagement")
ax1.grid(True)
st.pyplot(fig1)

# Visualisierung der Q-Werte
st.subheader("Gelerntes Entscheidungsverhalten (Q-Werte)")
q_matrix = np.zeros((STATE_SPACE, ACTION_SPACE))
for state in range(STATE_SPACE):
    q_values = model.predict(np.array([[state]]), verbose=0)[0]
    q_matrix[state, :] = q_values

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(q_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=ACTIONS,
            yticklabels=[f"{s}" for s in range(STATE_SPACE)], ax=ax2)
ax2.set_xlabel("Aktion (Bestellmenge)")
ax2.set_ylabel("Lagerbestand")
ax2.set_title("Q-Wert-Matrix nach Training")
st.pyplot(fig2)
