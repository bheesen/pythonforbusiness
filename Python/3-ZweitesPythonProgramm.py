# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:49:07 2024

@author: bernd
"""

#%% Visualisierung mit matplotlib
import matplotlib.pyplot as plt
#%%% Einfaches Balkendiagramm
# Daten des Diagramms
values = [1, 2, 3]
value_counts = [20, 30, 50]
labels = ["Schwein", "Rind", "Geflügel"]
colors = ["#1B676B", "#519548", "#88C425"]
# Abbildung konfigurieren
config = {
  "x": values,
  "height": value_counts,
  "width": 0.75,
  "tick_label": labels,
  "color": colors,
}
# Abbildung erstellen
plt.bar(**config)
# Beschriftungen festlegen und hinzufügen
tfont = {
  "family": "sans-serif",
  "weight": "bold",
  "size": "14",
}
lfont = {
  "family": "sans-serif",
  "size": "12",
}
plt.title("Fleischkonsum in Deutschland".upper(), **tfont)
plt.xlabel("Fleisch", **lfont)
plt.ylabel("Anteil", **lfont)
# Abbildung speichern und anzeigen
plt.savefig("balkendiagramm-Fleischkonsum", dpi=300)
plt.show()