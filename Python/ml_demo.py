# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 15:07:41 2025
@author: Bernd Heesen
@web:    www.profheesen.de

Testskript für ml_summary und ml_plot Funktionen
Erzeugt Simulationsdaten und durchläuft alle Varianten der beiden Funktionen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import os
import sys
# Legt die Verzeichnisse für eigene Module und Daten fest.
# - module_path: enthält eigene Python-Module (z. B. ml_plot.py, ml_summary.py),
#   die nicht im aktuellen Arbeitsverzeichnis liegen und so trotzdem importiert werden können.
#   von https://github.com/bheesen/pythonforbusiness/tree/main/Python/EigeneModule
# - data_path: enthält die Daten, die in den Beispielen und Übungen genutzt werden
#   von https://github.com/bheesen/pythonforbusiness/tree/main/Python/EigeneDaten
module_path = r"C:/Users/bernd/Documents/A-Python/EigeneModule"
data_path = r"C:/Users/bernd/Documents/A-Python/EigeneDaten"

# Modulpfad zu sys.path hinzufügen (für Imports von Modulen)
if module_path not in sys.path:
    sys.path.append(module_path)
from ml_plot import ml_plot, ml_colour_nom, ml_colour_ord, ml_colour_spect
from ml_summary import ml_summary

# ---------------- Flugdaten --------------------------------------------------
from joblib import load
datasets = load(os.path.join(data_path, "datasets.joblib"))
flug = datasets['Flights']

# ---------------- Autodaten --------------------------------------------------
autos=pd.read_csv(os.path.join(data_path, "autos.csv"),sep=",")
autos = autos[autos['Marke'].isin(['audi', 'bmw', 'opel'])]
autos_anzahl = autos.groupby(["Marke", "Kategorie"]).size().reset_index(name="Anzahl")

mtcars = sm.datasets.get_rdataset('mtcars').data.copy()
mtcars['Verbrauch100km'] = round(235.215 / mtcars['mpg'], 1)
mtcars['Gewicht'] = round(0.453592 * mtcars['wt'] * 1000, 0)
mtcars['Zylinder'] = mtcars['cyl']
mtcars['Marke'] = mtcars.index.to_series().str.split().str[0]
mtcars.loc[1, 'Verbrauch100km'] = 5
mtcars_filtered = mtcars[mtcars["Marke"].isin(["Merc", "Fiat", "Toyota"])]

# ---------------- Demodaten --------------------------------------------------
df_demo = pd.DataFrame({
    "Gruppe": pd.Categorical(
        pd.Series(["A", "B", "C", "D", "E", "A", "C", "E", "B", "D", "A", "A", "C"]),
        categories=["A", "B", "C", "D", "E"]
    ),
    "Wert": pd.Series(np.random.randint(1, 8, 13)),
    "Bewertung": pd.Categorical(
        pd.Series(["hoch", "mittel", "niedrig", "hoch", "mittel", "hoch",
                   "niedrig", "mittel", "hoch", "niedrig", "hoch", "mittel", "hoch"]),
        categories=["niedrig", "mittel", "hoch"],
        ordered=True
    ),
    "Stimmung": pd.Categorical(
        pd.Series(["sehr gut", "gut", "neutral", "schlecht", "sehr schlecht",
                   "gut", "sehr gut", "gut", "neutral", "schlecht",
                   "gut", "sehr schlecht", "gut"]),
        categories=["sehr gut", "gut", "neutral", "schlecht", "sehr schlecht"],
        ordered=True
    )
})

# ---------------- Simulationsdaten -------------------------------------------
# Daten generieren: 3x100 Zufallszahlen aus N(0,10) kumuliert je Zeile
np.random.seed(126)
werte = np.random.normal(loc=0, scale=10, size=(3, 100))
werte_kumuliert = np.cumsum(werte, axis=1)
# DataFrames für X1, X2, X3 erstellen
X1 = pd.DataFrame({
    "Minuten": np.arange(1, 101),
    "Variable": "X1",
    "Wert": werte_kumuliert[0]
})
X2 = pd.DataFrame({
    "Minuten": np.arange(1, 101),
    "Variable": "X2",
    "Wert": werte_kumuliert[1]
})
X3 = pd.DataFrame({
    "Minuten": np.arange(1, 101),
    "Variable": "X3",
    "Wert": werte_kumuliert[2]
})
# Daten zusammenführen
df_zeitreihe = pd.concat([X1, X2, X3], ignore_index=True)

np.random.seed(42); n = 100
true = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
pred = np.where(np.random.rand(n) < 0.95, true, 1 - true)
prob = np.where(pred == 1, np.random.uniform(0.7, 1.0, n), np.random.uniform(0.0, 0.3, n))
df_auc = pd.DataFrame({"true": true, "pred": pred, "prob": prob})

# ---------------- Demo ml_summary --------------------------------------------
print("\n===== Demo ml_summary =====")
# Numerische Variable
summary = ml_summary(flug["dep_delay"], 
                     titel="Verspätung beim Abflug", 
                     einheit="Minuten")
# Kategoriale Variable vollständig oder nur Top-N mit/ohne Prozent
summary, freq = ml_summary(flug["carrier"], 
                           titel="Fluggesellschaft", 
                           einheit="Anzahl Flüge")
summary, freq = ml_summary(flug["carrier"], 
                           titel="Fluggesellschaft", 
                           einheit="Anzahl Flüge", 
                           top_n=3)
summary, freq = ml_summary(flug["carrier"], 
                           titel="Fluggesellschaft", 
                           einheit="Anzahl Flüge",
                           top_n=3,
                           prozent=False)

# ---------------- Demo ml_plot -----------------
print("\n===== Demo ml_plot =====")
# Plot mit vier automatisch gewählten Farbpaletten-----------------------------
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
ml_plot(df_demo, "Gruppe", kind="bar", title=f"Nominal: {ml_colour_nom}", ax=axes[0])
ml_plot(df_demo, "Wert", kind="bar", title=f"Ordinal: {ml_colour_ord}", ax=axes[1])
ml_plot(df_demo, "Stimmung", kind="bar", title=f"Spektral (>3 Werte): {ml_colour_spect}", ax=axes[2])
ml_plot(df_demo, "Bewertung", kind="bar", title="Spektral (3 Werte): Ampel", ax=axes[3])
plt.tight_layout(); plt.show()

# Plot-Gestaltung--------------------------------------------------------------
# Keine Legende, X-Achsenskalierung bestimmt
ml_plot(df_demo, "Stimmung", kind="bar")
plt.title("Bar-Plot: Kundenzufriedenheit")
plt.xlabel("Stimmung")
plt.ylabel("Anzahl")
plt.ylim(0, 6)
plt.figtext(0.95, 0.01, "Quelle:(Heesen,2025)", ha="right")
plt.figtext(0.01, 0.01, "Abb-1", ha="left")
plt.tight_layout(); plt.show()
# Mit Legende, ohne X-Beschriftung & X-Skala
#%%%% Bar-Chart mit Legende
ml_plot(df_demo, "Stimmung", kind="bar", legend=True)
plt.title("Bar-Plot: Kundenzufriedenheit")
plt.ylabel("Anzahl")
plt.xlabel("")
plt.xticks([])
plt.figtext(0.95, 0.01, "Quelle:(Heesen,2025)", ha="right")
plt.figtext(0.01, 0.01, "Abb-2", ha="left")
plt.tight_layout(); plt.show()

#%%%% Typen von Plots----------------------------------------------------------
# Histogramm
ml_plot(df=autos, column="Alter", kind="hist", 
        title="Histogramm: Absolute Häufigkeit")
plt.tight_layout(); plt.show()

ml_plot(df=autos, column="Alter", kind="hist_density", 
        title="Histogramm: Relative Häufigkeit")
plt.tight_layout(); plt.show()

# Q-Q-Plot
ml_plot(autos, column="Alter", kind="qq", 
        title="Quantile-Quantile-Plot")
plt.tight_layout(); plt.show()

# Bar-Chart
ml_plot(df=autos, column="Marke", kind="bar", 
        title="Bar-Chart: Absolute Häufigkeit je Kategorie", legend=True)
plt.tight_layout(); plt.show()

# Dodge Column-Chart
ml_plot(df=autos_anzahl, column=("Marke", "Kategorie"), kpi="Anzahl",
        title="Column-Chart nebeneinander: Absolute Häufigkeit je Kategorie", 
        kind="dodgecolumn", legend=True)
plt.tight_layout(); plt.show()

# Stack Column-Chart
ml_plot(df=autos_anzahl, column=("Marke", "Kategorie"), kpi="Anzahl",
        title="Stacked Column-Chart: Absolute Häufigkeit je Kategorie", 
        kind="stackcolumn", legend=True)
plt.tight_layout(); plt.show()

# Stack Column-Chart 100%
ml_plot(df=autos_anzahl, column=("Marke", "Kategorie"), kpi="Anzahl",
        title="100%-Stacked Column-Chart: Relative Häufigkeit je Kategorie", 
        kind="stack100column", legend=True)
plt.tight_layout(); plt.show()

# Line-Chart
ml_plot(df=df_zeitreihe, column=("Minuten", "Variable"), kpi="Wert", 
        kind="line", title="Line-Chart: df_zeitreihe")
plt.tight_layout(); plt.show()

# Scatter-Plot (einfach) und mit Gruppierung
ml_plot(df=mtcars, column=("Gewicht", "Verbrauch100km"), kind="scatter", 
        title="Scatter-Plot: Gewicht vs. Verbrauch auf 100 km")
plt.tight_layout(); plt.show()

ml_plot(df=mtcars_filtered, column=("Gewicht","Verbrauch100km","Marke"), 
        kind="scatter", 
        title="Scatter-Plot: Gewicht in kg vs. Verbrauch auf 100 km", 
        legend=True)
plt.tight_layout(); plt.show()

# Scatter-Plots nebeneinander
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ml_plot(autos, ("Kilometer","Preis"), kind="scatter", 
        title="Scatter-Plot: Kilometer in 1000 vs. Preis", ax=axes[0])
ml_plot(autos, ("Alter","Preis"), kind="scatter", 
        title="Scatter-Plot: Alter vs. Preis", ax=axes[1])
ml_plot(autos, ("PS","Preis"), kind="scatter", 
        title="Scatter-Plot: PS vs. Preis", ax=axes[2])
plt.tight_layout(); plt.show(); plt.close(fig)

# Scatter-Joint
ml_plot(df=mtcars, column=("Gewicht", "Verbrauch100km"), kind="scatterjoint", 
        title="Joint-Plot: Gewicht vs. Verbrauch auf 100 km")
plt.tight_layout(); plt.show()

# Pairplot
ml_plot(df=mtcars_filtered, 
        column=["Gewicht", "Verbrauch100km", "Zylinder","Marke"], 
        kind="pairplot", title="Pairplot: Verbrauch, Gewicht, Zylinder, Marke")
plt.tight_layout(); plt.show()

# Pairplot gruppiert
ml_plot(df=mtcars_filtered, 
        column={"vars": ["Gewicht", "Verbrauch100km", "Zylinder"], "hue": "Marke"}, 
        kind="pairplot", title="Pairplot: Verbrauch, Gewicht, Zylinder nach Marke")
plt.tight_layout(); plt.show()

# Box-Plot
ml_plot(mtcars, column="Verbrauch100km", 
        kind="box", title="Box-Plot: Verbrauch")
plt.tight_layout(); plt.show()

# Box-Plot gruppiert
ml_plot(mtcars_filtered, column=("Marke", "Verbrauch100km"), 
        kind="box", legend=True, title="Box-Plot: Verbrauch nach Marke")
plt.tight_layout(); plt.show()

ml_plot(mtcars_filtered, column=("Marke", "Gewicht"), 
        kind="box", legend=True, title="Box-Plot: Gewicht nach Marke")
plt.tight_layout(); plt.show()

# Korrelationsmatrix
ml_plot(df=autos, column=["Preis", "PS", "Alter", "Kilometer"], 
        kind="cormatrix")
plt.tight_layout(); plt.show()

# Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
ml_plot(df=df_auc, column=("true", "pred"), kind="confmat", 
        title="Konfusionsmatrix (Positiv=Kunde kauft)", ax=ax)
plt.tight_layout(); plt.show(); plt.close(fig)

# AUC Plot
fig, ax = plt.subplots(figsize=(5, 4))
ml_plot(df=df_auc, column=("true", "prob"), kind="auc", 
        title="'Kunde kauft'", ax=ax)
plt.tight_layout(); plt.show(); plt.close(fig)

