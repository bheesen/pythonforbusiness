# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\regression_underfit_overfit.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import streamlit as st
import io

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="Overfit vs Underfit", layout="wide")
st.title("Vergleich von Modellen (Overfit vs Underfit)")
st.markdown(
    "Diese App vergleicht **Lineare Regression** und **Polynomiale Regression n-ten Grades** basierend auf dem *RMSE*. "
    "Diese App wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------

with st.sidebar:
    # Streamlit UI
    st.header("Parameter")

    n_extra = st.slider("Zusätzliche Datenpunkte", 0, 6, 0)
    degree1 = st.slider("Polynomgrad Modell 1", 2, 10, 4)
    degree2 = st.slider("Polynomgrad Modell 2", 2, 20, 10)

# ---------------- Daten laden/erzeugen ----------------

# Datenbasis
x_start = np.array([1, 2, 3, 4, 5, 6, 7])
y_start = np.array([0.5, 0.8, 0.2, -0.4, -0.8, -0.6, 0.0])
x_extra = np.array([8, 9, 10, 11, 12, 13])
y_extra = np.array([0.6, 1.0, 0.5, -0.2, -0.6, -1.0])
x = np.concatenate([x_start, x_extra[:n_extra]])
y = np.concatenate([y_start, y_extra[:n_extra]])
x_pred = np.linspace(0, 14, 300).reshape(-1, 1)

# Modellanpassung
def fit_model(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    return model

# Modelle und Fehlerberechnung
x_reshaped = x.reshape(-1, 1)
lin_model = LinearRegression().fit(x_reshaped, y)
poly_model_1 = fit_model(x, y, degree1)
poly_model_2 = fit_model(x, y, degree2)

rmse_lin = np.sqrt(mean_squared_error(y, lin_model.predict(x_reshaped)))
rmse_poly1 = np.sqrt(mean_squared_error(y, poly_model_1.predict(x_reshaped)))
rmse_poly2 = np.sqrt(mean_squared_error(y, poly_model_2.predict(x_reshaped)))

# Visualisierung
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 14)
ax.set_ylim(-2.5, 2.5)
ax.scatter(x, y, color='black', label='Datenpunkte')
ax.plot(x_pred, lin_model.predict(x_pred), label=f'Linear (RMSE={rmse_lin:.2f})')
ax.plot(x_pred, poly_model_1.predict(x_pred), label=f'Poly Grad {degree1} (RMSE={rmse_poly1:.2f})')
ax.plot(x_pred, poly_model_2.predict(x_pred), label=f'Poly Grad {degree2} (RMSE={rmse_poly2:.2f})')

# Titel setzen
ax.set_title(f"Modellvergleich mit {len(x)} Punkten")
ax.legend()
st.pyplot(fig)

# Download-Link
buf = io.BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.download_button(label="📥 Plot herunterladen als PNG",
                   data=buf,
                   file_name=f"regression_plot_n{len(x)}.png",
                   mime="image/png")
