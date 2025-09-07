# streamlit run C:\Users\bernd\Documents\A-Python\EigeneModuleDev\heesen-streamlit\tuning-comparison.py

# streamlit_app_searchcv_compare_plus.py
# GridSearchCV vs RandomizedSearchCV – erweiterter, didaktischer Vergleich
# Features:
# - Modelle: LogisticRegression, SVC (RBF), RandomForest, optional XGBoost (falls installiert)
# - Datensätze: Iris, Wine, Breast Cancer, synthetische 2D (moons, blobs)
# - Visuals: Grid-Heatmap (2D-Grid), Randomized-Scatter, Boxplots über Seeds
# - Exporte: cv_results_ als CSV (optional Excel)
# - Eindeutige Streamlit-Keys

import io, time, numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, train_test_split,
                                     StratifiedKFold, validation_curve)
from sklearn.metrics import accuracy_score, roc_auc_score  # <- kein f1_macro import!
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_blobs
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, randint, uniform

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ---- Eigene Module---------------------------------------

try:
    from ml_summary import ml_summary  # liefert schöne Tabellen
except Exception:
    ml_summary = None

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew, kurtosis, probplot, pearsonr

# Festlegung eigener Farbpaletten je Variablentyp------------------------------
ml_colour_nom = "Set1"
ml_colour_ord = "Blues"
ml_colour_spect = "coolwarm"
ml_colour_ampel = [sns.color_palette("RdYlGn", 9)[i] for i in [8, 4, 0]]
ml_colour_hist = sns.color_palette("Blues", 9)[6]

# -----------------------------------------------------------------------------
# Funktion zur Visualisierung
# -----------------------------------------------------------------------------
def ml_plot(df, column, kpi=None, title="", kind="bar", legend=False, ax=None):
    """
    Diese Funktion erzeugt verschiedene Arten von Diagrammen (bspw. Histogramme,
    Boxplots, Korrelationsmatrizen, ROC-Kurven usw.) basierend auf den Spalten eines
    übergebenen Pandas DataFrames. Farbpaletten werden automatisch in Abhängigkeit
    vom Variablentyp gewählt.

    Parameter:
    ----------
    df : pandas.DataFrame
        Datensatz, der geplottet werden soll. 
        - Für "auc", "confmat": df muss folgende Spalten enthalten:
          'true' (Ist-Werte) und 
          'pred/prob' (Vorhersagewerte oder Wahrscheinlichkeit) 
    column : str, tuple, list oder dict
        Spalte(n), die geplottet werden sollen. Abhängig vom gewählten Diagrammtyp:
        - "bar", "hist", "hist_density", "qq": ein Spaltenname (str)
        - "stackcolumn", "stack100column", "dodgecolumn", "line": Tupel (x, hue)
        - "scatter", "scatterjoint": Tupel (x, y) oder (x, y, hue)
        - "pairplot": Liste von Variablennamen oder dict mit {"vars": [...], "hue": "..."}
        - "box": entweder Spaltenname (str) oder Tupel (x, y)
        - "cormatrix": Liste (erste Spalte = Zielvariable, Rest = Prädiktoren)
        - "auc", "confmat": Tupel mit den relevanten Spalten
    kpi : str, optional
        Y-Variable für "stackcolumn", "stack100column", "dodgecolumn" und "line".
    title : str, optional
        Titel des Diagramms. Bei Bedarf werden automatisch statistische Kennwerte
        (z. B. Normalitätstest p-Wert, Schiefe, Kurtosis) ergänzt.
    kind : str, default='bar'
        Diagrammtyp, einer der:
        ["bar", "hist", "hist_density", "qq", "box", "stackcolumn", "stack100column",
         "dodgecolumn", "line", "scatter", "scatterjoint", "pairplot", "cormatrix",
         "auc", "confmat"]
    legend : bool, default=False
        Ob eine Legende angezeigt werden soll (falls sinnvoll für den Diagrammtyp).

    ax : matplotlib.axes.Axes, optional
        Vorhandene Achse, auf die geplottet werden soll. Falls None, wird eine neue Figur erstellt.

    Rückgabe:
    ---------
    ax : matplotlib.axes.Axes oder None
        Gibt die verwendete Achse zurück, sofern kein spezielles Plot-Objekt erstellt wird
        (z. B. pairplot, lmplot, jointplot erzeugen eigene Figures und geben None zurück).

    Hinweise:
    ---------
    - Farbpaletten werden automatisch in Abhängigkeit vom Variablentyp ausgewählt:
        * Numerisch: Blauabstufungen
        * Nominal: Set1
        * Ordinal: Spektralfarbskala (bei geordneten Kategorien)
        * Ampel (3 Ausprägungen): Rot-Gelb-Grün
    - Bei numerischen Verteilungen ('hist', 'qq') werden Mittelwert, Median und Modus
      im Plot angezeigt. Zudem werden Normalitätstest (Shapiro-Wilk), Schiefe und Kurtosis
      berechnet und als Untertitel ergänzt.
    - Für 'cormatrix' werden Signifikanzsterne basierend auf p-Werten (* p<0.05, ** p<0.01, *** p<0.001)
      in der Heatmap angezeigt.
    - Bei 'auc' wird ROC-Kurve mit AUC und Accuracy angezeigt.
    - Bei 'confmat' wird eine Konfusionsmatrix farblich hervorgehoben (Korrekt vs. Fehler).

    Raises:
    -------
    ValueError:
        Wenn ein nicht unterstützter Diagrammtyp oder ein fehlerhaftes column-Format
        angegeben wird.
    """

    df_name = next((name for name in globals() if globals()[name] is df), "df")
    subtitle = False
    hue = None
    paletten = {}

    def is_numeric(col): return pd.api.types.is_numeric_dtype(col)
    def is_spektral(col): return isinstance(col.dtype, pd.CategoricalDtype) and col.cat.ordered
    def is_ampel_case(col): return is_spektral(col) and col.nunique() == 3

    for col in df.columns:
        col_data = df[col]
        key = f"palette_{df_name}_{col.lower()}"
        if is_numeric(col_data):
            n = col_data.nunique()
            paletten[key] = sns.color_palette(ml_colour_ord, n_colors=n)
        elif is_ampel_case(col_data):
            paletten[key] = ml_colour_ampel
        elif is_spektral(col_data):
            n = col_data.nunique()
            paletten[key] = sns.color_palette(ml_colour_spect, n_colors=n)
        else:
            n = col_data.nunique()
            paletten[key] = sns.color_palette(ml_colour_nom, n_colors=n)
    if kind not in ["pairplot", "scatterjoint", "scatter"] and ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    if kind in ["hist", "hist_density"]:
        x = column
        hue = None
        n_bins = df[x].nunique() if is_numeric(df[x]) else 20
        if kind == "hist":
            sns.histplot(data=df, x=x, bins=n_bins, kde=False,
                         color=ml_colour_hist, edgecolor="white", ax=ax)
            ax.set_ylabel("Anzahl")
        elif kind == "hist_density":
            sns.histplot(data=df, x=x, bins=n_bins, stat="density",
                         kde=False, color=ml_colour_hist, edgecolor="white", ax=ax)
            ax.set_ylabel("Anteil")
        ax.set_xlabel(x)
        if is_numeric(df[x]):
            val = df[x].dropna()
            mean = val.mean()
            median = val.median()
            modus = val.mode().iloc[0]
            ax.axvline(mean, color="red", linestyle="--", label="Mittelwert")
            ax.axvline(median, color="blue", linestyle="-.", label="Median")
            ax.axvline(modus, color="green", linestyle=":", label="Modus")
            ax.legend()
        try:
            pval = shapiro(val)[1]
            skew_val = skew(val)
            kurt = kurtosis(val, fisher=True)
            subtitle = f"Normal: p={pval:.2f}, Schiefe: {skew_val:.2f}, Kurtosis: {kurt:.2f}"
        except Exception as e:
            subtitle = "Statistische Kennwerte nicht verfügbar"
    elif kind == "qq":
        x = column
        val = df[x].dropna()
        probplot(val, dist="norm", plot=ax)
        pval = shapiro(val)[1]
        skew_val = skew(val)
        kurt = kurtosis(val, fisher=True)
        subtitle = f"Normal: p={pval:.2f}, Schiefe: {skew_val:.2f}, Kurtosis: {kurt:.2f}"
        ax.set_xlabel("Theoretische Quantile (Normalverteilung)")
        ax.set_ylabel("Beobachtete Quantile")
    elif kind == "bar":
        hue = column
        key = f"palette_{df_name}_{column.lower()}"
        palette = paletten.get(key, None)
        sns.countplot(data=df, x=column, hue=column, legend=legend, palette=palette, ax=ax)
    elif kind in ["stackcolumn", "dodgecolumn", "stack100column"]:
        x, hue = column
        y = kpi
        key = f"palette_{df_name}_{hue.lower()}"
        palette = paletten.get(key, None)
        if kind in ["stackcolumn", "stack100column"]:
            df_pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc='sum', fill_value=0)
            if kind == "stack100column":
                df_pivot = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100  # Prozentwerte
            bottom = None
            for idx, level in enumerate(df_pivot.columns):
                values = df_pivot[level].values
                ax.bar(df_pivot.index, values, bottom=bottom, label=level,
                       color=palette[idx] if palette else None)
                bottom = values if bottom is None else bottom + values
        else:  # dodgecolumn
            sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, dodge=True, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if kind == "stack100column":
           ax.set_ylim(0, 100)
           ax.set_ylabel("Anteil in %")
    elif kind == "line":
        if isinstance(column, tuple) and len(column) == 2:
            x, hue = column
            y = kpi
        else:
            raise ValueError("Für 'line' muss column als Tupel (x, hue) übergeben werden.")
        key = f"palette_{df_name}_{hue.lower()}"
        palette = paletten.get(key, None)
        sns.lineplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    elif kind == "scatter":
        # Erwartet column als (x, y) oder (x, y, hue)
        if isinstance(column, tuple) and (len(column) == 2 or len(column) == 3):
            x, y = column[0], column[1]
            hue = column[2] if len(column) == 3 else None
            if hue:
                palette_key = f"palette_{df_name}_{hue.lower()}"
                palette = paletten.get(palette_key, sns.color_palette("Set1"))
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 5))
                # Punkte je Gruppe zeichnen
                sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax)
                # Regression pro Gruppe
                for grp in df[hue].unique():
                    grp_data = df[df[hue] == grp]
                    sns.regplot(data=grp_data, x=x, y=y, ax=ax,
                                scatter=False, color=palette[df[hue].unique().tolist().index(grp)])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                if legend:
                    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
                ax.set_title(title)
            else:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 5))
                sns.regplot(data=df, x=x, y=y, ax=ax, color=ml_colour_hist)
                common = df[[x, y]].dropna()
                r, p = pearsonr(common[x], common[y])
                subtitle = f"r = {r:.2f}, p = {p:.2f}"
                ax.set_title(f"{title}\n{subtitle}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
        else:
            raise ValueError("Für 'scatter' muss column als Tupel (x, y) übergeben werden.")
    elif kind == "scatterjoint":
        # Erwartet column als (x, y) oder (x, y, hue)
        if isinstance(column, tuple) and (len(column) == 2 or len(column) == 3):
            x, y = column[0], column[1]
            hue = column[2] if len(column) == 3 else None
            joint_kws = dict(kind="reg", x=x, y=y, data=df, height=6, ratio=4)
            palette = None
            g = sns.jointplot(**joint_kws)
            g.set_axis_labels(x, y)
            common = df[[x, y]].dropna()
            r, p = pearsonr(common[x], common[y])
            plt.suptitle(f"{title}\nr = {r:.2f}, p = {p:.2f}", y=1.02, fontsize=12)
            plt.tight_layout()
            return None  # jointplot erstellt eigene Figure
        else:
            raise ValueError("Für 'scatterjoint' muss column ein Tupel (x, y) oder (x, y, hue) sein.")        
    elif kind == "pairplot":
        # Erwartet: column als Liste der zu vergleichenden Variablen
        if isinstance(column, (list, tuple)):
            hue = None
            vars = column
        elif isinstance(column, dict):
            vars = column.get("vars")
            hue = column.get("hue")
        else:
            raise ValueError("Für 'pairplot' muss column eine Liste oder ein Dict mit 'vars' und optional 'hue' sein.")
        pairplot_kwargs = {"data": df, "vars": vars}
        if hue:
            key = f"palette_{df_name}_{hue.lower()}"
            pairplot_kwargs["hue"] = hue
            pairplot_kwargs["palette"] = paletten.get(key, None)
        g = sns.pairplot(**pairplot_kwargs)
        g.fig.suptitle(title, y=1.02)
        if hue:
            handles, labels = g._legend_data.values(), g._legend_data.keys()
            g._legend.remove()
            g.fig.legend(handles, labels, title=hue, loc='center right', bbox_to_anchor=(1.12, 0.5))
        return None  # pairplot erstellt eigene Figureelse:
    elif kind == "box":
        flierprops = dict(marker='o', markerfacecolor='red', markersize=6,
                  linestyle='none', markeredgecolor='black')
        if isinstance(column, tuple):  # Gruppierter Boxplot: (x, y)
            x, y = column
            key = f"palette_{df_name}_{x.lower()}"
            palette = paletten.get(key, None)
            # Sortiere die x-Gruppen nach Median von y
            grouped = df[[x, y]].dropna().groupby(x)[y].median().sort_values()
            order = grouped.index.tolist()
            # Palette entsprechend der sortierten Gruppen anpassen
            color_dict = dict(zip(order, palette[:len(order)]))  # sicherstellen, dass Länge passt
            sns.boxplot(data=df, x=x, y=y, hue=x, order=order, palette=color_dict,
                        flierprops=flierprops, ax=ax, dodge=False)
            # Linie für Gesamtmedian aller Daten
            y_median_global = df[y].median()
            if y_median_global >= 100:
                median_label = f"Median: {y_median_global:.0f}"
            else:
                median_label = f"Median: {y_median_global:.1f}"
            ax.axhline(y=y_median_global, color="blue", linestyle="--", linewidth=1)
            # Beschriftung der Linie
            ax.text(len(order) - 0.4, y_median_global, median_label,
                    va='center', ha='left', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            if legend:
                from matplotlib.patches import Patch
                handles = [Patch(color=color_dict[cat], label=str(cat)) for cat in order]
                ax.legend(handles=handles, title=x, loc="upper left", bbox_to_anchor=(1.01, 1))
        else:  # Einfacher Boxplot: nur y
            y = column
            val = df[y].dropna()
            median = val.median()
            q1 = val.quantile(0.25)
            q3 = val.quantile(0.75)
            iqr = q3 - q1
            whisker_min = val[val >= (q1 - 1.5 * iqr)].min()
            whisker_max = val[val <= (q3 + 1.5 * iqr)].max()
            sns.boxplot(data=df, y=y, flierprops=flierprops,
                        ax=ax, color=ml_colour_hist)
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_ylabel(y)
            # Annotationen
            stats = {
                "Min (Whisker)": whisker_min,
                "Q1": q1,
                "Median": median,
                "Q3": q3,
                "Max (Whisker)": whisker_max
            }
            for label, val in stats.items():
                ax.text(0.6, val, f"{label}: {val:.1f}",
                        ha='left', va='center', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'),
                        transform=ax.get_yaxis_transform())
    elif kind == "cormatrix":
        # column als Liste: erste Variable = Zielvariable
        if not isinstance(column, (list, tuple)) or len(column) < 2:
            raise ValueError("Für 'cormatrix' muss column eine Liste mit mindestens zwei Spalten sein (erste = Zielvariable).")
        data = df[column].dropna()
        y = column[0]
        predictors = column[1:]
        corr = data.corr(method="pearson")
        # p-Werte & Sterne
        pval = pd.DataFrame(np.ones_like(corr), columns=corr.columns, index=corr.index)
        stars = pd.DataFrame("", index=corr.index, columns=corr.columns)
        for i in corr.columns:
            for j in corr.columns:
                if i != j:
                    r, p = pearsonr(data[i], data[j])
                    pval.loc[i, j] = p
                    if p < 0.001:
                        stars.loc[i, j] = "***"
                    elif p < 0.01:
                        stars.loc[i, j] = "**"
                    elif p < 0.05:
                        stars.loc[i, j] = "*"
        # Sortieren nach r(y,x)
        sorted_predictors = corr[y].drop(y).sort_values(ascending=False).index.tolist()
        sort_order = [y] + sorted_predictors
        # Trimmen: entferne oberste Zeile und letzte Spalte
        corr = corr.loc[sort_order, sort_order]
        annot = corr.round(2).astype(str) + stars.loc[sort_order, sort_order]
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_trim = corr.iloc[1:, :-1]
        annot_trim = annot.iloc[1:, :-1]
        mask_trim = mask[1:, :-1]
        # Plot erzeugen
        sns.heatmap(corr_trim, mask=mask_trim, annot=annot_trim, fmt="", cmap=ml_colour_spect,
                    vmin=-1, vmax=1, linewidths=0.5, square=True,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f"Korrelationsmatrix mit Zielvariable: {y}\n"
                     "* p < 0.05   ** p < 0.01   *** p < 0.001", fontsize=12, pad=12)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        return ax
    elif kind == "auc":
        from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
        x, y = df[column[0]], df[column[1]]
        fpr, tpr, _ = roc_curve(x, y)
        auc = roc_auc_score(x, y)
        acc = accuracy_score(x, (y > 0.5).astype(int))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Falsch Positive Rate")
        ax.set_ylabel("Richtig Positive Rate")
        ax.set_title(f"{title} ROC-Kurve: AUC = {auc:.2f}, Accuracy = {acc:.2f}")
        return ax
    elif kind == "confmat":
        from sklearn.metrics import confusion_matrix
        from matplotlib.colors import ListedColormap
        true, pred = df[column[0]], df[column[1]]
        cm = confusion_matrix(true, pred, labels=[1, 0])
        # Farbzuordnung: ml_colour_ampel[1] = grün, ml_colour_ampel[3] = rot
        farben = [ml_colour_ampel[2], ml_colour_ampel[0]]
        cmap = ListedColormap(farben)
        highlight = np.zeros_like(cm, dtype=int)
        np.fill_diagonal(highlight, 1)
        ax = sns.heatmap(highlight, annot=cm, fmt="d", cmap=cmap, cbar=False,
                         xticklabels=["Positiv", "Negativ"],
                         yticklabels=["Positiv", "Negativ"],
                         linewidths=1, linecolor="black", ax=ax)
        if title == "":
            ax.set_title("Konfusionsmatrix", pad=20)
        else:
            ax.set_title(title, pad=20)
        ax.set_xlabel("Vorhergesagte Klasse")
        ax.set_ylabel("Tatsächliche Klasse")
        return ax
    else:        
        raise ValueError(f"Fehler: Unbekannter Abbildungstyp '{kind}'.")
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    if legend and hue:
        ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return ax

# ---------------- Hinweis --------------------------------
st.set_page_config(page_title="Tuning Hyperparameter", layout="wide")
st.title("GridSearchCV vs RandomizedSearchCV")
st.markdown(
    "Diese App vergleicht **GridSearchCV** (Raster) und **RandomizedSearchCV** (Stichprobe) – *Suchraum, Budget, Metrik, Laufzeit, Stabilität*. "
    "Diese App wurde von [Prof. Heesen](http://www.profheesen.de) ergänzend zu dem Buch "
    "[Künstliche Intelligenz im Business](https://www.amazon.de/K%C3%BCnstliche-Intelligenz-Business-Erstellung-Anwendungen/dp/3658495448) erstellt."
)

# ---------------- Sidebar: Daten & Modell ----------------


with st.sidebar:
    st.header("Daten & Modell")
    ds_kind = st.selectbox("Datensatz", ["iris", "wine", "breast_cancer", "moons (2D)", "blobs (2D)"],
                           index=0, key="ds_kind")
    model_opts = ["LogisticRegression", "SVC (RBF)", "RandomForest"]
    if XGB_AVAILABLE:
        model_opts.append("XGBoost")
    model_kind = st.selectbox("Modell", model_opts, index=1 if "SVC (RBF)" in model_opts else 0, key="model_kind")

    st.header("CV & Metrik")
    cv_folds = st.slider("CV-Folds", 3, 10, 5, 1, key="cv_folds")
    scoring_name = st.selectbox("Scoring", ["accuracy", "f1_score", "roc_auc_ovr"], index=0, key="scoring_name")

    st.header("Suchräume & Budget")
    # Modell-spezifische Gitter
    if model_kind == "SVC (RBF)":
        st.caption("Grid: C × γ; Randomized: Log-Uniform-Verteilungen.")
        grid_gamma = st.multiselect("Grid γ", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1, 3],
                                    default=[1e-3, 1e-2, 1e-1, 1], key="grid_gamma")
        grid_C     = st.multiselect("Grid C", [0.01, 0.1, 1, 3, 10, 30, 100],
                                    default=[0.1, 1, 10], key="grid_C")
    elif model_kind == "RandomForest":
        st.caption("Grid: n_estimators × max_depth; Randomized: mehrere int/float-Verteilungen.")
        grid_n_estimators = st.multiselect("Grid n_estimators", [50, 100, 200, 400], default=[100, 200], key="grid_n_estimators")
        grid_max_depth    = st.multiselect("Grid max_depth", [None, 4, 8, 16, 32], default=[None, 8, 16], key="grid_max_depth")
    elif model_kind == "LogisticRegression":
        st.caption("Grid: C × penalty/solver (nur lbfgs/l2), Randomized: Log-Uniform über C.")
        grid_C_logreg = st.multiselect("Grid C", [0.01, 0.1, 1, 3, 10, 30, 100], default=[0.1, 1, 10], key="grid_C_logreg")
    else:  # XGBoost
        st.caption("Grid: n_estimators × max_depth; Randomized: mehrere Hyperparameter-Verteilungen.")
        grid_n_estimators_xgb = st.multiselect("Grid n_estimators", [100, 200, 400], default=[100, 200], key="grid_n_estimators_xgb")
        grid_max_depth_xgb    = st.multiselect("Grid max_depth", [2, 4, 6, 8, 10], default=[3, 6], key="grid_max_depth_xgb")

    n_iter = st.slider("RandomizedSearch n_iter", 5, 300, 25, 5, key="n_iter")
    seed   = st.number_input("Seed (Basis)", 0, 100000, 42, 1, key="seed")

    st.header("Optionen")
    test_size = st.slider("Testgröße", 0.1, 0.5, 0.25, 0.05, key="test_size")
    show_val_curves = st.checkbox("Validierungskurven (nur SVC)", value=False, key="show_val_curves")

    st.header("Stabilität über Seeds")
    multi_seeds = st.checkbox("Mehrere Seeds vergleichen", value=False, key="multi_seeds")
    seeds_n = st.slider("Anzahl Seeds", 2, 15, 5, 1, key="seeds_n") if multi_seeds else 0

    run_btn = st.button("Vergleich starten", key="run_btn")

# ---------------- Daten laden/erzeugen ----------------
def load_dataset(name, seed=42, n=600):
    if name == "iris":
        Xy = load_iris(as_frame=True)
        return Xy.data, Xy.target                      # nur 2 Rückgaben
    if name == "wine":
        Xy = load_wine(as_frame=True)
        return Xy.data, Xy.target
    if name == "breast_cancer":
        Xy = load_breast_cancer(as_frame=True)
        return Xy.data, Xy.target
    if name == "moons (2D)":
        X, y = make_moons(n_samples=n, noise=0.25, random_state=seed)
        return pd.DataFrame(X, columns=["x1","x2"]), pd.Series(y, name="y")
    if name == "blobs (2D)":
        X, y = make_blobs(n_samples=n, centers=3, random_state=seed, cluster_std=2.0)
        return pd.DataFrame(X, columns=["x1","x2"]), pd.Series(y, name="y")
    raise ValueError("Unbekannter Datensatz")

X_df, y_s = load_dataset(ds_kind, seed=seed)
X = X_df.values; y = y_s.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=seed, stratify=y)

# Scorer: einfach Strings nutzen (so gibt es keinen Import-Fehler)
scoring = scoring_name  # "accuracy", "f1_macro" oder "roc_auc_ovr"

# --------- Pipelines & Suchräume ---------
def spaces_logreg():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))])
    return pipe, {"clf__C": grid_C_logreg if len(grid_C_logreg) > 0 else [0.1, 1, 10]}, {"clf__C": loguniform(1e-3, 1e3)}

def spaces_svc():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=seed))])
    pg = {"clf__C": grid_C if len(grid_C) > 0 else [0.1, 1, 10],
          "clf__gamma": grid_gamma if len(grid_gamma) > 0 else [1e-3, 1e-2, 1e-1, 1]}
    pdist = {"clf__C": loguniform(1e-3, 1e3), "clf__gamma": loguniform(1e-4, 1e1)}
    return pipe, pg, pdist

def spaces_rf():
    pipe = Pipeline([("clf", RandomForestClassifier(random_state=seed, n_jobs=-1))])
    pg = {"clf__n_estimators": grid_n_estimators if len(grid_n_estimators) > 0 else [100, 200],
          "clf__max_depth": grid_max_depth if len(grid_max_depth) > 0 else [None, 8, 16]}
    pdist = {"clf__n_estimators": randint(50, 600),
             "clf__max_depth": randint(2, 40),
             "clf__min_samples_split": randint(2, 20),
             "clf__min_samples_leaf": randint(1, 10),
             "clf__max_features": uniform(0.3, 0.7),
             "clf__bootstrap": [True, False]}
    return pipe, pg, pdist

if model_kind == "LogisticRegression":
    pipe, param_grid, param_dist = spaces_logreg()
elif model_kind == "SVC (RBF)":
    pipe, param_grid, param_dist = spaces_svc()
else:
    pipe, param_grid, param_dist = spaces_rf()

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

# --------- Funktionen: Suchen, Visualisieren, Bewerten ---------
def run_searches(rnd_seed):
    t0 = time.perf_counter()
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    gs.fit(X_train, y_train); t_grid = time.perf_counter() - t0

    t0 = time.perf_counter()
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                            scoring=scoring, random_state=rnd_seed, n_jobs=-1, return_train_score=True)
    rs.fit(X_train, y_train); t_rand = time.perf_counter() - t0
    return gs, rs, t_grid, t_rand

def evaluate_generalization(estimator):
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = None
    if scoring == "roc_auc_ovr" and len(np.unique(y)) > 2:
        try:
            s = estimator.decision_function(X_test)
        except Exception:
            try: s = estimator.predict_proba(X_test)
            except Exception: s = None
        if s is not None:
            try: auc = roc_auc_score(y_test, s, multi_class="ovr")
            except Exception: auc = None
    elif len(np.unique(y)) == 2:
        try:
            s = estimator.decision_function(X_test)
            if getattr(s, "ndim", 1) > 1: s = s[:, 1]
        except Exception:
            try: s = estimator.predict_proba(X_test)[:, 1]
            except Exception: s = None
        if s is not None:
            try: auc = roc_auc_score(y_test, s)
            except Exception: auc = None
    return acc, auc

def plot_grid_heatmap(cvres, title):
    params = cvres["params"]
    keys = sorted({k for p in params for k in p.keys()})
    if len(keys) != 2: st.info("Heatmap benötigt genau 2 Gitter-Parameter."); return
    X_vals = sorted({p[keys[0]] for p in params}); Y_vals = sorted({p[keys[1]] for p in params})
    if len(X_vals) * len(Y_vals) != len(params): st.info("Gitter ist nicht rechteckig."); return
    Z = np.zeros((len(Y_vals), len(X_vals)))
    for i,yv in enumerate(Y_vals):
        for j,xv in enumerate(X_vals):
            idx = [k for k,p in enumerate(params) if p[keys[0]]==xv and p[keys[1]]==yv][0]
            Z[i,j] = cvres["mean_test_score"][idx]
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(Z, origin="lower", aspect="auto")
    ax.set_xticks(range(len(X_vals))); ax.set_xticklabels(X_vals, rotation=45)
    ax.set_yticks(range(len(Y_vals))); ax.set_yticklabels(Y_vals)
    ax.set_xlabel(keys[0]); ax.set_ylabel(keys[1]); ax.set_title(f"{title} ({scoring_name})")
    fig.colorbar(im, ax=ax, shrink=0.8, label=scoring_name)
    st.pyplot(fig)

def plot_random_scatter(cvres, title):
    params = cvres["params"]; scores = cvres["mean_test_score"]
    keys = [k for k in ["clf__C","clf__gamma","clf__max_depth","clf__n_estimators"] if any(k in p for p in params)][:2]
    if len(keys) < 2: st.info("Scatter benötigt zwei kontinuierliche Parameter."); return
    xs,ys=[],[]; 
    for p in params:
        if keys[0] in p and keys[1] in p: xs.append(p[keys[0]]); ys.append(p[keys[1]])
    if len(xs)==0: st.info("Keine passenden Parameter-Kombinationen."); return
    fig, ax = plt.subplots(figsize=(6,4))
    sca = ax.scatter(xs, ys, c=scores, s=40); 
    if min(xs)>0: ax.set_xscale("log"); 
    if min(ys)>0: ax.set_yscale("log")
    ax.set_xlabel(keys[0]); ax.set_ylabel(keys[1]); ax.set_title(f"{title} ({scoring_name})")
    fig.colorbar(sca, ax=ax, shrink=0.8, label=scoring_name); st.pyplot(fig)

# --------- TABS ---------
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Datenexploration", "Einzeldurchlauf", "Visualisierung", "Stabilität", "Export"])

# ===== TAB 0: Datenexploration =====
with tab0:
    st.subheader("Explorative Analyse")
    st.write(f"Form: X = {X_df.shape}, y = {y_s.shape}, Klassen = {np.unique(y).size}")
    st.dataframe(X_df.head())

    if ml_summary is not None:
        st.markdown("**Statistische Zusammenfassungen (ml_summary)**")
        # Für jede numerische Spalte eine kompakte Zusammenfassung
        for col in X_df.columns:
            st.markdown(f"*Feature:* **{col}**")
            try:
                summary = ml_summary(X_df[col], titel=None, einheit="")
                st.dataframe(summary)
            except Exception:
                st.write("Zusammenfassung nicht verfügbar.")
    else:
        st.info("`ml_summary` nicht gefunden – Modul nicht geladen.")

    st.markdown("**Visualisierungen (ml_plot)**")
    # Beispiel: Histogramm/Boxplot für die ersten zwei Features
    for col in list(X_df.columns)[:2]:
        try:
            ax = ml_plot(X_df, col, kind="hist", title=f"Histogramm – {col}")
            st.pyplot(ax.figure); plt.close(ax.figure)
            ax = ml_plot(X_df, col, kind="box", title=f"Boxplot – {col}")
            st.pyplot(ax.figure); plt.close(ax.figure)
        except Exception:
            st.write(f"Plot für {col} nicht verfügbar.")

# ===== RUN =====
if run_btn:
    gs, rs, t_grid, t_rand = run_searches(seed)

    with tab1:
        st.subheader("Bestwerte, Laufzeiten, Generalisierung")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**GridSearchCV**")
            st.json({"best_params": gs.best_params_, "best_score": float(gs.best_score_)})
            acc, auc = evaluate_generalization(gs.best_estimator_)
            st.write(f"Test-Accuracy: **{acc:.4f}**" + (f", ROC-AUC: **{auc:.4f}**" if auc is not None else ""))
            st.caption(f"Laufzeit: **{t_grid:.2f} s**")
        with colB:
            st.markdown("**RandomizedSearchCV**")
            st.json({"best_params": rs.best_params_, "best_score": float(rs.best_score_)})
            acc, auc = evaluate_generalization(rs.best_estimator_)
            st.write(f"Test-Accuracy: **{acc:.4f}**" + (f", ROC-AUC: **{auc:.4f}**" if auc is not None else ""))
            st.caption(f"Laufzeit: **{t_rand:.2f} s**")

    with tab2:
        st.subheader("Suchraum-Visualisierung")
        st.markdown("**GridSearchCV – Heatmap**")
        try: plot_grid_heatmap(gs.cv_results_, "GridSearchCV – Heatmap (Mean CV-Score)")
        except Exception as e: st.info(f"Heatmap nicht verfügbar: {e}")
        st.markdown("**RandomizedSearchCV – Scatter**")
        try: plot_random_scatter(rs.cv_results_, "RandomizedSearchCV – Scatter (Mean CV-Score)")
        except Exception as e: st.info(f"Scatter nicht verfügbar: {e}")

    with tab3:
        st.subheader("Stabilität über Seeds")
        if not multi_seeds:
            st.info("Aktiviere „Mehrere Seeds vergleichen“ in der Sidebar.")
        else:
            seeds = [seed + i for i in range(seeds_n)]
            rows=[]
            for s in seeds:
                gs_s, rs_s, _, _ = run_searches(s)
                rows += [{"seed":s,"method":"Grid","cv_best":gs_s.best_score_,
                          "test_acc":evaluate_generalization(gs_s.best_estimator_)[0]},
                         {"seed":s,"method":"Randomized","cv_best":rs_s.best_score_,
                          "test_acc":evaluate_generalization(rs_s.best_estimator_)[0]}]
            df = pd.DataFrame(rows); st.dataframe(df)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.boxplot([df[df.method=="Grid"]["cv_best"], df[df.method=="Randomized"]["cv_best"]],
                       labels=["Grid CV-Best","Random CV-Best"])
            ax.set_title("CV-Bestscore – Verteilung über Seeds"); st.pyplot(fig)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.boxplot([df[df.method=="Grid"]["test_acc"], df[df.method=="Randomized"]["test_acc"]],
                       labels=["Grid Test-Acc","Random Test-Acc"])
            ax.set_title("Test-Accuracy – Verteilung über Seeds"); st.pyplot(fig)

    with tab4:
        st.subheader("Export der CV-Ergebnisse")
        df_grid = pd.DataFrame(gs.cv_results_); df_rand = pd.DataFrame(rs.cv_results_)
        st.download_button("GridSearchCV – cv_results_.csv",
                           data=df_grid.to_csv(index=False).encode("utf-8"),
                           file_name="grid_cv_results.csv", mime="text/csv")
        st.download_button("RandomizedSearchCV – cv_results_.csv",
                           data=df_rand.to_csv(index=False).encode("utf-8"),
                           file_name="randomized_cv_results.csv", mime="text/csv")
else:
    st.info("Parameter wählen → **„Vergleich starten“**. Die Datenexploration steht bereits im ersten Tab zur Verfügung.")
