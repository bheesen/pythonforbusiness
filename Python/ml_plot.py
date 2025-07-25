import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro, skew, kurtosis, probplot, pearsonr
# Festlegung eigener Paletten je Variablentyp------------------------------------
ml_colour_nom = "Set1"
ml_colour_ord = "Blues"
ml_colour_spect = "coolwarm"
ml_colour_ampel = [sns.color_palette("RdYlGn", 9)[i] for i in [8, 4, 0]]
ml_colour_hist = sns.color_palette("Blues", 9)[6]
#%%% ml_plot
def ml_plot(df, column, kpi=None, title="", kind="bar", legend=False, ax=None):
    """
    Automatisierte Visualisierung mit typbasierter Farbpalette.

    Parameter:
    - df       : pandas.DataFrame
    - column   : str oder Tuple (x, hue) für gruppierte Plots
    - kpi      : str (numerische Variable für y-Achse bei stack/dodge)
    - title    : Haupttitel
    - kind     : bar, hist, histdensity, qq, box, stackcolumn, stack100column,
                 dodgecolumn, line, scatter, scatterjoint, pairplot, cormatrix
    - legend   : Bool, ob Legende angezeigt wird
    - ax       : Optional: matplotlib-Achse für Subplots

    Rückgabe:
    - ax       : matplotlib-Achse mit Plot
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
    if kind not in ["pairplot", "scatterjoint"] and ax is None:
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
                palette = paletten.get(palette_key, None)
                sns.lmplot(data=df, x=x, y=y, hue=hue, palette=palette,
                           height=5, aspect=1.2, markers="o", ci=95, legend=False)
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(f"{title}: {subtitle}" if subtitle else title)
                if legend:
                    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
                return None  # lmplot erzeugt eigene Figure, daher kein ax-Rückgabe
            else:
                sns.regplot(data=df, x=x, y=y, ax=ax, color=ml_colour_hist)
                common = df[[x, y]].dropna()
                r, p = pearsonr(common[x], common[y])
                subtitle = f"r = {r:.2f}, p = {p:.2f}"
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
    else:        
        raise ValueError(f"Fehler: Unbekannter Abbildungstyp '{kind}'.")
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    if legend and hue:
        ax.legend(title=hue, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return ax
