# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 15:07:41 2025
@author: Bernd Heesen
@web:    www.profheesen.de
"""

import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# Funktion zur Anzeige von Lage- und Streuungsmaßen
# -----------------------------------------------------------------------------
def ml_summary(variable, titel="Variable", einheit="", prozent=True, top_n=None):
    """
    Erstellt eine Zusammenfassung (Summary) einer Variablen und gibt 
    statistische Kennzahlen sowie ggf. eine Häufigkeitstabelle aus.

    Parameter
    ----------
    variable : array-like (z.B. Pandas Series, Liste, NumPy Array)
        Die zu analysierende Variable.
    
    titel : str, optional (Default = "Variable")
        Bezeichnung der Variablen, die in der Ausgabe angezeigt wird.

    einheit : str, optional (Default = "")
        Einheit der Variablen, die in der Ausgabe angezeigt wird 
        (z.B. "kg", "%", "EUR").
    
    prozent : bool, optional (Default = True)
        Wird bei kategorialen Variablen verwendet. Gibt an, ob relative 
        Häufigkeiten (in Prozent) mit ausgegeben werden sollen.
    
    top_n : int oder None, optional (Default = None)
        Begrenzung der Ausgabetabelle auf die häufigsten Kategorien.
        - None: Alle Ausprägungen werden ausgegeben.
        - int: Nur die Top-N Ausprägungen werden in der Häufigkeitstabelle angezeigt.

    Rückgabewerte
    -------------
    summary : dict
        Dictionary mit den berechneten Kennzahlen:
        - Bei numerischen Variablen: Lage- und Streuungsmaße, Schiefe, Kurtosis usw.
        - Bei kategorialen Variablen: Anzahl Werte, Anzahl gültig, Anzahl fehlend, Anzahl eindeutiger Werte.
    
    freq_table : pd.DataFrame (nur bei kategorialen Variablen)
        Häufigkeitstabelle mit absoluter Anzahl und (optional) relativer Häufigkeit in Prozent.

    Hinweise
    --------
    - Erkennt automatisch den Variablentyp (numerisch vs. kategorial).
    - Für numerische Variablen werden u.a. Median, Mittelwert, Quartile, Standardabweichung,
      Whisker-Werte, Schiefe und Kurtosis berechnet.
    - Für kategoriale Variablen wird eine Häufigkeitstabelle erstellt.
    """

    variable = pd.Series(variable)
    n_total = len(variable)
    n_na = variable.isna().sum()
    n_valid = variable.count()
    print(f"\n📊 Summary für: {titel} ({einheit})")
    var_type = "numerisch" if pd.api.types.is_numeric_dtype(variable) else "kategorial"
    summary = {
        "typ": f"{var_type}",
        "anzahl": f"{n_total:,}",
        "anzahl.valid": f"{n_valid:,}",
        "anzahl.na": f"{n_na:,}"
    }
    # 🔹 NUMERISCH
    if var_type == "numerisch":
        variable = variable.dropna()
        q1 = variable.quantile(0.25)
        q3 = variable.quantile(0.75)
        iqr = q3 - q1
        whisker_min = q1 - 1.5 * iqr
        whisker_max = q3 + 1.5 * iqr
        skew = stats.skew(variable)
        kurt = stats.kurtosis(variable, fisher=False)
        skew_txt = (
            f"Rechte Schiefe: {skew:.2f} > 0, positive Schiefe, linkssteil, rechtsschief"
            if skew > 0 else
            f"Linke Schiefe: {skew:.2f} < 0, negative Schiefe, rechtssteil, linksschief"
            if skew < 0 else "Symmetrisch"
        )
        kurt_txt = (
            f"Steilgipflig mit Exzess Kurtosis {kurt:.2f} > 0"
            if kurt > 3 else
            f"Flachgipflig mit Exzess Kurtosis {kurt:.2f} < 3"
        )
        mode_val = variable.mode().iloc[0] if not variable.mode().empty else "—"
        summary.update({
            "modus": f"{mode_val:9.2f}",
            "median": f"{variable.median():9.2f}",
            "mean": f"{variable.mean():9.2f}",
            "min": f"{variable.min():9.2f}",
            "max": f"{variable.max():9.2f}",
            "sd": f"{variable.std():9.2f}",
            "q1": f"{q1:9.2f}",
            "q3": f"{q3:9.2f}",
            "iqr": f"{iqr:9.2f}",
            "whisker.min": f"{whisker_min:9.2f}",
            "whisker.max": f"{whisker_max:9.2f}",
            "skewness": f"{skew:9.2f}",
            "skewness.txt": skew_txt,
            "kurtosis": f"{kurt:9.2f}",
            "kurtosis.txt": kurt_txt,
        })
        for k, v in summary.items():
            print(f"{k:<14}: {v}")
        return summary
    # 🔹 KATEGORIAL
    else:
        variable_unique = variable.nunique(dropna=True)
        summary.update({
            "unique": f"{variable_unique:,}"
        })
        for k, v in summary.items():
            print(f"{k:<14}: {v}")
        variable_clean = variable.dropna()
        abs_freq = variable_clean.value_counts()
        rel_freq = (abs_freq / variable_clean.shape[0] * 100).round(2)
        if top_n is not None:
            abs_freq = abs_freq.head(top_n)
            rel_freq = rel_freq.loc[abs_freq.index]
            print(f"🧾 Häufigkeitstabelle Top-N({top_n}):")
        else:
            print("🧾 Häufigkeitstabelle:")
        if prozent:
            freq_table = pd.DataFrame({
                "Anzahl": abs_freq,
                "Prozent": rel_freq
            })
        else:
            freq_table = pd.DataFrame({
                "Anzahl": abs_freq
            })
        for idx, row in freq_table.iterrows():
            if prozent:
                print(f"{str(idx):<20}: {int(row['Anzahl']):>9,} ({row['Prozent']:>5.1f}%)")
            else:
                print(f"{str(idx):<20}: {int(row['Anzahl']):>9,}")
        return summary, freq_table
