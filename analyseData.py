import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set_style("whitegrid")

chemin_fichier = "acumatica_export.xlsx"
fichier_excel = pd.ExcelFile(chemin_fichier)

TOP_N = 12

with PdfPages("rapport_dashboard_propre.pdf") as pdf:

    for sheet in fichier_excel.sheet_names:
        df = pd.read_excel(fichier_excel, sheet_name=sheet)

        nb_lignes = len(df)
        nb_colonnes = len(df.columns)
        nb_doublons = df.duplicated().sum()

        # =====================================================
        # PAGE 1 : DASHBOARD GRAPHIQUE (ORIGINAL)
        # =====================================================
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"Dashboard Qualité des données - {sheet}",
                     fontsize=16, fontweight="bold")

        # Valeurs manquantes
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(TOP_N)

        if len(missing) > 0:
            sns.barplot(y=missing.index, x=missing.values,
                        palette="Oranges_r", ax=axes[0])
            axes[0].set_title("Colonnes avec valeurs manquantes")
            axes[0].set_xlabel("Nombre")
        else:
            axes[0].text(0.5, 0.5, "Aucune valeur manquante",
                         ha='center', va='center')
            axes[0].axis("off")

        # Valeurs aberrantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) |
                          (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                outlier_counts[col] = len(outliers)

        if len(outlier_counts) > 0:
            outliers_series = pd.Series(outlier_counts) \
                                .sort_values(ascending=False) \
                                .head(TOP_N)
            sns.barplot(y=outliers_series.index, x=outliers_series.values,
                        palette="Reds_r", ax=axes[1])
            axes[1].set_title("Colonnes avec valeurs aberrantes")
            axes[1].set_xlabel("Nombre")
        else:
            axes[1].text(0.5, 0.5, "Aucune valeur aberrante",
                         ha='center', va='center')
            axes[1].axis("off")

        # Statistiques globales
        axes[2].axis("off")
        axes[2].text(0.1, 0.7, f"Nombre de lignes doublantes : {nb_doublons}", fontsize=12)
        axes[2].text(0.1, 0.5, f"Nombre total de lignes : {nb_lignes}", fontsize=12)
        axes[2].text(0.1, 0.3, f"Nombre total de colonnes : {nb_colonnes}", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================
        # PAGE 2 : TABLEAU TOP 10 (ORIGINAL)
        # =====================================================
        fig2, ax = plt.subplots(figsize=(12, 10))
        ax.axis("off")

        resume_data = []
        for col in df.columns:
            miss = df[col].isnull().sum()
            outl = outlier_counts.get(col, 0)
            if miss > 0 or outl > 0:
                score = miss + outl
                resume_data.append([col, int(miss), int(outl), int(score)])

        if resume_data:
            resume_df = pd.DataFrame(
                resume_data,
                columns=["Colonne", "Val. manquantes", "Val. aberrantes", "Score"]
            )
            if sheet in ["StockItem", "PurchaseReceipt", "SalesOrder"]:
                resume_df = resume_df.sort_values("Score", ascending=False).head(10)
            else:
                resume_df = resume_df.sort_values("Score", ascending=False)

            table = ax.table(
                cellText=resume_df.values,
                colLabels=resume_df.columns,
                loc="center",
                cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.3)
            ax.set_title(f"Top colonnes problématiques - {sheet}", fontsize=14, pad=20)
        else:
            ax.text(0.5, 0.5, "Aucun problème détecté",
                    ha='center', va='center', fontsize=12)

        pdf.savefig(fig2)
        plt.close(fig2)

        # =====================================================
        # PAGE 3 : EDA — ANALYSE UNIVARIÉE
        # Statistiques descriptives + histogrammes
        # =====================================================
        if len(numeric_cols) > 0:
            fig3, axes3 = plt.subplots(2, 1, figsize=(12, 10))
            fig3.suptitle(f"EDA — Analyse univariée - {sheet}",
                          fontsize=16, fontweight="bold")

            # Tableau statistiques descriptives (moyenne, médiane, écart-type)
            desc = df[numeric_cols].describe().T[["mean", "50%", "std", "min", "max"]]
            desc.columns = ["Moyenne", "Médiane", "Écart-type", "Min", "Max"]
            desc = desc.round(2).head(TOP_N)

            axes3[0].axis("off")
            tbl = axes3[0].table(
                cellText=desc.reset_index().values,
                colLabels=["Colonne"] + list(desc.columns),
                loc="center",
                cellLoc="center"
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.4)
            axes3[0].set_title("Statistiques descriptives (colonnes numériques)",
                               fontsize=12, pad=14)

            # Histogramme de la colonne numérique la plus complète
            best_col = df[numeric_cols].isnull().mean().idxmin()
            data = df[best_col].dropna()
            axes3[1].hist(data, bins=30, color="#4472C4", edgecolor="white", alpha=0.85)
            axes3[1].axvline(data.mean(), color="red", linestyle="--",
                             linewidth=1.5, label=f"Moyenne = {data.mean():.2f}")
            axes3[1].axvline(data.median(), color="green", linestyle=":",
                             linewidth=1.5, label=f"Médiane = {data.median():.2f}")
            axes3[1].set_title(f"Distribution : {best_col}")
            axes3[1].legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig3)
            plt.close(fig3)

        # =====================================================
        # PAGE 4 : EDA — ANALYSE BIVARIÉE & MULTIVARIÉE
        # Matrice de corrélation
        # =====================================================
        if len(numeric_cols) >= 2:
            fig4, ax4 = plt.subplots(figsize=(12, 9))
            fig4.suptitle(f"EDA — Analyse bivariée & multivariée - {sheet}",
                          fontsize=16, fontweight="bold")

            corr_cols = list(numeric_cols[:12])
            corr_matrix = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                vmin=-1, vmax=1,
                linewidths=0.4,
                ax=ax4,
                annot_kws={"size": 8}
            )
            ax4.set_title("Matrice de corrélation entre colonnes numériques",
                          fontsize=12, pad=12)
            ax4.tick_params(labelsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig4)
            plt.close(fig4)

print("✅ PDF propre généré : rapport_dashboard_propre.pdf")