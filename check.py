import pandas as pd

file_path = "acumatica_cleaned.xlsx"
output_file = "data_quality_report.xlsx"

xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

summary_rows = []
categorical_rows = []

for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    categorical_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # --- résumé global ---
    summary_rows.append({
        "Sheet": sheet,
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing_Values": df.isna().sum().sum(),
        "Duplicate_Rows": df.duplicated().sum(),
        "Numeric_Columns_Count": len(numeric_cols),
        "Categorical_Columns_Count": len(categorical_cols),
    })

    # --- détails catégoriels ---
    for col in categorical_cols:
        unique_vals = df[col].dropna().unique()

        categorical_rows.append({
            "Sheet": sheet,
            "Column": col,
            "N_Unique": len(unique_vals),
            "Sample_Values": ", ".join(map(str, unique_vals[:10]))
        })

# Convertir en DataFrame
summary_df = pd.DataFrame(summary_rows)
categorical_df = pd.DataFrame(categorical_rows)

# --- Export Excel ---
with pd.ExcelWriter(output_file) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    categorical_df.to_excel(writer, sheet_name="Categorical_Details", index=False)

print(f"✅ Rapport sauvegardé dans : {output_file}")