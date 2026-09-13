import pandas as pd

# Fichiers d'entrée et de sortie
input_file = "acumatica_export.xlsx"
output_file = "acumatica_cleaned.xlsx"

# Lire toutes les feuilles
sheets = pd.read_excel(input_file, sheet_name=None)

# Colonnes à conserver par feuille
COLUMNS_TO_KEEP = {
    "StockItem": [
        "id", "rowNumber", "InventoryID", "Description",
        "ItemType", "ItemClass", "PostingClass", "SalesUOM",
        "DefaultWarehouseID", "DefaultPrice", "ItemStatus",
        "LastCost", "TaxCategory", "DefaultIssueLocationID",
        "DefaultReceiptLocationID", "QtyOnHand"
    ],
    "SalesOrder": [
        "id", "rowNumber", "OrderNbr", "Status",
        "OrderType", "CustomerID", "Date",
        "RequestedOn", "ShipOn", "CreatedDate",
        "OrderedQty", "OrderTotal", "CurrencyID",
        "PreferredWarehouseID", "DestinationWarehouseID",
        "InventoryID"
    ],
    "PurchaseOrder": [
        "id", "rowNumber", "OrderNbr", "Status",
        "Type", "Date", "VendorID",
        "LastModifiedDateTime", "CurrencyID",
        "OrderQty", "InventoryID"
    ],
    "PurchaseReceipt": [
        "id", "rowNumber", "ReceiptNbr", "Status",
        "TotalQty", "Type", "VendorID",
        "Date", "CurrencyID", "TotalCost",
        "Warehouse", "InventoryID"
    ],
    "Shipment": [
        "id", "rowNumber", "Description", "Type",
        "Status", "Operation", "WarehouseID",
        "ShippedQty", "ShippedWeight", "CustomerID",
        "CreatedDateTime", "ShipmentNbr",
        "ShipmentDate", "InventoryID"
    ]
}

cleaned_sheets = {}

for sheet_name, df in sheets.items():

    if sheet_name in COLUMNS_TO_KEEP:#traiter seul les col qui existent dans  columns_to_keep

        print(f"\n📄 Traitement de : {sheet_name}")

        # Normaliser noms colonnes (supp espaces)
        df.columns = df.columns.str.strip()

        # Garder seulement les colonnes existantes dans le df
        available_cols = [col for col in COLUMNS_TO_KEEP[sheet_name] if col in df.columns]
        df = df[available_cols]

        before = len(df)

        # Grouper par 'id' et supp les val vides et on prend la prem val valide
        #objct: supp doublons et fusionner lignes partielles
        df = df.groupby("id", as_index=False).agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)

        # Recalculer un rowNumber cohérent (1, 2, 3…) index propre
        if "rowNumber" in df.columns:
            df["rowNumber"] = range(1, len(df) + 1)  # mais si ya une col on la Rempla
        after = len(df)

        print(f"Lignes avant nettoyage : {before}")
        print(f"Lignes après nettoyage : {after}")

        cleaned_sheets[sheet_name] = df#sauvega vers nettoy de chaque sheet

# Sauvegarder toutes les feuilles nettoyées dans un nouveau fichier Excel
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for sheet_name, df in cleaned_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("\n✅ Nettoyage terminé.")
print("Fichier généré :", output_file)




