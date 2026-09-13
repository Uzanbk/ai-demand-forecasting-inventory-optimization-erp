"""
=============================================================
 Encodages appliqués :

 STOCKITEM
   ItemStatus              → Ordinal (0=Deleted … 3=Active)
   ItemType                → OHE
   ItemClass               → OHE
   DefaultWarehouseID      → OHE

 SALESORDER
   Status                  → OHE
   OrderType               → Regroupement métier (7→3) + OHE
   PreferredWarehouseID    → OHE
   DestinationWarehouseID  → Binaire is_transfert_interne (0/1)

 SHIPMENT
   Type                    → Binaire (0=Shipment, 1=Transfer)
   Status                  → Binaire (0=Confirmed, 1=Completed)
   WarehouseID             → OHE

 PURCHASEORDER
   Status                  → OHE
   Type                    → OHE
   VendorID                → Label Encoding
                             (à déclarer categorical_feature dans LightGBM)

 PURCHASERECEIPT
   Status                  → Drop (constante : Released)
   Type                    → Binaire (0=Receipt, 1=Transfer Receipt)
   Warehouse               → OHE
   VendorID                → Label Encoding
                             (à déclarer categorical_feature dans LightGBM)
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# UTILITAIRES COMMUNS
# ─────────────────────────────────────────────

def clean_base(df):
    """Nettoyages universels : doublons, chaînes vides, {}"""
    df = df.drop_duplicates()
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(r'^\{\}$', np.nan, regex=True)
    df = df.replace("{}", np.nan)
    return df


def parse_dates(df, date_cols):
    """Conversion robuste des colonnes dates + suppression timezone"""
    for col in date_cols:
        if col not in df.columns:
            continue
        #si la colonne est numérique on suppose que cest un format excel 
        #errors="coerce" transforme les valeurs invalides en nat(not a time)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], origin="1899-12-30", unit="D", errors="coerce")
        #si la colonne est texte pandas essaie de convertir automatiqu
        #"2024-01-01" en "01/01/2024"
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        #suppression des fuseaux horarires
        try:
            df[col] = df[col].dt.tz_localize(None)
        #si tz_localize ne marche pas en essaye tz_convert
        except TypeError:
            try:
                df[col] = df[col].dt.tz_convert(None)
            # si rien ne marche on ignore 
            except Exception:
                pass
    return df


def filter_negative(df, qty_cols):
    """Supprime les lignes avec quantités 0"""
    for col in qty_cols:
        if col in df.columns:
            before = len(df)
            df = df[df[col] >0]
            dropped = before - len(df)
            if dropped:
                print(f"  ⚠️  {col} : {dropped} ligne(s) avec quantité 0 supprimée(s)")
    return df

# on utilise la méthode statis basée sur Q1=25% Q2=75% IQR=Q3-Q1
#puis on definit une limite: Q3+upper*IQR
def winsorize_qty(df, qty_col, factor=3.0):
    """
    Plafonne les quantités extrêmes à IQR×factor.
    Conserve le flag binaire et la valeur originale.
    """
    if qty_col not in df.columns:
        return df
    #calcul des quartiles et iqr
    q1, q3 = df[qty_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + factor * iqr
    #crée une col si val extré(pic) 0 si normale
    df[f"{qty_col}_pic_flag"] = (df[qty_col] > upper).astype(int)
    #comptage de nbr de pics
    n_pics = df[f"{qty_col}_pic_flag"].sum()
    if n_pics > 0:
        print(f"  ℹ️  {qty_col} : {n_pics} valeur(s) plafonnée(s) à {upper:.0f} ")
    #revenir aux données initiales
    df[f"{qty_col}_original"] = df[qty_col].copy()
    #tte les val sup à upper seront remplacées par upper
    #clip(): fonction qui limite les val dans un intervalle
    df[qty_col] = df[qty_col].clip(upper=upper)
    return df


def report(name, df):
    print(f"\n{'='*55}")
    print(f"  ✅  {name}  |  Dimensions : {df.shape}")
    print(f"{'='*55}")
    nulls = df.isnull().sum()
    #on garde seulement les élem où la val>0
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  → Aucune valeur manquante")
    else:
        #nulls.to_string(): transforme nulls en série pandas en texte lisible complet 
        print(f"  → Valeurs manquantes :\n{nulls.to_string()}")
    print(f"  → Types :\n{df.dtypes.to_string()}")


# ────────────────
# 1. STOCK ITEM
# ────────────────

def clean_stock_item(path):
    print("\n\n📦 Nettoyage StockItem...")
    stock = pd.read_excel(path, sheet_name="StockItem")
    stock = clean_base(stock)

    cols_to_drop = [
        "id", "rowNumber", "Description", "PostingClass",
        "SalesUOM", "TaxCategory",
        "DefaultIssueLocationID", "DefaultReceiptLocationID",
        "NoteID", "LastModifiedDateTime", "CreatedDateTime"
    ]
    stock = stock.drop(columns=[c for c in cols_to_drop if c in stock.columns])

    stock = filter_negative(stock, ["QtyOnHand", "DefaultPrice", "LastCost"])

    # Flag articles à prix et coût nuls
    mask_zero = (stock["DefaultPrice"] == 0) & (stock["LastCost"] == 0)
    if mask_zero.sum():
        stock["price_zero_flag"] = mask_zero.astype(int)

    # Taux de marge
    if "DefaultPrice" in stock.columns and "LastCost" in stock.columns:
        stock["margin_rate"] = np.where(
            stock["DefaultPrice"] > 0,
            (stock["DefaultPrice"] - stock["LastCost"]) / stock["DefaultPrice"],
            np.nan
        )

    
    # Ordre métier : Active=3 > No Sales=2 > No Purchases=1 > Deleted=0
    if "ItemStatus" in stock.columns:
        ITEM_STATUS_ORDER = {
            "Active":               3,
            "No Sales":             2,
            "No Purchases":         1,
            "Marked for Deletion":  0,
        }
        #transforme en texte, enlève les espaces
        #.map(): remplace chaque texte par son nombre
        #si une valeur nest pas reconnue elle devient 0
        #et aprés transforme en entier
        stock["ItemStatus"] = (
            stock["ItemStatus"]
            .astype(str).str.strip()
            .map(ITEM_STATUS_ORDER)
            .fillna(0)
            .astype(int)
        )
        print("   ItemStatus → ordinal (0=Deleted … 3=Active)")

    # ── ItemType, ItemClass, DefaultWarehouseID : OHE ─────────────────────────
    # Variables nominales sans ordre naturel.
    # drop_first=True évite la multicolinéarité.
    ohe_cols = ["ItemType", "ItemClass", "DefaultWarehouseID"]
    for col in ohe_cols:
        if col in stock.columns:
            stock[col] = stock[col].astype(str).str.strip()
            stock[col] = stock[col].replace("nan", np.nan)
    stock = pd.get_dummies(
        stock,
        columns=[c for c in ohe_cols if c in stock.columns],
        drop_first=True
    )

    # Suppression des colonnes booléennes constantes éventuelles
    bool_cols = stock.select_dtypes(include="bool").columns
    constant_cols = [c for c in bool_cols if stock[c].nunique() == 1]
    if constant_cols:
        stock = stock.drop(columns=constant_cols)

    report("StockItem", stock)
    return stock


# ────────────────
# 2. SALES ORDER
# ────────────────

def clean_sales_order(path):
    print("\n\n🛒 Nettoyage SalesOrder...")
    sales = pd.read_excel(path, sheet_name="SalesOrder")
    sales = clean_base(sales)

    cols_to_drop = [
        "id", "rowNumber", "OrderNbr", "CreatedDate",
        "CustomerID", "OrderTotal", "CurrencyID", "NoteID"
    ]
    sales = sales.drop(columns=[c for c in cols_to_drop if c in sales.columns])

    date_cols = ["Date", "RequestedOn", "ShipOn"]
    sales = parse_dates(sales, date_cols)
    #je garde uniquement les colonnes qui existent dans sales
    existing_date_cols = [c for c in date_cols if c in sales.columns]
    #remplacer  les val manquantes par la dernière valeur valide précédente
    for col in existing_date_cols:
        sales[col] = sales[col].ffill()

    before = len(sales)
    sales = sales.dropna(subset=existing_date_cols)
    print(f"  → {before - len(sales)} ligne(s) sans date supprimée(s)")

    if "Date" in sales.columns and "ShipOn" in sales.columns:
        incoherent = sales["ShipOn"] < sales["Date"]
        #supprimer les lignes incohérentes
        sales = sales[~incoherent]
        sales["lead_time_days"] = (sales["ShipOn"] - sales["Date"]).dt.days
        #clip(lower=0): tte les val néga deviennent 0
        sales["lead_time_days"] = sales["lead_time_days"].clip(lower=0)

    qty_col = "OrderedQty" if "OrderedQty" in sales.columns else "OrderQty"
    sales = filter_negative(sales, [qty_col])
    sales = winsorize_qty(sales, qty_col, factor=3.0)

    if "InventoryID" in sales.columns and "lead_time_days" in sales.columns:
        sales["lead_time_hist_moy"] = (
            sales.sort_values("Date")
            .groupby("InventoryID")["lead_time_days"]
            #x.shift(1) décale les val dune ligne vers le bas
            #expanding().mean: calcul une moy cumulative
            .transform(lambda x: x.shift(1).expanding().mean())
        ).round(1)
        print(f"  ℹ️  lead_time_hist_moy calculé pour {sales['InventoryID'].nunique()} articles")

    if "Date" in sales.columns and "InventoryID" in sales.columns:
        #extrait le numéro de semaine(1à52/53)
        sales["semaine_iso"] = sales["Date"].dt.isocalendar().week.astype(int)
        #moy par produit et semaine
        moy_sem  = sales.groupby(["InventoryID", "semaine_iso"])[qty_col].transform("mean")
        #moy glob par produit
        moy_glob = sales.groupby("InventoryID")[qty_col].transform("mean")
        #comparer ventes dune semaine avec les ventes moy du produit
        #remplacer les 0 par nan pour éviter une divison par zéro
        sales["ratio_sem_vs_glob"] = (moy_sem / moy_glob.replace(0, np.nan)).round(3)
        #si ratio >= 1.5=semaine forte sinon 0
        sales["semaine_forte"] = (sales["ratio_sem_vs_glob"] >= 1.5).astype(int)
        n_fortes = sales["semaine_forte"].sum()
        print(f"  ℹ️  {n_fortes} lignes identifiées comme semaines commercialement fortes")

    # ── OrderType : regroupement métier (7→3) puis OHE ───────────────────────
    # SO/IN/PS → VENTE, TR → TRANSFERT, PC/QT/MO → AUTRE
    # Evite 7 colonnes OHE quasi-vides.
    if "OrderType" in sales.columns:
        ORDER_TYPE_MAP = {
            "SO": "VENTE", "IN": "VENTE", "PS": "VENTE",
            "TR": "TRANSFERT",
            "PC": "AUTRE",  "QT": "AUTRE",  "MO": "AUTRE",
        }
        sales["OrderType"] = (
            sales["OrderType"].astype(str).str.strip()
            .map(ORDER_TYPE_MAP).fillna("AUTRE")
        )
        print(f"  ℹ️  OrderType regroupé : {sales['OrderType'].value_counts().to_dict()}")

    # ── DestinationWarehouseID : Binaire → is_transfert_interne ──────────────
    # {} → NaN → 0 (vente normale), NIXISGROUP/MAGASIN → 1 (transfert interne)
    if "DestinationWarehouseID" in sales.columns:
        sales["is_transfert_interne"] = (
            #notna(): vérifie si la valeur nest pas manquante
            #resultat true=valeur existe false=valeur vide nan
            sales["DestinationWarehouseID"].notna()
        ).astype(int)
        sales = sales.drop(columns=["DestinationWarehouseID"])
        print("  ℹ️  DestinationWarehouseID → is_transfert_interne (0/1)")

    # ── Status, OrderType, PreferredWarehouseID : OHE ─────────────────────────
    # Variables nominales sans ordre naturel.
    # NaN ignorés par get_dummies, drop_first=True évite la multicolinéarité.
    ohe_cols = ["Status", "OrderType", "PreferredWarehouseID"]
    for col in ohe_cols:
        if col in sales.columns:
            sales[col] = sales[col].astype(str).str.strip()
            sales[col] = sales[col].replace("nan", np.nan)
    sales = pd.get_dummies(
        sales,
        columns=[c for c in ohe_cols if c in sales.columns],
        drop_first=True
    )

    report("SalesOrder", sales)
    return sales


# ────────────────
# 3. SHIPMENT
# ────────────────

def clean_shipment(path):
    print("\n\n🚚 Nettoyage Shipment...")
    shipment = pd.read_excel(path, sheet_name="Shipment")
    shipment = clean_base(shipment)

    cols_to_drop = [
        "id", "rowNumber", "Description", "Operation",
        "ShippedWeight", "CustomerID", "CreatedDateTime", "NoteID"
    ]
    shipment = shipment.drop(columns=[c for c in cols_to_drop if c in shipment.columns])

    date_cols = [c for c in shipment.columns if "date" in c.lower()]
    shipment = parse_dates(shipment, date_cols)
    before = len(shipment)
    shipment = shipment.dropna(subset=[c for c in date_cols if c in shipment.columns])
    print(f"  → {before - len(shipment)} ligne(s) sans date supprimée(s)")

    qty_col = "ShippedQty" if "ShippedQty" in shipment.columns else "ShipmentQty"
    shipment = filter_negative(shipment, [qty_col])
    shipment = winsorize_qty(shipment, qty_col, factor=3.0)

    # ── Type : Binaire ────────────────────────────────────────────────────────
    # 2 valeurs : "Shipment" (expédition client) = 0, "Transfer" (inter-entrepôts) = 1
    if "Type" in shipment.columns:
        shipment["Type"] = (
            shipment["Type"].astype(str).str.strip() == "Transfer"
        ).astype(int)
        print("  ℹ️  Type → binaire (0=Shipment, 1=Transfer)")

    # ── Status : Binaire ──────────────────────────────────────────────────────
    # 2 valeurs : "Confirmed" = 0, "Completed" = 1
    if "Status" in shipment.columns:
        shipment["Status"] = (
            shipment["Status"].astype(str).str.strip() == "Completed"
        ).astype(int)
        print("  ℹ️  Status → binaire (0=Confirmed, 1=Completed)")

    # ── WarehouseID : OHE ─────────────────────────────────────────────────────
    # 6 entrepôts sans ordre naturel, drop_first=True évite la multicolinéarité.
    if "WarehouseID" in shipment.columns:
        shipment["WarehouseID"] = shipment["WarehouseID"].astype(str).str.strip()
        shipment["WarehouseID"] = shipment["WarehouseID"].replace("nan", np.nan)
    shipment = pd.get_dummies(
        shipment,
        columns=[c for c in ["WarehouseID"] if c in shipment.columns],
        drop_first=True
    )

    report("Shipment", shipment)
    return shipment


# ────────────────
# 4. PURCHASE ORDER
# ────────────────

def clean_purchase_order(path):
    print("\n\n📋 Nettoyage PurchaseOrder...")
    po = pd.read_excel(path, sheet_name="PurchaseOrder")
    po = clean_base(po)

    cols_to_drop = [
        "id", "rowNumber", "OrderNbr",
        "LastModifiedDateTime", "CurrencyID",
        "TotalAmount", "NoteID"
    ]
    po = po.drop(columns=[c for c in cols_to_drop if c in po.columns])

    po = parse_dates(po, ["Date"])
    before = len(po)
    po = po.dropna(subset=["Date"])
    print(f"  → {before - len(po)} ligne(s) sans date supprimée(s)")

    po = filter_negative(po, ["OrderQty"])
    po = winsorize_qty(po, "OrderQty", factor=3.0)

    # ── Status, Type : OHE ───────────────────────────────────────────────────
    # Status : 3 valeurs (Closed, Completed, Open) sans ordre naturel
    # Type   : 3 valeurs (Normal, Drop-Ship, Blanket) sans ordre naturel
    # drop_first=True évite la multicolinéarité.
    ohe_cols = ["Status", "Type"]
    for col in ohe_cols:
        if col in po.columns:
            po[col] = po[col].astype(str).str.strip()
            po[col] = po[col].replace("nan", np.nan)
    po = pd.get_dummies(
        po,
        columns=[c for c in ohe_cols if c in po.columns],
        drop_first=True
    )

    # ── VendorID : Label Encoding ─────────────────────────────────────────────
    # 11 fournisseurs. Label Encoding choisi intentionnellement.
    # ⚠️  À déclarer dans categorical_feature de LightGBM pour que le modèle
    # ignore l'ordre numérique implicite et traite chaque valeur indépendamment.
    encoders_po = {}
    if "VendorID" in po.columns:
        #convertir tte les val en texte et remplacer les val manquantes nan par Unknown
        po["VendorID"] = po["VendorID"].astype(str).fillna("Unknown")
        le = LabelEncoder()
        po["VendorID"] = le.fit_transform(po["VendorID"])
        encoders_po["VendorID"] = le
        #zip()associe chaque valeur avec son code 
        print(f"  ℹ️  VendorID → Label Encoding : {dict(zip(le.classes_, le.transform(le.classes_).tolist()))}")

    report("PurchaseOrder", po)
    return po, encoders_po


# ────────────────
# 5. PURCHASE RECEIPT
# ────────────────

def clean_purchase_receipt(path):
    print("\n\n📥 Nettoyage PurchaseReceipt...")
    pr = pd.read_excel(path, sheet_name="PurchaseReceipt")
    pr = clean_base(pr)

    cols_to_drop = [
        "id", "rowNumber", "ReceiptNbr", "NoteID",
        "CurrencyID", "TotalCost",
        # Status supprimé : constante (Released = 1 seule valeur, zéro information)
        "Status",
    ]
    pr = pr.drop(columns=[c for c in cols_to_drop if c in pr.columns])

    pr = parse_dates(pr, ["Date"])
    before = len(pr)
    pr = pr.dropna(subset=["Date"])
    print(f"  → {before - len(pr)} ligne(s) sans date supprimée(s)")

    if "InventoryID" in pr.columns:
        before = len(pr)
        pr = pr.dropna(subset=["InventoryID"])
        print(f"  → {before - len(pr)} ligne(s) sans InventoryID supprimée(s)")

    pr = filter_negative(pr, ["TotalQty"])
    pr = winsorize_qty(pr, "TotalQty", factor=3.0)

    # ── Type : Binaire ────────────────────────────────────────────────────────
    # 2 valeurs : "Receipt" (réception normale) = 0,
    #             "Transfer Receipt" (réception inter-entrepôts) = 1
    if "Type" in pr.columns:
        pr["Type"] = (
            pr["Type"].astype(str).str.strip() == "Transfer Receipt"
        ).astype(int)
        print("  ℹ️  Type → binaire (0=Receipt, 1=Transfer Receipt)")

    # ── Warehouse : OHE ───────────────────────────────────────────────────────
    # 6 entrepôts sans ordre naturel, drop_first=True évite la multicolinéarité.
    if "Warehouse" in pr.columns:
        pr["Warehouse"] = pr["Warehouse"].astype(str).str.strip()
        pr["Warehouse"] = pr["Warehouse"].replace("nan", np.nan)
    pr = pd.get_dummies(
        pr,
        columns=[c for c in ["Warehouse"] if c in pr.columns],
        drop_first=True
    )

    # ── VendorID : Label Encoding ─────────────────────────────────────────────
    # 11 fournisseurs. Label Encoding choisi intentionnellement.
    # ⚠️  À déclarer dans categorical_feature de LightGBM pour que le modèle
    # ignore l'ordre numérique implicite et traite chaque valeur indépendamment.
    encoders_pr = {}
    if "VendorID" in pr.columns:
        pr["VendorID"] = pr["VendorID"].astype(str).fillna("Unknown")
        le = LabelEncoder()
        pr["VendorID"] = le.fit_transform(pr["VendorID"])
        encoders_pr["VendorID"] = le
        print(f"  ℹ️  VendorID → Label Encoding : {dict(zip(le.classes_, le.transform(le.classes_).tolist()))}")

    report("PurchaseReceipt", pr)
    return pr, encoders_pr


# ────────────────
# EXPORT
# ────────────────

def export_all(stock, sales, shipment, po, pr):
    stock.to_excel("StockItem_cleaned.xlsx",    index=False, engine="openpyxl")
    sales.to_excel("SalesOrder_cleaned.xlsx",    index=False, engine="openpyxl")
    shipment.to_excel("Shipment_cleaned.xlsx",   index=False, engine="openpyxl")
    po.to_excel("PurchaseOrder_cleaned.xlsx",    index=False, engine="openpyxl")
    pr.to_excel("PurchaseReceipt_cleaned.xlsx",  index=False, engine="openpyxl")


# ────────────────
# MAIN
# ────────────────

if __name__ == "__main__":
    SOURCE = "acumatica_cleaned.xlsx"

    stock               = clean_stock_item(SOURCE)
    sales               = clean_sales_order(SOURCE)
    shipment            = clean_shipment(SOURCE)
    po, po_enc          = clean_purchase_order(SOURCE)
    pr, pr_enc          = clean_purchase_receipt(SOURCE)

    export_all(stock, sales, shipment, po, pr)

    print("\n\n🎉 Tous les fichiers ont été nettoyés et exportés avec succès !")