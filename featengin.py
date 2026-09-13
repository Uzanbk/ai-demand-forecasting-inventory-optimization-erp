import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES FICHIERS NETTOYÉS
# ─────────────────────────────────────────────

print("📂 Chargement des fichiers nettoyés...")

stock    = pd.read_excel("StockItem_cleaned.xlsx")
sales    = pd.read_excel("SalesOrder_cleaned.xlsx")
shipment = pd.read_excel("shipment_cleaned.xlsx")
po       = pd.read_excel("PurchaseOrder_cleaned.xlsx")
pr       = pd.read_excel("PurchaseReceipt_cleaned.xlsx")

# ─────────────────────────────────────────────
# 2. PRÉPARATION DES DATES
# ─────────────────────────────────────────────

print("📅 Préparation des dates...")

def to_week(df, date_col):
    #si valeur invalide devient NaT
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")#transforme ay col date en format datetime
    #regroupe chaque date dans sa sem
    #2025-03-12 → semaine du 10 mars 2025
    #apply(lambda): recupere le debut de semaine ca transforme la sem en date réelle(2025-03-12 → 2025-03-10)
    df["semaine"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
    return df

sales    = to_week(sales,    "Date")
shipment = to_week(shipment, "ShipmentDate")
po       = to_week(po,       "Date")
pr       = to_week(pr,       "Date")

# ─────────────────────────────────────────────
# 3. GRILLE DE BASE : InventoryID × Semaine
# ─────────────────────────────────────────────

print("🔲 Construction de la grille InventoryID × Semaine...")


#select tt les col issues du ohe
item_type_cols  = [c for c in stock.columns if c.startswith("ItemType_")]
item_class_cols = [c for c in stock.columns if c.startswith("ItemClass_")]

if item_type_cols and item_class_cols:
    has_type  = stock[item_type_cols].any(axis=1)#detection des lignes axis verifie par ligne
    has_class = stock[item_class_cols].any(axis=1)

    # CORRECTION : filtre sur la colonne ordinale ItemStatus
    if "ItemStatus" in stock.columns:
        is_active = stock["ItemStatus"] >= 1   # exclut Marked for Deletion (0)
    #si la colonne n'existe pas jassume que tous les art sont valides
    else:
        is_active = pd.Series(True, index=stock.index)
#un article est stockable si a un type,classe,et actif
    stockable_ids = stock.loc[has_type & has_class & is_active, "InventoryID"].unique()
#si pas de features encodées on prend tous les art non nuls
else:
    stockable_ids = stock["InventoryID"].dropna().unique()




#si inventoryid appartient à la liste stock donc on garde la lig
sales_filtered = sales[sales["InventoryID"].isin(stockable_ids)].copy()
n_total     = sales["InventoryID"].nunique()
n_stockable = sales_filtered["InventoryID"].nunique()
print(f"  → {n_total - n_stockable} article(s) non stockable(s) exclus | {n_stockable} conservés")

all_items = sales_filtered["InventoryID"].dropna().unique()
date_min  = sales_filtered["Date"].min()
date_max  = sales_filtered["Date"].max()
#crée une liste de semaines entre 2 dates avec freq hebdo chaque lundi
all_weeks = pd.date_range(start=date_min, end=date_max, freq="W-MON")



#creation de tt les combin possibles(art*sem)
base = pd.MultiIndex.from_product(
    [all_items, all_weeks],
    names=["identifiant_article", "semaine"]
).to_frame(index=False)
#to_frame transforme le multiIndex en dataframe classique snn jobtient index compliqué à manipuler
sales = sales_filtered

# ─────────────────────────────────────────────
# 4. FEATURES TEMPORELLES
# ─────────────────────────────────────────────

print("🗓️  Calcul des features temporelles...")

base["annee"]            = base["semaine"].dt.year
base["mois"]             = base["semaine"].dt.month#saisonnalité mensuelle
base["semaine_annee"]    = base["semaine"].dt.isocalendar().week.astype(int)#saisonnalité fine
base["trimestre"]        = base["semaine"].dt.quarter
#qui vaut 1 si la semaine est dans les 6 derniers jrs du mois snn 0
base["fin_de_mois"]      = (base["semaine"].dt.days_in_month - base["semaine"].dt.day <= 6).astype(int)
#verifie si les mois est mars, juin,sep,décem(resultat booléen) astypeint tbadalha binaire
base["fin_de_trimestre"] = base["mois"].isin([3, 6, 9, 12]).astype(int)



# ─────────────────────────────────────────────
# 5. FEATURES DEMANDE (SalesOrder)
# ─────────────────────────────────────────────

print("📦 Calcul des features de demande...")

#parcourt tt les colo de sales et select eli fihom canceled(us), cancelled(uk)
status_canceled_col  = [c for c in sales.columns if "Canceled" in c or "Cancelled" in c]
status_backorder_col = [c for c in sales.columns if "Back" in c]
qty_col              = "OrderedQty" if "OrderedQty" in sales.columns else "OrderQty"

semaine_forte_col  = "semaine_forte"      if "semaine_forte"      in sales.columns else None
lead_time_hist_col = "lead_time_hist_moy" if "lead_time_hist_moy" in sales.columns else None
#colonne pic de demande (ex:orderedqty_pic_flag)
pic_flag_col       = f"{qty_col}_pic_flag" if f"{qty_col}_pic_flag" in sales.columns else None

agg_sales = sales.groupby(["InventoryID", "semaine"]).agg(
    quantite_commandee = (qty_col, "sum"),
    nb_commandes       = (qty_col, "count"),
).reset_index()
#on verifie si la var existe on calcule la feat sinon on ignore ce bloc
if semaine_forte_col:
    #on calcul la mean psq semaine_forte_col est souvent binaire(1sem forte semaine normale0 donc moy c la proport de sem fortes)
    agg_forte = sales.groupby(["InventoryID","semaine"])[semaine_forte_col].mean().reset_index()
    #renommage des col 
    agg_forte.columns = ["InventoryID","semaine","taux_semaine_forte"]
    #ajout du feature au dataset princ  gardant tte les lignes de agg_sales
    agg_sales = agg_sales.merge(agg_forte, on=["InventoryID","semaine"], how="left")
    agg_sales["taux_semaine_forte"] = agg_sales["taux_semaine_forte"].fillna(0)
 #resultat variable continue(exp:2J,4J,3J=3j)                         
if lead_time_hist_col:
    agg_lt = sales.groupby(["InventoryID","semaine"])[lead_time_hist_col].mean().reset_index()
    agg_lt.columns = ["InventoryID","semaine","lead_time_hist_moy"]
    agg_sales = agg_sales.merge(agg_lt, on=["InventoryID","semaine"], how="left")
#pic_flag_col est binaire .sum()= nbr de pics(1,0,1= 2pics)
if pic_flag_col:
    agg_pic = sales.groupby(["InventoryID","semaine"])[pic_flag_col].sum().reset_index()
    agg_pic.columns = ["InventoryID","semaine","nb_pics_semaine"]
    agg_sales = agg_sales.merge(agg_pic, on=["InventoryID","semaine"], how="left")
    agg_sales["nb_pics_semaine"] = agg_sales["nb_pics_semaine"].fillna(0)

if status_canceled_col:
    #status_canceled_col est une liste donc on prend la première colonne
    agg_c = sales.groupby(["InventoryID","semaine"])[status_canceled_col[0]].mean().reset_index()
    #renommer canceled en taux_d'annulation
    #mean() sur une col binaire donne un taux 
    agg_c.columns = ["InventoryID","semaine","taux_annulation"]
    #jointure left: garde tt les colonnes de agg_sales
    agg_sales = agg_sales.merge(agg_c, on=["InventoryID","semaine"], how="left")
else:
    agg_sales["taux_annulation"] = 0.0
#si la liste n'est pas vide 
if status_backorder_col:
    #on prend la colonne backorder et on calcule la moyenne
    agg_bo = sales.groupby(["InventoryID","semaine"])[status_backorder_col[0]].mean().reset_index()
    #renommage des colonnes
    agg_bo.columns = ["InventoryID","semaine","taux_rupture_commande"]
    #on ajoute l'info de agg_bo et on garde toutes les lignes de agg_sales
    agg_sales = agg_sales.merge(agg_bo, on=["InventoryID","semaine"], how="left")
#si la col n'existe pas donc on met 0 partout
else:
    agg_sales["taux_rupture_commande"] = 0.0

agg_sales = agg_sales.rename(columns={"InventoryID": "identifiant_article"})

base = base.merge(agg_sales, on=["identifiant_article","semaine"], how="left")
# on remplace tous les nan par 0
base["quantite_commandee"]    = base["quantite_commandee"].fillna(0)
base["nb_commandes"]          = base["nb_commandes"].fillna(0)
base["taux_annulation"]       = base["taux_annulation"].fillna(0)
base["taux_rupture_commande"] = base["taux_rupture_commande"].fillna(0)
for col in ["taux_semaine_forte","lead_time_hist_moy","nb_pics_semaine"]:
    if col in base.columns:
        base[col] = base[col].fillna(0)

# ── Lags et moyennes glissantes
print("   → Lags et moyennes glissantes...")
base = base.sort_values(["identifiant_article","semaine"])

base["lag_1_semaine"]  = base.groupby("identifiant_article")["quantite_commandee"].shift(1)
base["lag_2_semaines"] = base.groupby("identifiant_article")["quantite_commandee"].shift(2)

base["lag_3_semaines"] = base.groupby("identifiant_article")["quantite_commandee"].shift(3)
base["lag_4_semaines"] = base.groupby("identifiant_article")["quantite_commandee"].shift(4)
base["lag_8_semaines"] = base.groupby("identifiant_article")["quantite_commandee"].shift(8)
#shift(1)= enlève la semaine actuelle

#on prend les 4 dernières semaines et on calcule la moy
base["moyenne_glissante_4s"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)
base["moyenne_glissante_12s"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
)

# mesure de variabilité (instabilité) de la demande sur les 4 dernières semaines
#std: à quel point les valeurs s'écartent en moyenne de la moyenne
#faible std: les val sont proches, std élevé: les val sont plus étalées
base["std_glissante_4s"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=2).std())
).fillna(0)
#formule: écart-type/ moyenne
#on évite la division par 0 si moy=0 donc devient NaN
#on remplace les cas problématiques par 0
base["coef_variation_4s"] = (
    base["std_glissante_4s"] / base["moyenne_glissante_4s"].replace(0, np.nan)
).fillna(0).round(3)

# ── Features activité: pour mesurer à quel point un article est actif ou inactif 
print("   → Features activité...")
#on enlève la semaine actuelle et on garde juste le passé 
#(x==0) transforme en 1 si qté=0 sinon 0
#expanding().mean()= moyenne cumulative
base["taux_zeros_historique"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: (x.shift(1)==0).expanding().mean())
).round(2)
#on regarde la semaine précédente si qté >0 donc semaine active=1 sinn 0
base["semaine_active_lag1"] = (base["lag_1_semaine"] > 0).astype(int)
# meme logique mais pour ya 2 semaines
base["semaine_active_lag2"] = (base["lag_2_semaines"] > 0).astype(int)
#shift(1)>0:tranformation binaire: 1=ya eu une vente, 0=aucune vente
#rolling 12=on regarde une fenetre de 12 semaines
#on calcul la moy des 0 et 1
base["taux_activite_12s"]   = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: (x.shift(1)>0).rolling(12, min_periods=1).mean())
).round(2)
#pourcentage de semaines où ya eu des ventes sur les 12dernières semaines 
# ── NOUVELLES features pour demande intermittente ──────────────────────────
print("   → Features intermittentes (moy_actives, prob_reactivation, sem_depuis_vente)...")

# moy_actives_8s : moyenne des 8 dernières semaines OÙ la demande était > 0
# Différent de moyenne_glissante_8s qui inclut les zéros.
# Capture la taille typique d'une commande quand il y en a une.
base["moy_actives_8s"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: (
        x.shift(1)
         .rolling(8, min_periods=1)
         #dans chaque fenetre on ignore les zéros(w>0)on calcul la moyenne des val+ si 
         #n'ya aucune vente on met 0
         .apply(lambda w: w[w > 0].mean() if (w > 0).any() else 0, raw=True)
    ))
).fillna(0).round(1)

# pour chaque sem : depuis combien de sem n'ya pas eu de vente avant cette sem

def calc_sem_depuis_vente(serie):
    #resultat= tableau meme taille que la série
    result = np.zeros(len(serie))
    #compteur: nbr de sem sans vente depuis la dernière vente
    compteur = 0
    #parcourt les sem une par une
    for i, val in enumerate(serie):
        # Si serie[0] > 0 : result[0] = 0 (ok)
        # Si serie[0] == 0 : compteur = 1
        #si ya une vente la sem préc le cmpt=0
        #si ya pas le compteur commence à compter 
        if i == 0:#première semaine
            result[i] = 0 #tjrs on met 0 psq pas dhistorique avant pour la prem sem
            compteur = 0 if val > 0 else 1
        else:#pour les autres semaines
            # on enregistre dab le compt actuel,On compte les sem sans ventes avant cette semaine
            result[i] = compteur
            #si vente on met à jour le compt à 0
            if val > 0:
                compteur = 0
            #sinon on ajoute 1(sem de plus sans vente)
            else:
                compteur += 1
    return result
#pour chaque article,on prend la serie des ventes, on applique la fonction 
#transform: permet de garder la meme taille que le dataframe
base["sem_depuis_vente"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: calc_sem_depuis_vente(x.values))
).astype(int)

# probabilité qu'un article se vende à nouveau, sachant qu'il na pas été vendu depuis X sem
base["prob_reactivation"] = (
    base.groupby(["identifiant_article", "sem_depuis_vente"])["quantite_commandee"]
    .transform(lambda x: (x.shift(1) > 0).expanding().mean())
).fillna(0).round(3)
#shift(1) permet de regarder la valeur précédente (S-1) pour éviter d’utiliser le futur.

# ── NOUVELLES features v5 : EWM, seuil adaptatif, quantiles, momentum ─────────
print("   → Features v5 (EWM, quantiles, momentum, seuil adaptatif)...")

# BLOC A : EWM — moyenne ponderee exponentielle
base["ewm_4s"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: x.shift(1).ewm(span=4, adjust=False).mean())
).round(1).fillna(0)

base["ewm_12s"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: x.shift(1).ewm(span=12, adjust=False).mean())
).round(1).fillna(0)

base["ratio_ewm_court_long"] = (
    base["ewm_4s"] / base["ewm_12s"].replace(0, np.nan)
).fillna(1.0).clip(0.1, 10).round(3)

# BLOC B : Seuil pic adaptatif par article (percentile 85, plancher 200)
_seuil_pic_art = (
    base[base["quantite_commandee"] > 0]
    .groupby("identifiant_article")["quantite_commandee"]
    .quantile(0.85)
    .rename("seuil_pic_article")
    .reset_index()
)
_seuil_pic_art["seuil_pic_article"] = _seuil_pic_art["seuil_pic_article"].clip(lower=200)
base = base.merge(_seuil_pic_art, on="identifiant_article", how="left")
base["seuil_pic_article"] = base["seuil_pic_article"].fillna(500)

# BLOC C : Quantiles historiques par semaine_annee x article
base["q25_semaine_historique"] = (
    base.groupby(["identifiant_article", "semaine_annee"])["quantite_commandee"]
    .transform(lambda x: x.shift(1).expanding().quantile(0.25))
).round(1).fillna(0)

base["q75_semaine_historique"] = (
    base.groupby(["identifiant_article", "semaine_annee"])["quantite_commandee"]
    .transform(lambda x: x.shift(1).expanding().quantile(0.75))
).round(1).fillna(0)

base["iqr_semaine_historique"] = (
    base["q75_semaine_historique"] - base["q25_semaine_historique"]
).round(1)

# BLOC E : Lag 52 semaines + ratio YoY
base["lag_52_semaines"] = (
    base.groupby("identifiant_article")["quantite_commandee"].shift(52)
).fillna(0).round(1)

base["ratio_yoy_semaine"] = (
    base["lag_1_semaine"] / base["lag_52_semaines"].replace(0, np.nan)
).fillna(1.0).clip(0.1, 10).round(3)

# BLOC F : Saisonnalite mensuelle par article
base["moy_article_mois"] = (
    base.groupby(["identifiant_article", "mois"])["quantite_commandee"]
    .transform(lambda x: x.shift(1).expanding().mean())
).round(1).fillna(0)

_moy_glob_art = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
base["ratio_mois_vs_annuel"] = (
    base["moy_article_mois"] / _moy_glob_art.replace(0, np.nan)
).fillna(1.0).clip(0.1, 5).round(3)

print("   → 12 nouvelles features v5 ajoutees")

# ── Features saisonnières
print("   → Features saisonnières...")

base["lag_1_an"] = base.groupby(
    ["identifiant_article","semaine_annee"])["quantite_commandee"].shift(1)
base["lag_2_ans"] = base.groupby(
    ["identifiant_article","semaine_annee"])["quantite_commandee"].shift(2)

base["moy_semaine_historique"] = base.groupby(
    ["identifiant_article","semaine_annee"])["quantite_commandee"].transform(
    lambda x: x.shift(1).expanding().mean()
)

moy_globale = base.groupby("identifiant_article")["quantite_commandee"].transform("mean")
base["ratio_semaine_vs_moy"] = (
    base["moy_semaine_historique"] / moy_globale.replace(0, 1)
).round(3)
#ttes les val 0 seront remplacer par 1

# BLOC D : Momentum et acceleration
base["momentum_demande"] = (
    base["lag_1_semaine"] / base["moy_semaine_historique"].replace(0, np.nan)
).fillna(1.0).clip(0, 10).round(3)

base["acceleration_demande"] = (
    (base["lag_1_semaine"] - base["lag_2_semaines"])
    / base["lag_2_semaines"].replace(0, np.nan)
).fillna(0).clip(-2, 2).round(3)

# ── Tendance annuelle
print("   → Tendance annuelle...")

moy_annuelle = base.groupby(
    ["identifiant_article","annee"])["quantite_commandee"].transform("mean")
base["moy_annuelle"] = moy_annuelle

base["moy_annuelle_precedente"] = base.groupby(
    "identifiant_article")["moy_annuelle"].transform(lambda x: x.shift(52))

base["tendance_annuelle"] = (
    base["moy_annuelle"] / base["moy_annuelle_precedente"].replace(0, np.nan)#eviter divison par0
).fillna(1.0).round(3)#si on n'a pas d'info on met 1

base["delta_annuel"] = (
    base["moy_annuelle"] - base["moy_annuelle_precedente"].fillna(base["moy_annuelle"])
).round(1)#si la val de lannée prec nexiste pas on remplace par la val de lannée actuelle

base = base.drop(columns=["moy_annuelle","moy_annuelle_precedente"], errors="ignore")

# ─────────────────────────────────────────────
# 6. FEATURES LIVRAISON (Shipment)
# ─────────────────────────────────────────────

print("🚚 Features livraison...")

qty_ship_col = "ShippedQty" if "ShippedQty" in shipment.columns else "ShipmentQty"

agg_ship = shipment.groupby(["InventoryID","semaine"]).agg(
    quantite_expediee=(qty_ship_col, "sum"),
).reset_index()

# Type est maintenant une colonne binaire 0/1
# (0=Shipment client, 1=Transfer inter-entrepôts).
# On calcule directement ratio_transfert = mean(Type) par semaine.

if "Type" in shipment.columns:
    agg_tr = shipment.groupby(["InventoryID","semaine"])["Type"].mean().reset_index()
    agg_tr.columns = ["InventoryID","semaine","ratio_transfert"]
    agg_ship = agg_ship.merge(agg_tr, on=["InventoryID","semaine"], how="left")
else:
    agg_ship["ratio_transfert"] = 0.0

agg_ship = agg_ship.rename(columns={"InventoryID":"identifiant_article"})
base = base.merge(agg_ship, on=["identifiant_article","semaine"], how="left")
base["quantite_expediee"] = base["quantite_expediee"].fillna(0)
base["ratio_transfert"]   = base["ratio_transfert"].fillna(0)

base["taux_service"] = np.where(
    base["quantite_commandee"] > 0,
    (base["quantite_expediee"] / base["quantite_commandee"]).clip(0, 1),#forcer les val dans une plage 0 1
    np.nan
)

# ─────────────────────────────────────────────
# 7. FEATURES APPROVISIONNEMENT (PO + PR)
# ─────────────────────────────────────────────

print("🛒 Features approvisionnement...")

po_qty_col = "OrderQty"

agg_po = po.groupby(["InventoryID","semaine"]).agg(
    quantite_commande_fournisseur=(po_qty_col, "sum"),
).reset_index()

# on cherche ttes les col qui contiennent open et status
po_open_cols = [c for c in po.columns if "Open" in c and "Status" in c]
if po_open_cols:
    #pour chaque pd et chaque sem on calcule la qté totale des commandes ouvertes
    agg_open = po.groupby(["InventoryID","semaine"]).apply(
        lambda x: (x[po_open_cols[0]] * x[po_qty_col]).sum()
    ).reset_index()
    agg_open.columns = ["InventoryID","semaine","commandes_fournisseur_ouvertes"]
    agg_po = agg_po.merge(agg_open, on=["InventoryID","semaine"], how="left")
else:
    agg_po["commandes_fournisseur_ouvertes"] = 0

agg_po = agg_po.rename(columns={"InventoryID":"identifiant_article"})
base = base.merge(agg_po, on=["identifiant_article","semaine"], how="left")
base["quantite_commande_fournisseur"]  = base["quantite_commande_fournisseur"].fillna(0)
base["commandes_fournisseur_ouvertes"] = base["commandes_fournisseur_ouvertes"].fillna(0)

agg_pr = pr.groupby(["InventoryID","semaine"]).agg(
    quantite_recue=("TotalQty","sum"),
).reset_index().rename(columns={"InventoryID":"identifiant_article"})
base = base.merge(agg_pr, on=["identifiant_article","semaine"], how="left")
base["quantite_recue"] = base["quantite_recue"].fillna(0)

base["taux_reception_fournisseur"] = np.where(
    base["quantite_commande_fournisseur"] > 0,
    (base["quantite_recue"] / base["quantite_commande_fournisseur"]).clip(0, 1),
    np.nan
)

# ─────────────────────────────────────────────
# 8. FEATURES ARTICLE (StockItem)
# ─────────────────────────────────────────────

print("🏷️  Jointure StockItem...")

stock_cols_keep = ["InventoryID"]

# Colonnes numériques statiques
for col in ["DefaultPrice","LastCost","QtyOnHand","margin_rate","price_zero_flag","ItemStatus"]:
    if col in stock.columns:
        stock_cols_keep.append(col)

#on sinteresse aux col qui commencent par ces mots
ohe_prefixes = ("ItemType_", "ItemClass_", "DefaultWarehouseID_")
#on garde seule les col qui commencent par un des prefixes ohe
#et qui sont bien des col num/binaires
ohe_cols_stock = [
    c for c in stock.columns
    if c.startswith(ohe_prefixes)
    and stock[c].dtype in [np.uint8, np.int8, np.int64, bool, np.bool_]
]
#on ajoute ces col à la liste finale des var utiles
stock_cols_keep += ohe_cols_stock

stock_feat = stock[stock_cols_keep].copy()

# Convertir les colonnes OHE uint8/bool en int pour cohérence
for col in ohe_cols_stock:
    stock_feat[col] = stock_feat[col].astype(int)

# garder seul les col qui existent dans stock_feat
num_sc  = [c for c in ["DefaultPrice","LastCost","QtyOnHand","margin_rate",
                        "price_zero_flag","ItemStatus"]
           if c in stock_feat.columns]
#pour tt les col num on prend la moy
agg_dict = {c: "mean" for c in num_sc}
#ajout des col ohe et on fait max(est-ce que cette catégorie apparaît 
# au moins une fois ? [0, 0, 1, 0] → max = 1 [0, 0, 0] → max = 0)
agg_dict.update({c: "max" for c in ohe_cols_stock if c in stock_feat.columns})
stock_feat = stock_feat.groupby("InventoryID", as_index=False).agg(agg_dict)
####### comme si je transforme mes donn stock en une seule ligne par pd avec des moy(prix
###### stock) et des indic (ohe)
# Renommage lisible
rename_map = {
    "DefaultPrice":  "prix_vente",
    "LastCost":      "cout_achat",
    "QtyOnHand":     "stock_disponible",
    "margin_rate":   "taux_marge",
    "price_zero_flag": "flag_prix_zero",
    "ItemStatus":    "statut_article_ordinal",
}
for col in ohe_cols_stock:
    rename_map[col] = (
        col.replace("ItemType_",          "type_article_")
           .replace("ItemClass_",         "classe_article_")
           .replace("DefaultWarehouseID_","entrepot_")
           .replace(" ", "_")
           .lower()
    )
stock_feat = stock_feat.rename(columns=rename_map)
stock_feat = stock_feat.rename(columns={"InventoryID": "identifiant_article"})
assert stock_feat["identifiant_article"].is_unique

base = base.merge(stock_feat, on="identifiant_article", how="left")

# ─────────────────────────────────────────────
# 9. VARIABLES CIBLES
# ─────────────────────────────────────────────

print("🎯 Calcul des cibles...")

base = base.sort_values(["identifiant_article","semaine"])

base["cible_demande_1_semaine"] = (
    base.groupby("identifiant_article")["quantite_commandee"].shift(-1)
)
base["cible_demande_4_semaines"] = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform(lambda x: x.shift(-1).rolling(4, min_periods=1).sum())
)
base["cible_rupture_stock"] = (
    (base["quantite_commandee"] > 0) & (base["taux_service"] < 0.8)
).astype(int)

n_rup = base["cible_rupture_stock"].sum()
print(f"  → Ruptures : {n_rup}/{len(base)} ({n_rup/len(base)*100:.1f}%)")

# ── Cible surstock ───────────────────────────────────────────────────────────
SEUIL_SURSTOCK = 8  # semaines de couverture au-delà desquelles on parle de surstock

demande_moy_hebdo = (
    base.groupby("identifiant_article")["quantite_commandee"]
    .transform("mean")
)
# Couverture en semaines = stock_disponible / demande_moy_hebdo
# Si demande nulle → couverture infinie, mais pas de surstock actionnable → NaN
if "stock_disponible" in base.columns:
    base["couverture_semaines"] = np.where(
        demande_moy_hebdo > 0,
        base["stock_disponible"] / demande_moy_hebdo,
        np.nan
    ).round(1)
    base["cible_surstock"] = (
        base["couverture_semaines"] > SEUIL_SURSTOCK
    ).astype(int)
else:
    # Fallback si stock_disponible absent : utiliser quantite_recue >> demande
    base["couverture_semaines"] = np.nan
    base["cible_surstock"] = (
        (base["quantite_recue"] > SEUIL_SURSTOCK * demande_moy_hebdo)
        & (demande_moy_hebdo > 0)
    ).astype(int)

n_sur = base["cible_surstock"].sum()
print(f"  → Surstock (couverture > {SEUIL_SURSTOCK} semaines) : "
      f"{n_sur}/{len(base)} ({n_sur/len(base)*100:.1f}%)")

# ── Cible mauvaise anticipation ──────────────────────────────────────────────
SEUIL_RATIO_ANTICIPATION = 2.0  # commande fourn > N× demande réelle

commande_fourn_active = base["quantite_commande_fournisseur"] > 0

# Sur-commande : commande > 2× demande (demande > 0)
sur_commande = (
    commande_fourn_active
    & (base["quantite_commandee"] > 0)
    & (base["quantite_commande_fournisseur"] > SEUIL_RATIO_ANTICIPATION * base["quantite_commandee"])
)
# Commande inutile : commande > 0 alors que demande = 0
commande_sans_demande = (
    commande_fourn_active
    & (base["quantite_commandee"] == 0)
)
#fusion logique or une seule des 2 suffit
base["cible_mauvaise_anticipation"] = (
    sur_commande | commande_sans_demande
).astype(int)

n_ant = base["cible_mauvaise_anticipation"].sum()
n_sur_cmd = sur_commande.sum()
n_sans_dem = commande_sans_demande.sum()
print(f"  → Mauvaise anticipation : {n_ant}/{len(base)} ({n_ant/len(base)*100:.1f}%)")
print(f"     dont sur-commande (fourn > 2× demande) : {n_sur_cmd}")
print(f"     dont commande sans demande (demande=0)  : {n_sans_dem}")

# ── Cible anomalie stock = surstock OU mauvaise anticipation ─────────────────
# Cible binaire unique regroupant les deux situations de mauvaise gestion stock.
# Séparée de cible_rupture_stock car les causes et actions correctives diffèrent :
#   rupture      → réapprovisionner d'urgence
#   anomalie     → réduire les commandes / écouler le stock
base["cible_anomalie_stock"] = (
    (base["cible_surstock"] == 1) | (base["cible_mauvaise_anticipation"] == 1)
).astype(int)

n_ano = base["cible_anomalie_stock"].sum()
print(f"  → Anomalie stock (surstock OU mauvaise anticipation) : "
      f"{n_ano}/{len(base)} ({n_ano/len(base)*100:.1f}%)")


# Résumé des 3 composantes internes
print(f"\n  Détail des composantes :")
print(f"    cible_surstock               : {base['cible_surstock'].sum()} semaines")
print(f"    cible_mauvaise_anticipation  : {base['cible_mauvaise_anticipation'].sum()} semaines")
print(f"    cible_anomalie_stock (union) : {n_ano} semaines")

# ─────────────────────────────────────────────
# 10. NETTOYAGE FINAL
# ─────────────────────────────────────────────

print("🧹 Nettoyage final...")

base = base.rename(columns={"semaine":"date_debut_semaine"})

before = len(base)
base = base.drop_duplicates(subset=["identifiant_article","date_debut_semaine"])
print(f"  → {before-len(base)} doublon(s) supprimé(s)")

constant_cols = [c for c in base.columns if base[c].nunique() == 1]
if constant_cols:
    base = base.drop(columns=constant_cols)
    print(f"  → Colonnes constantes supprimées : {constant_cols}")

if "taux_service" in base.columns:
    base["taux_service"] = base["taux_service"].fillna(1.0)

base = base.dropna(subset=["lag_1_semaine","lag_2_semaines"])
base = base.dropna(subset=["cible_demande_1_semaine"])

for col in ["taux_zeros_historique","semaine_active_lag1","semaine_active_lag2",
            "taux_activite_12s","lag_1_an","lag_2_ans","moy_semaine_historique",
            "delta_annuel","std_glissante_4s","coef_variation_4s",
            "lag_3_semaines","moy_actives_8s","sem_depuis_vente","prob_reactivation"]:
    if col in base.columns:
        base[col] = base[col].fillna(0)

for col in ["ratio_semaine_vs_moy","tendance_annuelle"]:
    if col in base.columns:
        base[col] = base[col].fillna(1.0)

num_cols = base.select_dtypes(include=[np.number]).columns
base[num_cols] = base[num_cols].fillna(0)

# ─────────────────────────────────────────────
# 11. RAPPORT FINAL
# ─────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  ✅ BASE ML FINALE")
print(f"{'='*55}")
print(f"  Dimensions     : {base.shape}")
print(f"  Articles       : {base['identifiant_article'].nunique()}")
print(f"  Période        : {base['date_debut_semaine'].min().date()} → {base['date_debut_semaine'].max().date()}")
print(f"  Valeurs nulles : {base.isnull().sum().sum()}")
print(f"\n  Colonnes ({len(base.columns)}) :")
for col in base.columns:
    print(f"    • {col}")

# ─────────────────────────────────────────────
# 12. EXPORT
# ─────────────────────────────────────────────

print("\n Export...")
base.to_excel("base_ml_finale.xlsx", index=False, engine="openpyxl")
print("  ✅ base_ml_finale.xlsx")
print("\n Feature engineering terminé !")