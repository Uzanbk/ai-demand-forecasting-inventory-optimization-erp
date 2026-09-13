"""
=============================================================
 PARTIE 1 — Régression (2 modèles : Normal + Pic)
            Perte unique : MAE | Métriques : MAE + wMAPE
 PARTIE 2 — Classification Rupture de stock
            Perte unique : Binary Logloss | Métriques : Accuracy + F1
 PARTIE 3 — Classification Anomalie stock (NOUVEAU)
            Surstock (couverture > 8 semaines) OU mauvaise anticipation
            (commande fournisseur sans demande réelle)
            Perte unique : Binary Logloss
 PARTIE 4 — Rapport HTML unifié (tous les graphiques)

 f1score=indicateur de performance qui combine la précision et le rappel 
 precision_score=rapport entre le nombre de vrais positifs et le nombre de faux positifs
 classification_report= rapport presentant les principales métriques de classification
 recall_score: métrique de class qui mesure capacité du md à identifier ts les cas positifs parmi lensemble des echantillons posit
roc_auc_score= donne un score unique qui mesure la qualité globale du modèle
roc_curve= calcule les points pour tracer la courbe(tpr,fpr selon seuils)
accuracy=proportion de bonnes prédictions
log_loss=mesure à quel point les probabilités sont mauvaises
RandomizedSearchCV= teste aléatoirement plusieurs combinaisons et garde la meilleure
TimeSeriesSplit= tester le modèle sur des données dans le bon ordre(sans mélange futur et passé)

=============================================================
=============================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    classification_report,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, accuracy_score,
    log_loss
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import matplotlib
matplotlib.use("Agg")#n'utilise pas l'affichage à l'écran 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker#controler l'affichage des graduations(ticks) sur les graphiques
#base64=transformer (img)en texte khtr bch namlou des graphes en matplot w bch nestockihom f fichier html
#os: sert à interagir avec le système(verifier si fich existe, creer des doss,gerer les chemins)
#io: stocker une image manipuler des donn en mémoire
#pickle: sauvegarder modele entrainé le recharger plus tard

import base64, io, os, pickle
import warnings
warnings.filterwarnings("ignore")#sert à masquer les avertissements dans python

# ════════════════════════════════════════════════════════════
#  transformer les graphes en images html et les stocker pour faire un rapport auto
# ════════════════════════════════════════════════════════════
#prend un graphe
def fig_to_b64(fig):
    #créer un faux fichier(espace mémoire pas un vrai fichier)
    buf = io.BytesIO()
    #transformer le graph en image png
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    #revenir au début repotionne le curseur
    buf.seek(0)
    #convertir en texte
    return base64.b64encode(buf.read()).decode()

GRAPHIQUES_HTML = []   # liste de tuples (titre, b64_img, description)
#sert à ajouter un graph dans la liste 
def ajouter_graphique(titre, fig, description=""):
    #stocker titre image et description 
    GRAPHIQUES_HTML.append((titre, fig_to_b64(fig), description))
    #ferme le graphe pour éviter surcharge mémoire
    plt.close(fig)

# ════════════════════════════════════════════════════════════
#  1. CHARGEMENT & FILTRES
# ════════════════════════════════════════════════════════════

print("📂 Chargement de la base ML...")
df = pd.read_excel("base_ml_finale.xlsx")
df["date_debut_semaine"] = pd.to_datetime(df["date_debut_semaine"])
df = df.sort_values(["identifiant_article", "date_debut_semaine"]).reset_index(drop=True)
print(f"  → {len(df)} lignes | {df['identifiant_article'].nunique()} articles")

df = df[df["identifiant_article"] != "EXPEDIRECTE"]
#calcul % des zéros pour chaque article
taux_zeros = df.groupby("identifiant_article")["quantite_commandee"].apply(
    lambda x: (x == 0).mean()
)
articles_inactifs = taux_zeros[taux_zeros >= 0.90].index.tolist()
df = df[~df["identifiant_article"].isin(articles_inactifs)]
print(f"  → {len(articles_inactifs)} articles exclus (≥90% zéros)")
print(f"  → {df['identifiant_article'].nunique()} articles | {len(df)} lignes")

# Feature temporelle : nb semaines depuis le début du dataset
# Permet au modèle d'apprendre la tendance de croissance sans couper les données
df["semaines_depuis_debut"] = (
    (df["date_debut_semaine"] - df["date_debut_semaine"].min()).dt.days / 7
).astype(int)

# Transformations log pour rendre les données plus stables (très grandes valeurs)
cols_log = [
    "quantite_expediee", "quantite_recue", "quantite_commande_fournisseur",
    "lag_1_semaine", "lag_2_semaines", "lag_4_semaines", "lag_8_semaines",
    "moyenne_glissante_4s", "moyenne_glissante_12s",
    "lag_1_an", "lag_2_ans", "moy_semaine_historique", "delta_annuel",
    # NOUVELLES v5
    "ewm_4s", "ewm_12s", "moy_article_mois",
    "lag_52_semaines", "q25_semaine_historique", "q75_semaine_historique",
]
for col in cols_log:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])

SEUIL_PIC = 500
df["est_pic"] = (df["cible_demande_1_semaine"] >= SEUIL_PIC).astype(int)#astype transforme true false en 0 1

# ════════════════════════════════════════════════════════════
#  2. FEATURES & SPLIT
# ════════════════════════════════════════════════════════════

EXCLURE_BASE = [
    "identifiant_article", "date_debut_semaine",
    # cibles régression
    "cible_demande_1_semaine", "cible_demande_4_semaines",
    # cibles classification — toutes exclues des features
    "cible_rupture_stock",
    "cible_surstock",                  
    "cible_mauvaise_anticipation",     
    "couverture_semaines",             # variable intermédiaire, pas une feature
    "est_pic",
    # variables contemporaines (leakage)
    "quantite_commandee", "nb_commandes",
    "taux_annulation", "taux_rupture_commande", "taux_service",
    "quantite_expediee", "ratio_transfert", 
]
FEATURES = [c for c in df.columns if c not in EXCLURE_BASE]

###### Suppression des var trop similaires entre elles #####
#matrice de corré entre toutes les var numériques
corr_matrix = df[FEATURES].select_dtypes(include=[np.number]).corr().abs()

#garde seulement la moitié du tableau(évite de comparer 2fois les mm var)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

#si une var est corrélée à plus de 0.95 avec une autre elle est marquée pour supp
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

#on enlève ces var du modèle
FEATURES = [f for f in FEATURES if f not in to_drop]
print(f"\n→ {len(to_drop)} variables supprimées (corrélation > 0.95)")

# Ajouter nouvelles features 
nouvelles = [
    "lag_1_an", "lag_2_ans", "moy_semaine_historique", "ratio_semaine_vs_moy",
    "tendance_annuelle", "delta_annuel", "taux_zeros_historique",
    "semaine_active_lag1", "semaine_active_lag2", "taux_activite_12s",
    "taux_semaine_forte", "lead_time_hist_moy", "nb_pics_semaine",
    "std_glissante_4s", "coef_variation_4s",
    "statut_article_ordinal",
    "is_transfert_interne",
    # features v4 — demande intermittente
    "lag_3_semaines",
    "moy_actives_8s",
    "sem_depuis_vente",
    "prob_reactivation",
    # NOUVELLES features v5 — amelioration MAPE
    "ewm_4s",                    # EWM court terme (reactivite)
    "ewm_12s",                   # EWM long terme (stabilite)
    "ratio_ewm_court_long",      # tendance locale EWM
    "q25_semaine_historique",    # borne basse saisonnalite
    "q75_semaine_historique",    # borne haute saisonnalite
    "iqr_semaine_historique",    # variabilite saisonniere
    "momentum_demande",          # ratio lag1/moy_historique
    "acceleration_demande",      # variation lag1-lag2 en %
    "lag_52_semaines",           # meme semaine annee passee (exact)
    "ratio_yoy_semaine",         # tendance YoY locale
    "moy_article_mois",          # saisonnalite mensuelle par article
    "ratio_mois_vs_annuel",      # mois vs moyenne annuelle
    "seuil_pic_article",         # seuil pic adaptatif
    "semaines_depuis_debut",     # tendance temporelle globale
]
for f in nouvelles:
    if f in df.columns and f not in FEATURES:
        FEATURES.append(f)

DATE_COUPURE = pd.Timestamp("2024-10-01")
# Train complet depuis le début — toutes les données sont utilisées
# Le modèle apprend lui-même la tendance via semaines_depuis_debut
train = df[df["date_debut_semaine"] < DATE_COUPURE].copy()
test  = df[df["date_debut_semaine"] >= DATE_COUPURE].copy()

le = LabelEncoder()
train["article_encode"] = le.fit_transform(train["identifiant_article"])
#ici on utilise transform pas fit_transform(psq on doit garder le mm encodage
# pas train:abc12=0  et test= abc12=1 )
#fit apprendre les catégories, transform(applique le mm mapping)
test["article_encode"]  = le.transform(test["identifiant_article"])

FEATURES_ALL = FEATURES + ["article_encode"]
X_train = train[FEATURES_ALL]
X_test  = test[FEATURES_ALL]

print(f"\n📅 Split temporel :")
print(f"  Train : {train['date_debut_semaine'].min().date()} → "
      f"{train['date_debut_semaine'].max().date()} ({len(train)} lignes)")
print(f"  Test  : {test['date_debut_semaine'].min().date()} → "
      f"{test['date_debut_semaine'].max().date()} ({len(test)} lignes)")
print(f"  → {len(FEATURES_ALL)} features")

tscv = TimeSeriesSplit(n_splits=3)

# ════════════════════════════════════════════════════════════
#  FONCTIONS DE PERTE UNIQUES
# ════════════════════════════════════════════════════════════



def wmape(y_true, y_pred):
    """wMAPE = MAE / mean(réel) — métrique principale sur demande intermittente.
    Pondère par le volume : les grosses semaines comptent plus que les semaines à 3 unités. ∑[|yt​−y​p​|​/∑yt]×100"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if mask.sum() == 0 or y_true[mask].mean() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask])) / y_true[mask].mean() * 100


def evaluer_regression(nom, y_true, y_pred):
    """Evalue MAE + wMAPE (métrique principale sur demande intermittente)"""
    y_pred = np.maximum(y_pred, 0)
    y_true_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    mae   = mean_absolute_error(y_true_arr, y_pred)
    wmap_ = wmape(y_true_arr, y_pred)
    print(f"\n  {'─'*46}")
    print(f"   {nom}")
    print(f"  {'─'*46}")
    print(f"    Perte (MAE)           : {mae:.2f} unités")
    print(f"    wMAPE                 : {wmap_:.1f}%")
    return {"Modèle": nom, "MAE": round(mae, 2), "wMAPE": round(wmap_, 1)}


def evaluer_classification(nom, y_true, y_pred, y_proba):
    """
    Perte unique : Binary Loglos or binary cross-entropy (binary_logloss dans LightGBM)
    Métriques rapportées : Accuracy + F1-Score
    """
    #−1/n​∑​[yi​log(pi​)+(1−yi​)log(1−pi​)
    loss  = log_loss(y_true, y_proba)                #  PERTE UNIQUE CLASSIFICATION dont la sortie est une val de proba
    acc   = accuracy_score(y_true, y_pred)#predic correctes/ nbr tot
    f1    = f1_score(y_true, y_pred, zero_division=0)#2*(precision*reacall/precision+recall)
    rec   = recall_score(y_true, y_pred, zero_division=0)#tp/tp+fn(true pos:le nbr de cas posi correct prédits, false nega:le nbr de cas posi que le mdl a manqués)
    #tp+fn=nbr total de cas positifs  rappel: Parmi les vrais positifs réels, combien ont été détectés
    pr    = precision_score(y_true, y_pred, zero_division=0)#tp/tp+fp 
    #precision: Parmi les positifs prédits par le modèle, combien sont réellement vrais ?”
    #Area Under the ROC Curve
    auc   = roc_auc_score(y_true, y_proba)
    print(f"\n  {'─'*46}")
    print(f"   {nom}")
    print(f"  {'─'*46}")
    print(f"    Perte (Binary Logloss): {loss:.4f}")
    print(f"    Métrique (Accuracy)   : {acc*100:.1f}%")
    print(f"    Métrique (F1-Score)   : {f1:.3f}")
    print(f"    Rappel                : {rec:.3f} ({rec*100:.1f}% ruptures détectées)")
    print(f"    AUC-ROC               : {auc:.3f}")
    return {"Perte (Logloss)": round(loss, 4), "Accuracy": round(acc*100, 1),
            "F1": round(f1, 3), "Rappel": round(rec, 3), "AUC": round(auc, 3)}


resultats_reg = []

# ════════════════════════════════════════════════════════════
#  PARTIE 1 — RÉGRESSION QUANTITÉ (2 modèles : Normal + Pic)
# ════════════════════════════════════════════════════════════

print("\n\n" + "█"*55)
print("  PARTIE 1 — RÉGRESSION : Prévision quantité")
print("  Architecture : 2 modèles séparés (Normal + Pic)")
print("  Perte unique : MAE (regression_l1)")
print("█"*55)

CIBLE_REG   = "cible_demande_1_semaine"
y_train_reg = train[CIBLE_REG]
y_test_reg  = test[CIBLE_REG]
#Résultat création des masques bool(true/false)pour filtrer les données
mask_train_norm = train[CIBLE_REG] < SEUIL_PIC#selec les lign où la cib <SEUIL
mask_train_pic  = train[CIBLE_REG] >= SEUIL_PIC
mask_test_norm  = test[CIBLE_REG] < SEUIL_PIC
mask_test_pic   = test[CIBLE_REG] >= SEUIL_PIC
#var expli pour les cas normaux            #cible correspondante
X_train_norm = X_train[mask_train_norm]; y_train_norm = y_train_reg[mask_train_norm]
X_train_pic  = X_train[mask_train_pic];  y_train_pic  = y_train_reg[mask_train_pic]

print(f"\n  Répartition train : {mask_train_norm.sum()} normales ({mask_train_norm.mean()*100:.1f}%) "
      f"| {mask_train_pic.sum()} pics ({mask_train_pic.mean()*100:.1f}%)")
print(f"  Répartition test  : {mask_test_norm.sum()} normales ({mask_test_norm.mean()*100:.1f}%) "
      f"| {mask_test_pic.sum()} pics ({mask_test_pic.mean()*100:.1f}%)")

# ── Baseline
print("\n" + "="*55)
print("  1️⃣  BASELINE — Moyenne Mobile 4 semaines")
print("="*55)
#on prend les val des lags et on fait la moy ligne par ligne jobtiens prédic simple
cols_lag_ok = [c for c in ["lag_1_semaine", "lag_2_semaines",
               "lag_4_semaines", "lag_8_semaines"] if c in X_test.columns]
pred_base = X_test[cols_lag_ok].mean(axis=1).values
res = evaluer_regression("Baseline (Moy. Mobile)", y_test_reg, pred_base)
resultats_reg.append(res)

# ── Tuning hyperparamètres LightGBM global
print("\n" + "="*55)
print("  2⃣  PARAMS FIXES (issus du meilleur run)")
print("="*55)

# Meilleurs hyperparamètres connus — suppression du RandomizedSearchCV
# (trop lent sur 73 features ; early_stopping ajuste n_estimators auto)
best_lgb = {
    "num_leaves"       : 127,
    "max_depth"        : -1,
    "learning_rate"    : 0.03,
    "min_child_samples": 5,    # optimisé sur fenêtre 18 mois
    "subsample"        : 0.9,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.3,
    "reg_lambda"       : 1.0,
}
n_est_lgb = 1000
print(f"  → Params : {best_lgb}")

# Modèle global LightGBM avec early stopping
modele_lgb = lgb.LGBMRegressor(**best_lgb, objective="regression_l1",
    n_estimators=2000, random_state=42, verbose=-1, n_jobs=-1)
modele_lgb.fit(X_train, np.log1p(y_train_reg),
    eval_set=[(X_test, np.log1p(y_test_reg))],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
pred_lgb_global = np.expm1(modele_lgb.predict(X_test))
res_lgb = evaluer_regression("LightGBM global", y_test_reg, pred_lgb_global)
resultats_reg.append(res_lgb)
#créer tab des var les plus impor du mdl
#chaque fois la varest utilisée pour couper un arbre, la var gagne des points
#le mdl teste plus séparations possibles(exp: prix>10ou20ou30) et garde celle qui améliore le plus les prédict
importance_lgb = pd.DataFrame({
    "feature": FEATURES_ALL, "importance": modele_lgb.feature_importances_
}).sort_values("importance", ascending=False)

# ── Modèle Normal
print("\n" + "="*55)
print(f"  3️⃣  MODÈLE NORMAL — semaines < {SEUIL_PIC} unités")
print("="*55)
modele_normal = lgb.LGBMRegressor(**best_lgb, objective="regression_l1",
    n_estimators=min(n_est_lgb * 2, 1200), random_state=42, verbose=-1, n_jobs=-1)
modele_normal.fit(X_train_norm, np.log1p(y_train_norm),
    eval_set=[(X_test[mask_test_norm], np.log1p(y_test_reg[mask_test_norm]))],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
pred_normal_test = np.expm1(modele_normal.predict(X_test[mask_test_norm]))

# ── Modèle Pic
print("\n" + "="*55)
print(f"  4⃣  MODÈLE PIC — semaines ≥ {SEUIL_PIC} unités")
print("="*55)

# Params fixes pic — suppression RandomizedSearchCV
best_pic = {
    "num_leaves"       : 63,
    "max_depth"        : 10,
    "learning_rate"    : 0.03,  # plus lent → meilleure convergence
    "min_child_samples": 5,
    "subsample"        : 0.9,
    "colsample_bytree" : 0.7,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
}
n_est_pic = 800  # plus d'arbres avec lr plus bas
print(f"  → Params pic : {best_pic}")

modele_pic = lgb.LGBMRegressor(**best_pic, objective="regression_l1",
    n_estimators=1200, random_state=42, verbose=-1, n_jobs=-1)
if mask_test_pic.sum() > 0:
    modele_pic.fit(X_train_pic, np.log1p(y_train_pic),
        eval_set=[(X_test[mask_test_pic], np.log1p(y_test_reg[mask_test_pic]))],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    pred_pic_test = np.expm1(modele_pic.predict(X_test[mask_test_pic]))
    evaluer_regression(f"Modèle Pic (≥{SEUIL_PIC})", y_test_reg[mask_test_pic], pred_pic_test)
#quand mm le mdl entraine mais aucune prédic va retourner tab vide
else:
    modele_pic.fit(X_train_pic, np.log1p(y_train_pic))
    pred_pic_test = np.array([])


# Modèles fallback
modele_global_norm = lgb.LGBMRegressor(**best_lgb, objective="regression_l1",
                                        n_estimators=800, random_state=42, verbose=-1)
modele_global_norm.fit(X_train_norm, np.log1p(y_train_norm))
modele_global_pic = lgb.LGBMRegressor(**best_pic, objective="regression_l1",
                                       n_estimators=600, random_state=42, verbose=-1)
modele_global_pic.fit(X_train_pic, np.log1p(y_train_pic))
# ════════════════════════════════════════════════════════════
#  LIGHTGBM PAR ARTICLE — 2 modèles (Normal + Pic)
"""
```text
                           ┌──────────────────────────┐
                           │   pred_par_article = 0  │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────┐
                     │ Boucle sur chaque article       │
                     │ for article in unique()         │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                  ┌──────────────────────────────────────┐
                  │ Séparation des données               │
                  │                                      │
                  │ TRAIN : normal / pic                 │
                  │ TEST  : normal / pic                 │
                  └────────────────┬─────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────────┐
               │ Calcul semaines actives                    │
               │                                            │
               │ n_actives_norm = Σ(y>0)                    │
               │ n_actives_pic  = Σ(y>0)                    │
               └────────────────┬───────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
     ┌────────────────────┐          ┌─────────────────────┐
     │ SEMAINES NORMALES  │          │   SEMAINES PIC      │
     └─────────┬──────────┘          └──────────┬──────────┘
               │                                │
               ▼                                ▼
     ┌────────────────────┐          ┌─────────────────────┐
     │ Existe des lignes  │          │ Existe des lignes   │
     │ test normales ?    │          │ test pic ?          │
     └─────────┬──────────┘          └──────────┬──────────┘
               │ Oui                             │ Oui
               ▼                                 ▼

══════════════════════════════════════════════════════════════════
                 PARTIE SEMAINES NORMALES
══════════════════════════════════════════════════════════════════

               ▼
     ┌────────────────────┐
     │ n_actives_norm >=15│
     └─────────┬──────────┘
               │
      ┌────────┴────────┐
      │                 │
     Oui               Non
      │                 │
      ▼                 ▼

┌─────────────────┐   ┌────────────────────┐
│ Calcul du CV    │   │ n_actives_norm >=3 │
│ CV = std/mean   │   └─────────┬──────────┘
└────────┬────────┘             │
         │                      │
   ┌─────┴─────┐          ┌─────┴─────┐
   │           │          │           │
 CV>0.8      CV≤0.8      Oui         Non
   │           │          │           │
   ▼           ▼          ▼           ▼

┌──────────┐ ┌────────────────┐ ┌────────────────┐ ┌──────────┐
│ Médiane  │ │ LightGBM local │ │ Méthode        │ │ Pred=0   │
│ robuste  │ │ par article    │ │ statistique    │ │          │
└────┬─────┘ └──────┬─────────┘ └──────┬─────────┘ └────┬─────┘
     │               │                  │                 │
     ▼               ▼                  ▼                 ▼

┌────────────────────────────────────────────────────────────┐
│ prédiction =                                              │
│ médiane × taux_activité OU modèle ML                      │
└────────────────────────────────────────────────────────────┘


══════════════════════════════════════════════════════════════════
                     PARTIE SEMAINES PIC
══════════════════════════════════════════════════════════════════

               ▼
     ┌──────────────────┐
     │ n_actives_pic>=8 │
     └────────┬─────────┘
              │
      ┌───────┴───────┐
      │               │
     Oui             Non
      │               │
      ▼               ▼

┌────────────────┐  ┌──────────────────┐
│ LightGBM local │  │ Modèle global    │
│ spécial pics   │  │ spécial pics     │
└────────┬───────┘  └────────┬─────────┘
         │                   │
         └─────────┬─────────┘
                   ▼

      ┌─────────────────────────────┐
      │ Stockage dans               │
      │ pred_par_article            │
      └────────────┬────────────────┘
                   │
                   ▼

      ┌─────────────────────────────┐
      │ np.maximum(pred,0)          │
      │ suppression valeurs nég.    │
      └────────────┬────────────────┘
                   │
                   ▼

          ┌─────────────────────
          │ PRÉDICTIONS FINALES │
          └─────────────────────
```

"""
# ════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  5️⃣  LIGHTGBM PAR ARTICLE — 2 modèles (Normal + Pic)")
print("="*55)

pred_par_article = np.zeros(len(test))#création de tab rempli de 0
#chaque case une ligne du test

# Seuil : nb semaines actives minimum pour entraîner un modèle ML par article
# un article doit avoir au - 15 sem actives(dem>0) pour entrain un vrai mdl sinn pas assez de données
SEUIL_ACTIVES_ML = 15

#boucle sur chaque art indiv
for article in test["identifiant_article"].unique():
    mask_te          = test["identifiant_article"] == article#selec les lig du test de cet artc
    mask_te_norm_art = mask_te & mask_test_norm
    mask_te_pic_art  = mask_te & mask_test_pic
    mask_tr          = train["identifiant_article"] == article
    mask_tr_norm     = mask_tr & mask_train_norm #recup des données cibles
    mask_tr_pic      = mask_tr & mask_train_pic

    y_tr_norm = y_train_reg[mask_tr_norm]
    y_tr_pic  = y_train_reg[mask_tr_pic]#dema réelle
    X_tr_norm = X_train[mask_tr_norm]#features
    X_tr_pic  = X_train[mask_tr_pic]

    # Semaines actives (demande > 0) 
    n_actives_norm = int((y_tr_norm > 0).sum())
    n_actives_pic  = int((y_tr_pic  > 0).sum())

    if mask_te_norm_art.sum() > 0:#est ce que ya des sem norma à predire pour cet article
        X_te_n = X_test[mask_te_norm_art]#sert à récuperer les lig normales de cet art dans le test

        if n_actives_norm >= SEUIL_ACTIVES_ML:#cas bq de données 
            # Vérifier le coef de variation : si demande trop irrégulière ou stable
            actives_vals_n = y_tr_norm[y_tr_norm > 0]#garde les sem où il ya des ventes
            #sigma(écart-type)/moy
            cv = float(actives_vals_n.std() / actives_vals_n.mean()) if actives_vals_n.mean() > 0 else 0
            if cv > 0.8:#cas cv trop elevé md utilise médiane
                #moins sensible aux pics extremes
                mediane_active = float(actives_vals_n.median())
                #calcul taux dactivité: prop de periodes où il ya eu des ventes
                taux_activ = float((y_tr_norm > 0).mean())
                #si on a une var plus précise on lutilise
                if "taux_activite_12s" in test.columns:
                    #on prend la médiane on la multiplie par le taux dactivité de chaque ligne 
                    #plus un article est actif récemment plus la prédiction augmente
                    taux_semaine = test.loc[mask_te_norm_art, "taux_activite_12s"].values
                    pred_par_article[mask_te_norm_art.values] = mediane_active * taux_semaine
                #si pas de feat fine on utilise un taux global
                else:
                    pred_par_article[mask_te_norm_art.values] = mediane_active * taux_activ
            else:#kn cas adi md entraine un vrai lightgbm
                # Variance normale → modèle ML par article
                md_n = lgb.LGBMRegressor(**best_lgb, objective="regression_l1",
                                          n_estimators=600, random_state=42, verbose=-1)
                md_n.fit(X_tr_norm, np.log1p(y_tr_norm))
                pred_par_article[mask_te_norm_art.values] = np.expm1(md_n.predict(X_te_n))

        elif n_actives_norm >= 3:#cas peu de données:on évite le ml et on utilise une approche stat
            #on garde unique les sem où ya demande
            actives_vals = y_tr_norm[y_tr_norm > 0]
            mediane_active = float(actives_vals.median())
            taux_activ = float((y_tr_norm > 0).mean())
            # Pondérer par le taux d'activité récent de la semaine
            if "taux_activite_12s" in test.columns:
                taux_semaine = test.loc[mask_te_norm_art, "taux_activite_12s"].values
                pred_par_article[mask_te_norm_art.values] = mediane_active * taux_semaine
            else:
                pred_par_article[mask_te_norm_art.values] = mediane_active * taux_activ

        else:
            # Vraiment rien → prédire 0 (article quasi-inactif)
            pred_par_article[mask_te_norm_art.values] = 0.0

    if mask_te_pic_art.sum() > 0:
        X_te_p = X_test[mask_te_pic_art]
        if n_actives_pic >= 8:
            md_p = lgb.LGBMRegressor(**best_pic, objective="regression_l1",
                                      n_estimators=500, random_state=42, verbose=-1)
            md_p.fit(X_tr_pic, np.log1p(y_tr_pic))
            pred_par_article[mask_te_pic_art.values] = np.expm1(md_p.predict(X_te_p))
        else:
            pred_par_article[mask_te_pic_art.values] = np.expm1(
                modele_global_pic.predict(X_te_p))


pred_par_article = np.maximum(pred_par_article, 0)

res_art = evaluer_regression("LightGBM par article (2 modèles)", y_test_reg, pred_par_article)
resultats_reg.append(res_art)

# ── Comparaison
print("\n" + "="*55)
print("  COMPARAISON RÉGRESSION")
print("="*55)

df_res_reg = pd.DataFrame(resultats_reg)#Transforme la liste des résultats en tab
#creation du score global(eval équilibrée si ya bon mae mauvais wmape)
df_res_reg["Score"] = df_res_reg["MAE"].rank() + df_res_reg["wMAPE"].rank()
df_res_reg = df_res_reg.sort_values("Score")#tri des mdl
print(df_res_reg[["Modèle", "MAE", "wMAPE"]].to_string(index=False))

meilleur_reg = df_res_reg.iloc[0]["Modèle"]#selec auto du prem ligne donc meill mdl
#dic des prédic nom du mdl -> prédic correspondates
pred_map = {
    "LightGBM par article (2 modèles)": pred_par_article,
    "LightGBM global"                 : pred_lgb_global,
    "Baseline (Moy. Mobile)"          : pred_base,
}
pred_reg_fin = pred_map.get(meilleur_reg, pred_par_article)#recup des prédic finales du meill mdl
#recup la val du mae du mdl baseline 
mae_base = df_res_reg[df_res_reg["Modèle"].str.contains("Baseline")]["MAE"].values[0]
#calcul du gain par rapport à la baseline
gain = (mae_base - df_res_reg.iloc[0]["MAE"]) / mae_base * 100
print(f"\n  🏆 Meilleur modèle : {meilleur_reg}")
print(f"  📈 Gain MAE vs Baseline : +{gain:.1f}%")
#######  analyser les perfor du mdl par article ######
test_reg = test.copy()
#ajout des predic finales, supp des val négatives(devient 0), arrondi,conversion en entier
test_reg["prediction_reg"] = np.maximum(pred_reg_fin, 0).round().astype(int)
#calcul de lerreur absolue
test_reg["erreur_abs"]     = (test_reg[CIBLE_REG] - test_reg["prediction_reg"]).abs()
#pour chaque art on mesure dem reell moy, predic moy, erreur moy, et wmape
perf_art = test_reg.groupby("identifiant_article").agg(
    reel_moyen  =(CIBLE_REG, "mean"),
    pred_moyen  =("prediction_reg", "mean"),
    mae_article =("erreur_abs", "mean"),#err_abs(|yt-yp|)
).round(1).sort_values("mae_article", ascending=False)
#wmape=(mae/yt)*100 si reel_moy=0 on remplace par nan
perf_art["wmape_%"] = (perf_art["mae_article"] / perf_art["reel_moyen"].replace(0, np.nan) * 100).round(1)

# boucle manuelle pour calculer le wmape de chaque art, compter ses sem actives
#sa demande moy,puis construire un dataframe final
wmape_art = {}#dicti vide pour stocker les métriques de chaque art
for art in test_reg["identifiant_article"].unique():
    #masque selec les lignes de cet article
    mask_a = test_reg["identifiant_article"] == art
    #recup yt les dem réelles de cet article
    y_a = test_reg.loc[mask_a, CIBLE_REG].values
    #extarc des prédic
    p_a = test_reg.loc[mask_a, "prediction_reg"].values
    #appelle fonct calcul wmape
    wm = wmape(y_a, p_a)
    #comptage sem actives
    n_actif = (y_a > 0).sum()
    #stockage des métriques
    #si wmape est impossible à calculer :somme reelle=0,divi impo donc on met 999
    #pour signaler article problématique
    wmape_art[art] = {"wMAPE_%": round(wm, 1) if wm is not np.nan else 999,
                      "n_semaines_actives": int(n_actif),
                      "demande_moy": round(y_a[y_a > 0].mean(), 1) if n_actif > 0 else 0}
#convertit dic en tab, .T transpose le tab(avant: metr, art/ après: art/metr) avec tri decroi(du pire au meill)
df_wmape_art = pd.DataFrame(wmape_art).T.sort_values("wMAPE_%", ascending=False)
print("\n  wMAPE par article :")
print(df_wmape_art.to_string())

# Métriques métier pour vérifier si le mdl fonctionne differemment selon contexte dem

#mask_test_norm: filtre indique(true:sem normale, false:pas norm), test_reg[cib]:garde seul les sem où ya vente
#erreur_abs: val absolue(reel-predi)
mae_norm_v = test_reg.loc[mask_test_norm & (test_reg[CIBLE_REG] > 0), "erreur_abs"].mean()#erreur moy pendant les sem normales: garde les sem norm et actives
mae_pic_v  = test_reg.loc[mask_test_pic, "erreur_abs"].mean()
mape_norm  = wmape(test_reg.loc[mask_test_norm, CIBLE_REG].values,
                    test_reg.loc[mask_test_norm, "prediction_reg"].values)#% d'erreur sur sem normales
mape_pic   = wmape(test_reg.loc[mask_test_pic,  CIBLE_REG].values,
                    test_reg.loc[mask_test_pic,  "prediction_reg"].values)#% d'erreur sur sem pics

print(f"\n  Métriques détaillées du meilleur modèle :")
print(f"    wMAPE globale   : {wmape(test_reg[CIBLE_REG].values, test_reg['prediction_reg'].values):.1f}%")
print(f"    wMAPE normales  : {wmape(test_reg.loc[mask_test_norm, CIBLE_REG].values, test_reg.loc[mask_test_norm, 'prediction_reg'].values):.1f}%")
print(f"    wMAPE pics      : {wmape(test_reg.loc[mask_test_pic, CIBLE_REG].values, test_reg.loc[mask_test_pic, 'prediction_reg'].values):.1f}%")

###### combien de prédic sont acceptables
mask_actif_te = test_reg[CIBLE_REG] > 0#garde seul les sem actives
#partie gauche : recupère |yt-yp|    partie droite: recupere les vraies demandes
err_pct = test_reg.loc[mask_actif_te, "erreur_abs"] / test_reg.loc[mask_actif_te, CIBLE_REG] * 100
#transforme en booléens:exp(5,40,6-->true,false,true)
t20 = (err_pct <= 20).mean() * 100
t30 = (err_pct <= 30).mean() * 100

#####compare 3 mdl: baseline, lightgbm global

mape_base_norm = wmape(y_test_reg[mask_test_norm].values, pred_base[mask_test_norm])
mape_base_pic  = wmape(y_test_reg[mask_test_pic].values,  pred_base[mask_test_pic])
mape_lgb_norm  = wmape(y_test_reg[mask_test_norm].values, pred_lgb_global[mask_test_norm])
mape_lgb_pic   = wmape(y_test_reg[mask_test_pic].values,  pred_lgb_global[mask_test_pic])
mape_2m_norm   = wmape(y_test_reg[mask_test_norm].values, pred_par_article[mask_test_norm])
mape_2m_pic    = wmape(y_test_reg[mask_test_pic].values,  pred_par_article[mask_test_pic])

# ════════════════════════════════════════════════════════════
#  PARTIE 2 — CLASSIFICATION RUPTURE DE STOCK
# ════════════════════════════════════════════════════════════

print("\n\n" + "█"*55)
print("  PARTIE 2 — CLASSIFICATION RUPTURE DE STOCK")
print("  Perte unique : Binary Logloss")
print("  Métriques : Accuracy + F1-Score")
print("█"*55)

CIBLE_CLF   = "cible_rupture_stock"
y_train_clf = train[CIBLE_CLF]#recup colonne cible
y_test_clf  = test[CIBLE_CLF]
n_rup   = y_train_clf.sum()#nbr tot de ruptures
n_total = len(y_train_clf)#nbr tot lignes dans train
#sert à mesurer à quel point les classes sont déséquilibrées
#nbr_tot_non_rupt/nb_ruptures
ratio   = (n_total - n_rup) / max(n_rup, 1)
#tester plus combi de parametres pour trouver le meill modèle
grille_clf = {
    "num_leaves"       : [15, 31, 63],#combien de feuilles un arbre peut avoir
    "max_depth"        : [4, 6, 8],#profondeur maxi des arbres
    "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
    "n_estimators"     : [200, 400, 600],#nbr darbres construits
    "min_child_samples": [20, 30, 50],#nbr min dexmp dans une feuille
    "subsample"        : [0.6, 0.7, 0.8],#% des lignes utilisées à chq arbre
    "colsample_bytree" : [0.6, 0.7, 0.8],#% des var utilisées par arbre
    "reg_alpha"        : [0.1, 0.3, 0.5, 1.0],#régularisation L1: pénaliser les mdl trop complexes
    "reg_lambda"       : [0.1, 0.3, 0.5, 1.0],#rég L2: limite les poids trop elv, réduire overfit
}
tscv_clf = TimeSeriesSplit(n_splits=5)#valid croisée: ne melange pas les dates 
#passé predit futur: avec 5 split(mdl fait 5 entrain differents)

#obj=binary(mdl comprend que ya 2classes 0,1)
#random=tjs obtenir les mm résultats alea
#njobs= utilise tous les coeurs du proc
#le meill mdl sera choisi selon F1
#verbose -1: aucun msg du mdl sera affiché
search_clf = RandomizedSearchCV(
    lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss",  # perte unique : binary logloss
        scale_pos_weight=ratio, random_state=42, verbose=-1, n_jobs=-1
    ),
    grille_clf, n_iter=40, scoring="f1",
    cv=tscv_clf, random_state=42, n_jobs=-1, verbose=0
)
search_clf.fit(X_train, y_train_clf)
print(f"\n  → Meilleurs params : {search_clf.best_params_}")

#recup meill param(meilleure combinaison trouvée)
best_clf  = search_clf.best_params_.copy()
#recup n_estim w suppri de best_clf pour lutiliser (on double nbr darb dans mdl final)
n_est_clf = best_clf.pop("n_estimators", 400)
modele_clf = lgb.LGBMClassifier(
    **best_clf, objective="binary", metric="binary_logloss",
    scale_pos_weight=ratio, n_estimators=n_est_clf * 2,
    random_state=42, verbose=-1, n_jobs=-1
)
#entrainement du mdl avec xtrain ytrain
#eval(données de validation: donner un jeu de test pendant entrain) pour surveiller la perfo du mdl sur données qui ne voit pas
#early: si mdl ne sameliore plus pendant 50 iter alors il arrete auto lentrain
#log(period=-1): aucun affichage pendant lentrain
modele_clf.fit(
    X_train, y_train_clf,
    eval_set=[(X_test, y_test_clf)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
)
proba_test = modele_clf.predict_proba(X_test)[:, 1]#recuperer les proba de classe 1(rup)

# On a proba de test mais pour dire si une rup ou pas il faut un seuil
#on teste tous les seuils entre 0.1 et 0.79
seuils_fin = np.arange(0.1, 0.8, 0.01)
meilleur_seuil = 0.5; meilleur_f1 = 0.0
#tester chaque seuil possible
for s in seuils_fin:
    #transfor proba en classes: si proba>= s donc rup sinn 0
    pred_s = (proba_test >= s).astype(int)
    #condition metier: le mdl doit detecter au moins 90% des vraies rup
    if recall_score(y_test_clf, pred_s, zero_division=0) >= 0.90:
        #evaluer la qualité global
        f1_s = f1_score(y_test_clf, pred_s, zero_division=0)
        #si seuil est meilleur on le garde
        if f1_s > meilleur_f1:
            meilleur_f1 = f1_s; meilleur_seuil = s
#appliquer meilleur s trouvé et tester limpact de diff seuils sur F1
#si proba >= meill s donc 1(rupt)
pred_clf_fin = (proba_test >= meilleur_seuil).astype(int)
#création dune liste de seuils
seuils    = np.arange(0.2, 0.8, 0.05)
#calcul du F1 pour chaque seuil: convertit les proba en classes après calcul F1
f1_scores = [f1_score(y_test_clf, (proba_test >= s).astype(int), zero_division=0) for s in seuils]
#appelle au fonction qui calcul les indica
metriques_clf = evaluer_classification("LightGBM Classification", y_test_clf, pred_clf_fin, proba_test)

f1  = metriques_clf["F1"]
rec = metriques_clf["Rappel"]
auc = metriques_clf["AUC"]
acc = metriques_clf["Accuracy"]

print(f"\n  → Seuil optimal (rappel≥90%) : {meilleur_seuil:.2f}  (F1={meilleur_f1:.3f})")
print(f"\n{classification_report(y_test_clf, pred_clf_fin, target_names=['Pas rupture','Rupture'])}")

# Performances par article classification
test_clf_df = test.copy()
test_clf_df["proba_rupture"] = proba_test#ajout des resultats au mdl(proba predite)
test_clf_df["pred_rupture"]  = pred_clf_fin#decision finale
#creationn des indic derreur(le mdl predit une rup et la rup existe reellement)
test_clf_df["vrai_positif"]  = ((pred_clf_fin == 1) & (y_test_clf == 1)).astype(int)
#le modele na pas predit mais la rupture existe
test_clf_df["faux_negatif"]  = ((pred_clf_fin == 0) & (y_test_clf == 1)).astype(int)
#regroupe par article pour analyser perfor pd par pd
perf_clf_art = test_clf_df.groupby("identifiant_article").agg(
    ruptures_reelles  =(CIBLE_CLF, "sum"),
    ruptures_detectees=("vrai_positif", "sum"),
    manquees          =("faux_negatif", "sum"),
    proba_moy         =("proba_rupture", "mean"),
).round(2)
perf_clf_art["taux_detection_%"] = (
    perf_clf_art["ruptures_detectees"] /
    perf_clf_art["ruptures_reelles"].replace(0, np.nan) * 100
).round(1)

# ════════════════════════════════════════════════════════════
#  PARTIE 3 — CLASSIFICATION ANOMALIE STOCK
#  Surstock (stock_disponible > 8 sem. de demande moy.)
#  OU mauvaise anticipation (commande fourn. > 2× demande réelle)
#  Cible construite dans featengin_v3.py → cible_anomalie_stock
# ════════════════════════════════════════════════════════════

print("\n\n" + "█"*55)
print("  PARTIE 3 — ANOMALIE STOCK (Surstock + Mauvaise anticipation)")
print("  Seuil surstock        : couverture > 8 semaines")
print("  Seuil mauvaise antici.: commande fourn. > 2× demande réelle")
print("  Perte unique : Binary Logloss | Rappel cible ≥ 85%")
print("█"*55)

if "cible_anomalie_stock" not in df.columns:
    raise ValueError(
        "cible_anomalie_stock absente de base_ml_finale.xlsx.\n"
        "Relancer featengin_v3.py avant modelisation_v3.py."
    )

# Réutilise les mêmes splits train/test et X_train/X_test que les parties 1 & 2
CIBLE_ANO   = "cible_anomalie_stock"
y_train_ano = train[CIBLE_ANO]
y_test_ano  = test[CIBLE_ANO]
n_ano_tr    = y_train_ano.sum()
n_total_ano = len(y_train_ano)
ratio_ano   = (n_total_ano - n_ano_tr) / max(n_ano_tr, 1)

print(f"\n  Anomalies train : {n_ano_tr}/{n_total_ano} ({n_ano_tr/n_total_ano*100:.1f}%)")
print(f"  Anomalies test  : {y_test_ano.sum()}/{len(y_test_ano)} "
      f"({y_test_ano.mean()*100:.1f}%)")
for col, label in [("cible_surstock",            "surstock          "),
                   ("cible_mauvaise_anticipation","mauvaise anticip. ")]:
    if col in test.columns:
        print(f"    dont {label}: {test[col].sum()} cas test")

# Tuning hyperparamètres — même grille que classification rupture
grille_ano = {
    "num_leaves"       : [15, 31, 63],
    "max_depth"        : [4, 6, 8],
    "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
    "n_estimators"     : [200, 400, 600],
    "min_child_samples": [10, 20, 30],
    "subsample"        : [0.6, 0.7, 0.8],
    "colsample_bytree" : [0.6, 0.7, 0.8],
    "reg_alpha"        : [0.1, 0.3, 0.5, 1.0],
    "reg_lambda"       : [0.1, 0.3, 0.5, 1.0],
}
tscv_ano = TimeSeriesSplit(n_splits=5)

search_ano = RandomizedSearchCV(
    lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss",
        scale_pos_weight=ratio_ano, random_state=42, verbose=-1, n_jobs=-1
    ),
    grille_ano, n_iter=40, scoring="f1",
    cv=tscv_ano, random_state=42, n_jobs=-1, verbose=0
)
search_ano.fit(X_train, y_train_ano)
print(f"\n  → Meilleurs params : {search_ano.best_params_}")

best_ano  = search_ano.best_params_.copy()
n_est_ano = best_ano.pop("n_estimators", 400)
modele_ano = lgb.LGBMClassifier(
    **best_ano, objective="binary", metric="binary_logloss",
    scale_pos_weight=ratio_ano, n_estimators=n_est_ano * 2,
    random_state=42, verbose=-1, n_jobs=-1
)
modele_ano.fit(
    X_train, y_train_ano,
    eval_set=[(X_test, y_test_ano)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
)
proba_ano = modele_ano.predict_proba(X_test)[:, 1]

# Optimisation seuil — contrainte rappel ≥ 85%
# Moins strict que rupture (90%) : le surstock n'est pas une urgence immédiate.
# On tolère quelques faux négatifs pour limiter les fausses alertes inutiles.
seuils_ano_fin  = np.arange(0.1, 0.8, 0.01)
meilleur_seuil_ano = 0.5; meilleur_f1_ano = 0.0
for s in seuils_ano_fin:
    pred_s = (proba_ano >= s).astype(int)
    if recall_score(y_test_ano, pred_s, zero_division=0) >= 0.85:
        f1_s = f1_score(y_test_ano, pred_s, zero_division=0)
        if f1_s > meilleur_f1_ano:
            meilleur_f1_ano = f1_s; meilleur_seuil_ano = s

pred_ano_fin    = (proba_ano >= meilleur_seuil_ano).astype(int)
seuils_ano_plot = np.arange(0.2, 0.8, 0.05)
f1_scores_ano   = [
    f1_score(y_test_ano, (proba_ano >= s).astype(int), zero_division=0)
    for s in seuils_ano_plot
]

metriques_ano = evaluer_classification(
    "LightGBM Anomalie Stock", y_test_ano, pred_ano_fin, proba_ano
)
f1_ano  = metriques_ano["F1"]
rec_ano = metriques_ano["Rappel"]
auc_ano = metriques_ano["AUC"]
acc_ano = metriques_ano["Accuracy"]

print(f"\n  → Seuil optimal (rappel≥85%) : {meilleur_seuil_ano:.2f}  "
      f"(F1={meilleur_f1_ano:.3f})")
print(f"\n{classification_report(y_test_ano, pred_ano_fin, target_names=['Normal','Anomalie'])}")

# Performances par article — anomalie
test_ano_df = test.copy()
test_ano_df["proba_anomalie"]    = proba_ano
test_ano_df["pred_anomalie"]     = pred_ano_fin
test_ano_df["vrai_positif_ano"]  = (
    (pred_ano_fin == 1) & (y_test_ano == 1)
).astype(int)
test_ano_df["faux_negatif_ano"]  = (
    (pred_ano_fin == 0) & (y_test_ano == 1)
).astype(int)

perf_ano_art = test_ano_df.groupby("identifiant_article").agg(
    anomalies_reelles  =(CIBLE_ANO, "sum"),
    anomalies_detectees=("vrai_positif_ano", "sum"),
    manquees           =("faux_negatif_ano", "sum"),
    proba_moy          =("proba_anomalie", "mean"),
).round(2)
perf_ano_art["taux_detection_%"] = (
    perf_ano_art["anomalies_detectees"] /
    perf_ano_art["anomalies_reelles"].replace(0, np.nan) * 100
).round(1)

# ════════════════════════════════════════════════════════════
#  PARTIE 3 renommée PARTIE 4 — GRAPHIQUES → RAPPORT HTML
# ════════════════════════════════════════════════════════════

print("\n\n📊 Génération des graphiques pour le rapport HTML...")

PALETTE = {
    "bleu"   : "#2563eb",
    "vert"   : "#16a34a",
    "rouge"  : "#dc2626",
    "orange" : "#d97706",
    "violet" : "#7c3aed",
    "gris1"  : "#94a3b8",
    "gris2"  : "#64748b",
    "fond"   : "#f8fafc",
}

# ── G2 : Prévision top 3 articles(comparer ventes réelles et prédites par mdl)
top3 = perf_art.sort_values("reel_moyen", ascending=False).head(3).index.tolist()
fig_g2, axes_g2 = plt.subplots(3, 1, figsize=(14, 15))
fig_g2.suptitle(f"Prévision demande — {meilleur_reg}", fontsize=13, fontweight="bold")
#parcourir chaque subplot et article(du top 3)
for ax, article in zip(axes_g2, top3):
    #selec uniqu les lignes de cet article
    mask  = test_reg["identifiant_article"] == article
    #recup données: dates en axe X(SEMAINES)
    dates = test_reg.loc[mask, "date_debut_semaine"]
    #vraies ventes
    reel  = test_reg.loc[mask, CIBLE_REG].values
    #predic du mdl
    pred  = test_reg.loc[mask, "prediction_reg"].values
    #calcul du mae
    mae_a = mean_absolute_error(reel, pred)
    #courbe réelle (vraie demande en bleu avec points ronds)
    ax.plot(dates, reel, label="Réel", color=PALETTE["bleu"], lw=2, marker="o", ms=5)
    #courbe predite(predic du mdl en rouge pointillé)
    ax.plot(dates, pred, label="Prédiction", color=PALETTE["rouge"], lw=2, ls="--", marker="s", ms=5)
    #zone d'erreur:colore lespace entre reel et predit plus zone est grande plus erreur forte
    ax.fill_between(dates, reel, pred, alpha=0.1, color=PALETTE["rouge"])
    #ajout une ligne horizontale(seuil de forte demande/pic)
    ax.axhline(SEUIL_PIC, color=PALETTE["orange"], ls=":", lw=1.5, alpha=0.7,
               label=f"Seuil pic ({SEUIL_PIC})")
    #affiche identif artc, mae,wmape
    ax.set_title(f"{article}  |  MAE={mae_a:.1f}  wMAPE={wmape(reel,pred):.1f}%", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylabel("Quantité")
    #force laxe des quantités à afficher des entiers
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#ajoute le nom de l'axe horizontal
axes_g2[-1].set_xlabel("Semaine")
plt.tight_layout()
ajouter_graphique("Prévision demande — Top 3 articles", fig_g2,
    "Comparaison réel vs prédiction pour les 3 articles à plus forte demande")

# ── G3 : MAPE normal vs pic
fig_g3, ax_g3 = plt.subplots(figsize=(12, 5))
labels_s = [f"Baseline\nnormales", f"Baseline\npics",
            f"1 modèle\nnormales", f"1 modèle\npics",
            f"2 modèles\nnormales", f"2 modèles\npics"]
vals_s   = [mape_base_norm, mape_base_pic, mape_lgb_norm, mape_lgb_pic, mape_2m_norm, mape_2m_pic]
colors_s = [PALETTE["gris1"], PALETTE["gris2"], PALETTE["bleu"], "#1d4ed8", PALETTE["vert"], "#15803d"]
bars_g3  = ax_g3.bar(labels_s, vals_s, color=colors_s, alpha=0.85, edgecolor="white", width=0.6)
#afficher les val au dessus, boucle sur chaque barre
for bar, val in zip(bars_g3, vals_s):
    #evite erreur si val manqu
    if not np.isnan(val):
        #affiche val au dessus de chque barre
        ax_g3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")
ax_g3.set_title(f"wMAPE : impact séparation Normal / Pic (seuil={SEUIL_PIC})", fontsize=12, fontweight="bold")
ax_g3.set_ylabel("wMAPE % (moins = meilleur)"); ax_g3.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
ajouter_graphique("wMAPE Normal vs Pic", fig_g3,
    "Impact de l'architecture à 2 modèles sur la précision par segment")

# ── G4 : MAE par article
fig_g4, ax_g4 = plt.subplots(figsize=(12, 6))
perf_s   = perf_art.sort_values("mae_article")
colors_g4= [PALETTE["vert"] if v < 100 else PALETTE["orange"] if v < 300 else PALETTE["rouge"]
            for v in perf_s["mae_article"]]
ax_g4.barh(perf_s.index, perf_s["mae_article"], color=colors_g4, alpha=0.85)
#ajout ligne verticale mae moy globale du mdl
ax_g4.axvline(df_res_reg.iloc[0]["MAE"], color="black", ls="--", alpha=0.6,
              label=f"MAE globale={df_res_reg.iloc[0]['MAE']:.1f}")
ax_g4.set_title("MAE par article", fontsize=12, fontweight="bold")
ax_g4.set_xlabel("MAE (unités)"); ax_g4.legend(); ax_g4.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
ajouter_graphique("MAE par article", fig_g4,
    "Vert < 100, Orange 100-300, Rouge > 300 unités d'erreur")

# ── G5 : Comparaison MAE modèles
df_comp = df_res_reg[df_res_reg["Modèle"].isin(
    ["Baseline (Moy. Mobile)", "LightGBM global", "LightGBM par article (2 modèles)"]
)].copy()
fig_g5, ax_g5 = plt.subplots(figsize=(10, 5))
cols_g5  = [PALETTE["rouge"], PALETTE["bleu"], PALETTE["vert"]]
bars_g5  = ax_g5.bar(df_comp["Modèle"], df_comp["MAE"],
                     color=cols_g5[:len(df_comp)], alpha=0.85, edgecolor="white", width=0.5)
ax_g5.set_title("Comparaison MAE globale", fontsize=12, fontweight="bold")
ax_g5.set_ylabel("MAE — moins = meilleur"); ax_g5.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars_g5, df_comp["MAE"]):
    ax_g5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{val:.0f}", ha="center", fontsize=11, fontweight="bold")
plt.xticks(rotation=10, ha="right"); plt.tight_layout()
ajouter_graphique("Comparaison MAE — 3 modèles", fig_g5,
    f"Gain vs Baseline : +{gain:.1f}%")

# ── G6 : Importance features
fig_g6, ax_g6 = plt.subplots(figsize=(10, 7))
top15   = importance_lgb.head(15)
#generer une palette de couleurs à partir de colormap(rouge jaune vert)
colors6 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
#creation du graph horizontal([::-1]= inverse l'ordre sinn la feat la plus import serait en bas)
ax_g6.barh(top15["feature"][::-1], top15["importance"][::-1], color=colors6, alpha=0.85)
ax_g6.set_title("Top 15 features — LightGBM", fontsize=12, fontweight="bold")
ax_g6.set_xlabel("Importance (gain)"); ax_g6.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
ajouter_graphique("Importance des features", fig_g6,
    "Top 15 features selon le gain moyen LightGBM")

# ── G7 : Courbe ROC:graphe evaluant la perf dun mdl de class bin:elle trace le taux 
# de vrais positifs en fct du taux de faux positifs
#logique roc: plus le TPR est eleve->meill detecteur de rup
#plus FPR est faible -> moins de fausses alertes
#y_test_clf: les vraies classes, proba_test:les proba predites
fpr, tpr, _ = roc_curve(y_test_clf, proba_test)#calcul des points de la courbe roc
fig_g7, ax_g7 = plt.subplots(figsize=(8, 6))
#auc: aire sous courbe roc:plus auc proche de 1 meill mdl et separation entre rup et non rup
ax_g7.plot(fpr, tpr, color=PALETTE["vert"], lw=2, label=f"LightGBM (AUC={auc:.3f})")
#Ça dessine une droite allant du point (0,0) à (1,1) Sur une ROC, ça représente la performance d’un classifieur totalement aleat
ax_g7.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aléatoire (AUC=0.5)")
ax_g7.set_xlabel("Taux fausses alertes (FPR)")
ax_g7.set_ylabel("Taux détection (TPR)")
ax_g7.set_title("Courbe ROC — Détection rupture", fontsize=12, fontweight="bold")
ax_g7.legend(); ax_g7.grid(True, alpha=0.3); plt.tight_layout()
ajouter_graphique("Courbe ROC — Classification rupture", fig_g7,
    f"AUC-ROC = {auc:.3f} — Plus proche de 1 = meilleur discriminant")

# ── G8 : Optimisation seuil
fig_g8, ax_g8 = plt.subplots(figsize=(8, 4))
ax_g8.plot(seuils, f1_scores, color=PALETTE["bleu"], lw=2, marker="o", ms=5)
ax_g8.axvline(meilleur_seuil, color=PALETTE["rouge"], ls="--", lw=2,
              label=f"Seuil={meilleur_seuil:.2f} (F1={meilleur_f1:.3f})")
ax_g8.set_xlabel("Seuil de décision"); ax_g8.set_ylabel("F1-Score")
ax_g8.set_title("Optimisation du seuil — Rupture", fontsize=12, fontweight="bold")
ax_g8.legend(); ax_g8.grid(True, alpha=0.3); plt.tight_layout()
ajouter_graphique("Optimisation du seuil de classification", fig_g8,
    f"Seuil optimal = {meilleur_seuil:.2f} avec contrainte rappel ≥ 90%")

# ── G9 : ROC anomalie stock
fpr_ano, tpr_ano, _ = roc_curve(y_test_ano, proba_ano)
fig_g9, ax_g9 = plt.subplots(figsize=(8, 6))
ax_g9.plot(fpr_ano, tpr_ano, color=PALETTE["violet"], lw=2,
           label=f"LightGBM Anomalie (AUC={auc_ano:.3f})")
ax_g9.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aléatoire (AUC=0.5)")
ax_g9.set_xlabel("Taux fausses alertes (FPR)")
ax_g9.set_ylabel("Taux détection (TPR)")
ax_g9.set_title("Courbe ROC — Détection anomalie stock", fontsize=12, fontweight="bold")
ax_g9.legend(); ax_g9.grid(True, alpha=0.3); plt.tight_layout()
ajouter_graphique("Courbe ROC — Anomalie stock", fig_g9,
    f"AUC-ROC = {auc_ano:.3f} | Surstock + Mauvaise anticipation")

# ── G10 : Optimisation seuil anomalie
#le mdl reste parfait meme quand le seuil change
#la ligne est plate psq les proba très hautes et les cas normaux ont des proba très faibles
#lalgo cherche meill F1 avec rappel >= 85% et comme tous les seuils donnent F1=1 
#donc il prend le premier seuil valide
fig_g10, ax_g10 = plt.subplots(figsize=(8, 4))
ax_g10.plot(seuils_ano_plot, f1_scores_ano, color=PALETTE["violet"], lw=2, marker="o", ms=5)
ax_g10.axvline(meilleur_seuil_ano, color=PALETTE["rouge"], ls="--", lw=2,
               label=f"Seuil={meilleur_seuil_ano:.2f} (F1={meilleur_f1_ano:.3f})")
ax_g10.set_xlabel("Seuil de décision"); ax_g10.set_ylabel("F1-Score")
ax_g10.set_title("Optimisation du seuil — Anomalie stock", fontsize=12, fontweight="bold")
ax_g10.legend(); ax_g10.grid(True, alpha=0.3); plt.tight_layout()
ajouter_graphique("Optimisation du seuil — Anomalie stock", fig_g10,
    f"Seuil optimal = {meilleur_seuil_ano:.2f} avec contrainte rappel ≥ 85%")

# ── G11 : Répartition des types d'anomalies par article
if "cible_surstock" in test_ano_df.columns and "cible_mauvaise_anticipation" in test_ano_df.columns:
    #regrouper les données par article puis additionner les 1
    counts_sur = test_ano_df.groupby("identifiant_article")["cible_surstock"].sum()
    counts_ant = test_ano_df.groupby("identifiant_article")["cible_mauvaise_anticipation"].sum()
    #creation du tab final 
    df_ano_bar = pd.DataFrame({"Surstock": counts_sur, "Mauvaise anticipation": counts_ant}).fillna(0)
    #supp des articles sans anomalie(garder seul les art ayant au moins une anomalie)
    #on prend les art avec plus de surstock puis les 15 premiers
    df_ano_bar = df_ano_bar[(df_ano_bar > 0).any(axis=1)].sort_values(
        "Surstock", ascending=False).head(15)

    fig_g11, ax_g11 = plt.subplots(figsize=(12, 6))
    #crée les pos des barres pour placer les articles
    x_pos = np.arange(len(df_ano_bar))
    #largeur des barres
    w = 0.4
    #les barres orange =surstock   (-w/2, +w/2):pour mettre les 2 barres côte à côte pour le mm article
    ax_g11.bar(x_pos - w/2, df_ano_bar["Surstock"],w, label="Surstock", color=PALETTE["orange"], alpha=0.85)
    ax_g11.bar(x_pos + w/2, df_ano_bar["Mauvaise anticipation"], w, label="Mauvaise anticipation", color=PALETTE["violet"], alpha=0.85)
    ax_g11.set_xticks(x_pos); ax_g11.set_xticklabels(df_ano_bar.index, rotation=35, ha="right")
    ax_g11.set_title("Répartition des anomalies par article (test)", fontsize=12, fontweight="bold")
    ax_g11.set_ylabel("Nombre de semaines anomalies"); ax_g11.legend(); ax_g11.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ajouter_graphique("Anomalies par article — Surstock vs Mauvaise anticipation", fig_g11,
        "Top 15 articles par nombre de semaines en surstock ou avec mauvaise anticipation")

# ── G12 : Distribution erreurs prédiction
fig_g12, axes_g12 = plt.subplots(1, 2, figsize=(13, 5))
#reel - predit
residus = test_reg[CIBLE_REG] - test_reg["prediction_reg"]
#distribution des erreurs(limite les erreurs extremes entre min -1000 et max 1000)
#axes_g12[0]: graphe à gauche et axes_g12[1]: graphe à droite
axes_g12[0].hist(residus.clip(-1000, 1000), bins=50, color=PALETTE["bleu"], alpha=0.7, edgecolor="white")
#ligne verticale erreur=0 sur premier axe
axes_g12[0].axvline(0, color=PALETTE["rouge"], ls="--", lw=2)
axes_g12[0].set_title("Distribution des résidus (régression)", fontsize=11, fontweight="bold")
axes_g12[0].set_xlabel("Erreur (réel − prédit)"); axes_g12[0].grid(True, alpha=0.3)
#clip(0,2000): limite les val très grandes
axes_g12[1].scatter(test_reg["prediction_reg"].clip(0, 2000),
                   test_reg[CIBLE_REG].clip(0, 2000),
                   alpha=0.3, s=8, color=PALETTE["bleu"])
#calcul de la lim max: prendre la plus grande val pred et reell du graph pour tracer la diag correc
#compare les 2 val et prend max exp max(900,700) -> lim = 900
lim = max(test_reg["prediction_reg"].max(), test_reg[CIBLE_REG].max())
#un point sur diag signifie reel=predit donc predic parfaite
axes_g12[1].plot([0, lim], [0, lim], color=PALETTE["rouge"], ls="--", lw=2, label="Parfait")
axes_g12[1].set_xlabel("Prédit"); axes_g12[1].set_ylabel("Réel")
axes_g12[1].set_title("Prédit vs Réel (régression)", fontsize=11, fontweight="bold")
axes_g12[1].legend(); axes_g12[1].grid(True, alpha=0.3)
plt.tight_layout()
ajouter_graphique("Analyse des résidus — Régression", fig_g12,
    "Distribution des erreurs et nuage prédit vs réel")

print(f"  ✅ {len(GRAPHIQUES_HTML)} graphiques générés")

# ════════════════════════════════════════════════════════════
#  GÉNÉRATION DU RAPPORT HTML UNIFIÉ
# ════════════════════════════════════════════════════════════

print("\n📄 Génération du rapport HTML unifié...")
#kpi regression
#recu prem ligne du tab des res(meill mdl)
best_row = df_res_reg.iloc[0]
#construction des cartes kpi chaque carte affiche kpi et sa val
html_cards_reg = f"""
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-val">{best_row['MAE']:.1f}</div><div class="kpi-label">MAE — Perte unique</div></div>
  <div class="kpi-card"><div class="kpi-val">{best_row['wMAPE']:.1f}%</div><div class="kpi-label">wMAPE — Métrique</div></div>
  <div class="kpi-card good"><div class="kpi-val">+{gain:.1f}%</div><div class="kpi-label">Gain vs Baseline</div></div>
  <div class="kpi-card"><div class="kpi-val">{t20:.1f}%</div><div class="kpi-label">Erreur ≤ 20%</div></div>
  <div class="kpi-card"><div class="kpi-val">{mape_norm:.1f}%</div><div class="kpi-label">wMAPE Normales</div></div>
  <div class="kpi-card"><div class="kpi-val">{mape_pic:.1f}%</div><div class="kpi-label">wMAPE Pics</div></div>
</div>"""

html_cards_clf = f"""
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-val">{metriques_clf['Perte (Logloss)']:.4f}</div><div class="kpi-label">Logloss — Perte unique</div></div>
  <div class="kpi-card good"><div class="kpi-val">{acc:.1f}%</div><div class="kpi-label">Accuracy</div></div>
  <div class="kpi-card good"><div class="kpi-val">{f1:.3f}</div><div class="kpi-label">F1-Score</div></div>
  <div class="kpi-card good"><div class="kpi-val">{rec:.1%}</div><div class="kpi-label">Rappel (ruptures)</div></div>
  <div class="kpi-card good"><div class="kpi-val">{auc:.3f}</div><div class="kpi-label">AUC-ROC</div></div>
  <div class="kpi-card"><div class="kpi-val">{meilleur_seuil:.2f}</div><div class="kpi-label">Seuil optimal</div></div>
</div>"""

html_cards_ano = f"""
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-val">{metriques_ano['Perte (Logloss)']:.4f}</div><div class="kpi-label">Logloss — Perte unique</div></div>
  <div class="kpi-card good"><div class="kpi-val">{acc_ano:.1f}%</div><div class="kpi-label">Accuracy</div></div>
  <div class="kpi-card good"><div class="kpi-val">{f1_ano:.3f}</div><div class="kpi-label">F1-Score</div></div>
  <div class="kpi-card good"><div class="kpi-val">{rec_ano:.1%}</div><div class="kpi-label">Rappel (anomalies)</div></div>
  <div class="kpi-card good"><div class="kpi-val">{auc_ano:.3f}</div><div class="kpi-label">AUC-ROC</div></div>
  <div class="kpi-card"><div class="kpi-val">{meilleur_seuil_ano:.2f}</div><div class="kpi-label">Seuil optimal</div></div>
</div>"""

# Table comparaison modèles
#chaine vide servira à stocker tt les log html du tab
table_rows = ""
#boucle sur les mdl(parcourt chaque ligne du DF) iterrows:permet de lire ligne par ligne
for _, row in df_res_reg.iterrows():
    #verifier si ce mdl est le meill
    #si le mdl est lightgbm(yaany le meill)donc best_cls = ' class="best-row"' sinn best_cls = ""
    best_cls = ' class="best-row"' if row["Modèle"] == meilleur_reg else ""
    #Construction de la ligne html
    table_rows += f"<tr{best_cls}><td>{row['Modèle']}</td><td>{row['MAE']:.2f}</td><td>{row['wMAPE']:.1f}%</td></tr>"

# Table perf par article

art_rows = ""
#parcourir chaque ligne du df perf_art(art:identif de larticle, row: les val de la ligne)
for art, row in perf_art.iterrows():
    #attribue une class css selon erreur mae(si mae<100: bonne preditcion, 
    #snn mae<300: erreur moyenne ,snn mauvaise prediction
    cls = "good" if row["mae_article"] < 100 else ("warn" if row["mae_article"] < 300 else "bad")
    #ajout dune lig html au tabl(:.0faffiche float sans déci)
    art_rows += f"<tr><td>{art}</td><td>{row['reel_moyen']:.0f}</td><td>{row['pred_moyen']:.0f}</td><td class='{cls}'>{row['mae_article']:.1f}</td><td>{row['wmape_%']:.1f}%</td></tr>"

# Blocs graphiques
graphiques_html_str = ""#var vide qui va contenir tous les blocs graphi html
#boucle sur les graph: chaque elem de graph_html contient titre, image convertie en base64 et desc
for i, (titre, b64, desc) in enumerate(GRAPHIQUES_HTML):
    #ajoute nv bloc graphique 
    graphiques_html_str += f"""
    <div class="chart-block">
      <h3 class="chart-title">{titre}</h3>
      <!-- #si desc existe elle est affichée snn rien nest ajouté-->
      {"<p class='chart-desc'>" + desc + "</p>" if desc else ""} 
      <img src="data:image/png;base64,{b64}" alt="{titre}" class="chart-img"/>
    </div>"""
#construction du doc html complet
html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Rapport ML — Prévision demande & Rupture stock</title>
<style>
<!-- partie css(couleurs, design, styles)-->
  :root {{
    --bleu:   #2563eb;
    --vert:   #16a34a;
    --rouge:  #dc2626;
    --orange: #d97706;
    --fond:   #f1f5f9;
    --blanc:  #ffffff;
    --texte:  #1e293b;
    --gris:   #64748b;
    --border: #e2e8f0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--fond); color: var(--texte); }}
<!-- header rapp: titre prj, architec ml,date génération-->
  /* HEADER */
  .header {{
    background: linear-gradient(135deg, #1e40af 0%, #0f766e 100%);
    color: white; padding: 40px 48px; position: relative; overflow: hidden;
  }}
  .header::before {{
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 250px; height: 250px; border-radius: 50%;
    background: rgba(255,255,255,0.06);
  }}
  .header h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }}
  .header p  {{ margin-top: 8px; opacity: .8; font-size: 1rem; }}
  .header .badge {{ display:inline-block; background:rgba(255,255,255,.18);
    border-radius: 20px; padding: 4px 14px; font-size:.82rem; margin-top:12px; }}

  /* NAV */
  .nav {{ background: var(--blanc); border-bottom: 1px solid var(--border);
    padding: 0 48px; display: flex; gap: 0; position: sticky; top: 0; z-index: 100;
    box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
  .nav a {{ padding: 16px 20px; text-decoration: none; color: var(--gris);
    font-size: .9rem; font-weight: 500; border-bottom: 3px solid transparent;
    transition: all .2s; }}
  .nav a:hover {{ color: var(--bleu); border-bottom-color: var(--bleu); }}

  /* SECTIONS */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}
  .section {{ background: var(--blanc); border-radius: 16px; padding: 32px;
    margin-bottom: 32px; box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
  .section-title {{ font-size: 1.35rem; font-weight: 700; margin-bottom: 8px;
    display: flex; align-items: center; gap: 10px; }}
  .section-title .dot {{ width:12px; height:12px; border-radius:50%; background: var(--bleu); }}
  .section-subtitle {{ color: var(--gris); font-size: .9rem; margin-bottom: 24px; }}

  /* KPI CARDS */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px; margin-bottom: 28px; }}
  .kpi-card {{ background: var(--fond); border-radius: 12px; padding: 18px 20px;
    border-left: 4px solid var(--bleu); }}
  .kpi-card.good {{ border-left-color: var(--vert); }}
  .kpi-card.warn {{ border-left-color: var(--orange); }}
  .kpi-val {{ font-size: 1.7rem; font-weight: 800; color: var(--texte); }}
  .kpi-label {{ font-size: .78rem; color: var(--gris); margin-top: 4px; }}

  /* TABLE */
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th {{ background: #f8fafc; padding: 10px 14px; text-align: left;
    font-weight: 600; border-bottom: 2px solid var(--border); color: var(--gris); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #f8fafc; }}
  tr.best-row td {{ background: #f0fdf4; font-weight: 700; color: var(--vert); }}
  td.good {{ color: var(--vert); font-weight: 600; }}
  td.warn {{ color: var(--orange); font-weight: 600; }}
  td.bad  {{ color: var(--rouge); font-weight: 600; }}

  /* CHARTS */
  .charts-grid {{ display: grid; grid-template-columns: 1fr; gap: 28px; }}
  .chart-block {{ background: var(--fond); border-radius: 12px; padding: 20px;
    border: 1px solid var(--border); }}
  .chart-title {{ font-size: 1rem; font-weight: 700; margin-bottom: 6px; color: var(--texte); }}
  .chart-desc  {{ font-size: .83rem; color: var(--gris); margin-bottom: 12px; }}
  .chart-img   {{ width: 100%; border-radius: 8px; display: block; }}

  /* INFO BOXES */
  .info-box {{ background: #eff6ff; border-left: 4px solid var(--bleu);
    border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; font-size: .88rem; }}
  .info-box strong {{ color: var(--bleu); }}

  /* FOOTER */
  .footer {{ background: #1e293b; color: #94a3b8; text-align: center;
    padding: 28px; font-size: .84rem; margin-top: 40px; }}
</style>
</head>
<body>

<div class="header">
  <h1> Rapport ML — Prévision demande & Rupture stock</h1>
  <p>Modélisation LightGBM · Architecture 2 modèles (Normal + Pic) · Rupture + Anomalie stock</p>
  <!-- Génère automatiquement la date actuelle-->
  <span class="badge">Généré automatiquement · {pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")}</span>
</div>

<nav class="nav">
  <a href="#resume">Résumé</a>
  <a href="#regression">Régression</a>
  <a href="#classification">Rupture stock</a>
  <a href="#anomalie">Anomalie stock</a>
  <a href="#graphiques">Graphiques</a>
</nav>

<div class="container">

  <!-- RÉSUMÉ -->
  <div class="section" id="resume">
    <div class="section-title"><span class="dot"></span>Résumé exécutif</div>
    <div class="section-subtitle">Vue d'ensemble des performances des deux tâches ML</div>
    <div class="info-box">
      <strong>Architecture :</strong> 2 modèles LightGBM séparés (Normal &lt; {SEUIL_PIC} unités / Pic ≥ {SEUIL_PIC} unités)
      · <strong>Perte régression :</strong> MAE (regression_l1)
      · <strong>Perte classification :</strong> Binary Logloss
    </div>
    <div class="kpi-grid">
      <div class="kpi-card good"><div class="kpi-val">MAE</div><div class="kpi-label">Perte unique — Régression</div></div>
      <div class="kpi-card good"><div class="kpi-val">Logloss</div><div class="kpi-label">Perte unique — Classification</div></div>
      <div class="kpi-card"><div class="kpi-val">{len(FEATURES_ALL)}</div><div class="kpi-label">Features totales</div></div>
      <div class="kpi-card"><div class="kpi-val">{len(train)}</div><div class="kpi-label">Lignes train</div></div>
      <div class="kpi-card"><div class="kpi-val">{len(test)}</div><div class="kpi-label">Lignes test</div></div>
      <div class="kpi-card warn"><div class="kpi-val">3</div><div class="kpi-label">Modèles entraînés</div></div>
    </div>
  </div>

  <!-- RÉGRESSION -->
  <div class="section" id="regression">
    <div class="section-title"><span class="dot" style="background:var(--bleu)"></span>Partie 1 — Régression quantité</div>
    <div class="section-subtitle">Perte unique : MAE (regression_l1) · Métriques rapportées : MAE + wMAPE</div>
    {html_cards_reg}
    <h4 style="margin-bottom:12px;color:var(--gris)">Comparaison des modèles</h4>
    <table>
      <thead><tr><th>Modèle</th><th>MAE (perte)</th><th>wMAPE (%)</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    <br/>
    <h4 style="margin-bottom:12px;color:var(--gris)">Performances par article</h4>
    <table>
      <thead><tr><th>Article</th><th>Réel moyen</th><th>Prédit moyen</th><th>MAE</th><th>wMAPE %</th></tr></thead>
      <tbody>{art_rows}</tbody>
    </table>
  </div>

  <!-- CLASSIFICATION RUPTURE -->
  <div class="section" id="classification">
    <div class="section-title"><span class="dot" style="background:var(--rouge)"></span>Partie 2 — Classification rupture de stock</div>
    <div class="section-subtitle">Perte unique : Binary Logloss · Métriques : Accuracy + F1-Score · Seuil optimal (rappel ≥ 90%)</div>
    {html_cards_clf}
    <div class="info-box">
      <strong>Seuil optimal :</strong> {meilleur_seuil:.2f} — sélectionné pour maximiser le F1-Score sous contrainte rappel ≥ 90%
      · <strong>scale_pos_weight :</strong> {ratio:.1f} (rééquilibrage classes)
    </div>
  </div>

  <!-- ANOMALIE STOCK -->
  <div class="section" id="anomalie">
    <div class="section-title"><span class="dot" style="background:#7c3aed"></span>Partie 3 — Anomalie stock</div>
    <div class="section-subtitle">
      Surstock (stock disponible &gt; 8 semaines de demande) · Mauvaise anticipation (commande fournisseur &gt; 2× demande réelle)
      · Perte unique : Binary Logloss · Seuil optimal (rappel ≥ 85%)
    </div>
    {html_cards_ano}
    <div class="info-box">
      <strong>Surstock :</strong> stock_disponible / demande_moy &gt; 8 semaines de couverture
      · <strong>Mauvaise anticipation :</strong> commande_fournisseur &gt; 2 × demande_réelle cette semaine (ou commande sans demande)
      · <strong>Seuil optimal :</strong> {meilleur_seuil_ano:.2f} (rappel ≥ 85%)
      · <strong>scale_pos_weight :</strong> {ratio_ano:.1f}
    </div>
  </div>

  <!-- GRAPHIQUES -->
  <div class="section" id="graphiques">
    <div class="section-title"><span class="dot" style="background:var(--vert)"></span>Visualisations complètes</div>
    <div class="section-subtitle">{len(GRAPHIQUES_HTML)} graphiques — tous les résultats en un seul rapport</div>
    <div class="charts-grid">
      {graphiques_html_str}
    </div>
  </div>

</div>

<div class="footer">
  Rapport généré automatiquement · Projet Prévision demande & Optimisation stocks · Eya Ben Khlifa
</div>
</body>
</html>"""
#enregistrer le tab html dans un fichier
with open("rapport_ml.html", "w", encoding="utf-8") as f:
    f.write(html)
print("  ✅ rapport_ml.html — rapport unifié généré")

# ════════════════════════════════════════════════════════════
#  EXPORT
# ════════════════════════════════════════════════════════════

print("\n\n💾 Export des résultats...")
#exporter les resultats de reg dans un fich excel(avant exp supp col score)
df_res_reg.drop(columns=["Score"], errors="ignore").to_excel(
    "resultats_regression.xlsx", index=False, engine="openpyxl")

#creat du df selectionnant certaines col du df test_reg
df_pred_reg = test_reg[["identifiant_article", "date_debut_semaine",
                          CIBLE_REG, "prediction_reg", "erreur_abs", "est_pic"]].rename(
    columns={CIBLE_REG: "reel", "prediction_reg": "prediction"})#renommer les col pour les rendre lisibles
#export excel des predic
df_pred_reg.to_excel("predictions_demande.xlsx", index=False, engine="openpyxl")
#export des perfor par article
perf_art.to_excel("performances_par_article.xlsx", engine="openpyxl")

# création df rupture(selec des col utiles du dataset test)
df_pred_clf = test_clf_df[["identifiant_article", "date_debut_semaine",
                             CIBLE_CLF, "proba_rupture", "pred_rupture"]].rename(
    columns={CIBLE_CLF: "rupture_reelle"})#renommage cible réelle
#arrondi proba à 3 decim
df_pred_clf["proba_rupture"] = df_pred_clf["proba_rupture"].round(3)
#export
df_pred_clf.to_excel("predictions_rupture.xlsx", index=False, engine="openpyxl")
perf_clf_art.to_excel("performances_rupture_par_article.xlsx", engine="openpyxl")

# Export anomalie stock (surstock + mauvaise anticipation) — NOUVEAU v3
df_pred_ano = test_ano_df[["identifiant_article", "date_debut_semaine",
                             CIBLE_ANO, "proba_anomalie", "pred_anomalie"]].rename(
    columns={CIBLE_ANO: "anomalie_reelle"})
# Ajouter le détail surstock / mauvaise anticipation si disponible
#si la col existe donc on fait copie
for col, new_col in [("cible_surstock", "surstock_reel"),
                     ("cible_mauvaise_anticipation", "mauvaise_anticipation_reel")]:
    if col in test.columns:#evite les err si la col nexiste pas
        df_pred_ano[new_col] = test[col].values
df_pred_ano["proba_anomalie"] = df_pred_ano["proba_anomalie"].round(3)
df_pred_ano.to_excel("predictions_anomalie_stock.xlsx", index=False, engine="openpyxl")
perf_ano_art.to_excel("performances_anomalie_par_article.xlsx", engine="openpyxl")

# Sauvegarde modèles — pickle: biblio py qui sauvg des objets et les recharger plus tard
import pickle
with open("modeles_ml.pkl", "wb") as f:#ouverture du fich en ecri binaire
    #enregistrer un dic complet dans le fichier
    pickle.dump({
        # Régression — LightGBM par article (2 modèles Normal + Pic)
        "modele_normal"      : modele_normal,
        "modele_pic"         : modele_pic,
        "modele_global_norm" : modele_global_norm,
        "modele_global_pic"  : modele_global_pic,
        "modele_lgb_global"  : modele_lgb,
        # Classification rupture
        "modele_clf"         : modele_clf,
        # Classification anomalie stock
        "modele_ano"         : modele_ano,
        # Encodeurs et config
        "label_encoder"      : le,
        "features"           : FEATURES_ALL,
        "seuil_pic"          : SEUIL_PIC,
        "seuil_clf"          : meilleur_seuil,
        "seuil_ano"          : meilleur_seuil_ano,
        "best_lgb"           : best_lgb,
        "best_pic"           : best_pic,
        "best_ano"           : best_ano,
    }, f)

print("  ✅ Tous les fichiers exportés")
print("  ✅ modeles_ml.pkl — 3 modèles sauvegardés pour l'agent")

print(f"\n\n{'█'*55}")
print(f"  🎉 RÉSUMÉ FINAL")
print(f"{'█'*55}")
print(f"\n  RÉGRESSION — Perte unique : MAE (regression_l1)")
print(f"  {'─'*50}")
print(f"  Architecture    : 2 modèles (Normal<{SEUIL_PIC} + Pic≥{SEUIL_PIC})")
print(f"  Meilleur modèle : {meilleur_reg}")
print(f"    MAE  : {best_row['MAE']:.2f} unités")
print(f"    wMAPE : {best_row['wMAPE']:.1f}%")
print(f"    Gain vs Baseline  : +{gain:.1f}%")
print(f"    wMAPE normales (<{SEUIL_PIC}) : {mape_norm:.1f}%")
print(f"    wMAPE pics     (≥{SEUIL_PIC}) : {mape_pic:.1f}%")
print(f"    Taux erreur ≤ 20%     : {t20:.1f}%")
print(f"\n  CLASSIFICATION RUPTURE — Perte unique : Binary Logloss")
print(f"  {'─'*50}")
print(f"    Logloss   : {metriques_clf['Perte (Logloss)']:.4f}")
print(f"    Accuracy  : {acc:.1f}%")
print(f"    F1-Score  : {f1:.3f}")
print(f"    Rappel    : {rec:.3f}  ({rec*100:.1f}% ruptures détectées)")
print(f"    AUC-ROC   : {auc:.3f}")
print(f"    Seuil     : {meilleur_seuil:.2f}")
print(f"\n  CLASSIFICATION ANOMALIE STOCK — Perte unique : Binary Logloss")
print(f"  {'─'*50}")
print(f"    Définition : surstock (couv.>8sem.) OU commande fourn.>2×demande")
print(f"    Logloss   : {metriques_ano['Perte (Logloss)']:.4f}")
print(f"    Accuracy  : {acc_ano:.1f}%")
print(f"    F1-Score  : {f1_ano:.3f}")
print(f"    Rappel    : {rec_ano:.3f}  ({rec_ano*100:.1f}% anomalies détectées)")
print(f"    AUC-ROC   : {auc_ano:.3f}")
print(f"    Seuil     : {meilleur_seuil_ano:.2f}")
print(f"\n  📄 Rapport HTML : rapport_ml.html")
print(f"  📊 Exports      : predictions_demande.xlsx | predictions_rupture.xlsx | predictions_anomalie_stock.xlsx")



"""
=======================================================================
3. Rapport HTML & Exports
12 graphiques générés en mémoire (base64 PNG) et intégrés dans un rapport HTML autonome rapport_ml.html :

Prévision vs réel — top 3 articles
MAPE Normal vs Pic par architecture
MAE par article (code couleur vert/orange/rouge)
Comparaison MAE des 3 modèles
Top 15 features par importance (gain LightGBM)
Courbes ROC + optimisation seuil (rupture & anomalie)
Répartition anomalies par article (surstock vs mauvaise anticipation)
Distribution des résidus + nuage prédit/réel
========================================================================
"""