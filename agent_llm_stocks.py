"""
=============================================================
 AGENT LLM v3 — Optimisation des stocks & Interprétation KPI
Script autonome qui charge les sorties du pipeline ML et les soumet 
à un LLM local (Ollama llama3.2:3b) pour produire trois analyses 
textuelles exportées en .txt.
=============================================================
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime

OLLAMA_URL   = "http://localhost:11434/api/generate" #mdl local
OLLAMA_MODEL = "llama3.2:3b"#mdl ollama utilisé


# ════════════════════════════════════════════════════════════
#  1. CHARGEMENT
# ════════════════════════════════════════════════════════════

def charger_donnees():
    print("Chargement des fichiers du projet v3...")

    pred_dem     = pd.read_excel("predictions_demande.xlsx")
    pred_rup     = pd.read_excel("predictions_rupture.xlsx")
    pred_ano     = pd.read_excel("predictions_anomalie_stock.xlsx")   
    perf_art     = pd.read_excel("performances_par_article.xlsx")
    perf_rup_art = pd.read_excel("performances_rupture_par_article.xlsx")
    perf_ano_art = pd.read_excel("performances_anomalie_par_article.xlsx")  
    res_reg      = pd.read_excel("resultats_regression.xlsx")
    base         = pd.read_excel("base_ml_finale.xlsx")
    #parcouri 4 df et convertir la col date en vrai format date pandas
    #sans ça la col peut etre string(avant: une fois convertir on peut faire des filtres temp)
    for df in [pred_dem, pred_rup, pred_ano, base]:
        df["date_debut_semaine"] = pd.to_datetime(df["date_debut_semaine"])
    #affichage du nbr dart unique dans pred_dem
    print(f"  {pred_dem['identifiant_article'].nunique()} articles modélisés")
    #affichage periode test
    print(f"  Période test ML : {pred_dem['date_debut_semaine'].min().date()} "
          f"-> {pred_dem['date_debut_semaine'].max().date()}")

    return pred_dem, pred_rup, pred_ano, perf_art, perf_rup_art, perf_ano_art, res_reg, base


# ════════════════════════════════════════════════════════════
#  2. TABLEAU ARTICLES 
# ════════════════════════════════════════════════════════════

#fonct qui construit une tab enrichie par art combinant les predic ml et les stat
def construire_tableau_articles(pred_dem, pred_rup, pred_ano, perf_art,
                                 perf_rup_art, perf_ano_art, base):
    #recup la liste des artc présents dans les predic
    articles_ml = pred_dem["identifiant_article"].unique()
    #selec des col utiles
    cols = ["identifiant_article","stock_disponible","prix_vente","cout_achat","lead_time_hist_moy"]
    #filtrage de base: garder les art, les col selec
    statiques = (base[base["identifiant_article"].isin(articles_ml)][cols]
                 .groupby("identifiant_article").last().reset_index())
#grouper les donn article par article pour calc des indicateurs
    stats_dem = pred_dem.groupby("identifiant_article").agg(
        demande_moy=("reel","mean"),
        demande_std=("reel","std"),
        prediction_s1=("prediction","last"),#dernière predict
        derniere_semaine=("date_debut_semaine","max"),
    ).reset_index()
    #trie des donn par date(plus anc au plus réc):respc lordre temp des sem
    proba_recente = (pred_rup.sort_values("date_debut_semaine")
                    .groupby("identifiant_article")
                    #calcul sur les 4dern sem(pour chaque art on garde les 4 der sem)
                    #on prend la proba max de rupture sur ces  4 sem
                    .apply(lambda x: x.tail(4)["proba_rupture"].max())
                    .reset_index())
    proba_recente.columns = ["identifiant_article","proba_rupture_max"]

    # proba anomalie max sur 4 dernières semaines
    proba_ano = (pred_ano.sort_values("date_debut_semaine")
                 .groupby("identifiant_article")
                 .apply(lambda x: x.tail(4)["proba_anomalie"].max())
                 .reset_index())
    proba_ano.columns = ["identifiant_article","proba_anomalie_max"]

    #verifie si col mdl existe dans perf_art
    #si col existe on lutilise snn on utilise identif artic
    col_mod = "Modèle" if "Modèle" in perf_art.columns else "identifiant_article"
    #tab reg selec uniqu les col importantes
    perf_reg = perf_art[["identifiant_article","mae_article","wmape_%"]].copy()
    #tab de class 
    perf_clf = perf_rup_art[["identifiant_article","ruptures_reelles","ruptures_detectees","taux_detection_%"]].copy()

    # tableau des perfo du mdl danomalie
    #création du df perf_anor_merge contenant 3 col
    #il gère 2cas si identif est existe il selec les 3 col 
    #snn si identif est dans lindex reset.index() transforme lind en col et recup les 3 mm col
    perf_ano_merge = perf_ano_art[["identifiant_article","anomalies_reelles","anomalies_detectees"]].copy() \
        if "identifiant_article" in perf_ano_art.columns \
        else perf_ano_art.reset_index()[["identifiant_article","anomalies_reelles","anomalies_detectees"]].copy()
#on part du df principal après on ajoute dautre infos
#fusionner les df avec jointure gauche(càd garder ts les art presents dans stat mm sil 
#nexistent pas dans les autres tables)
    df = (statiques
          .merge(stats_dem,     on="identifiant_article", how="left")
          .merge(proba_recente, on="identifiant_article", how="left")
          .merge(proba_ano,     on="identifiant_article", how="left")
          .merge(perf_reg,      on="identifiant_article", how="left")
          .merge(perf_clf,      on="identifiant_article", how="left")
          .merge(perf_ano_merge, on="identifiant_article", how="left"))

    df["couverture_sem"] = (df["stock_disponible"] / df["demande_moy"].replace(0, np.nan)).round(1)
    df["valeur_stock"]   = df["stock_disponible"] * df["prix_vente"]

    df["statut"] = df["proba_rupture_max"].apply(
        lambda x: "CRITIQUE" if x >= 0.90 else ("ATTENTION" if x >= 0.55 else "OK"))
    df["statut_anomalie"] = df["proba_anomalie_max"].apply(
        lambda x: "SURSTOCK/ANTICIPATION" if x >= 0.85 else ("A SURVEILLER" if x >= 0.55 else "NORMAL"))
    #on fait lordre comme ca psq pandas fait tri selon les chaines par ordre alpha
    ordre = {"CRITIQUE":0,"ATTENTION":1,"OK":2}
    #transformation statut en val num(exp: avant ok après=2)
    df["_tri"] = df["statut"].map(ordre)
    #tri sur 2 col(critère1: ordre croissant sur statut)
    #critère2: ordre decr proba_rupture
    df = df.sort_values(["_tri","proba_rupture_max"], ascending=[True,False]).drop(columns=["_tri"])
    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
#  3. APPEL OLLAMA
# ════════════════════════════════════════════════════════════
#fonction qui envoie un prompt à un llm via olama et recup la reponse generee
def appeler_llm(prompt, max_tokens=1500):#prompt(le txt envoyé au mdl)
    #end: permet de rester sur mm lig, flush:force laffichage dans le terminal
    print("  LLM en cours de réflexion", end="", flush=True)
    #on essaie dappeler serv ollama
    #si err arrive: serv eteint, timeout, json invalide le pg ne crash pas
    try:
        #envoie de la requete http(post)
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,#stream: attend tte la rep avant de la retourner
                  "options": {"temperature": 0.3, "num_predict": max_tokens}},
                  #temp:niveau daleatoire(plus il est bas plus mdl choisit des rep plus probables)
            timeout=300#pg attend maxi 5min, prompt(texte envoyé)
        )
        print(" OK")
        #recup reponse reçue du serveur ollam
        return response.json()["response"]
    #capturer les erreurs de connexion(ollama nest pas lancé,mauvais prot, serveur inacc)
    except requests.exceptions.ConnectionError:
        print("\nERREUR : Ollama n'est pas démarré !")
        return None#comme ya une erreur , fonct renvoie none: cela evite crash pg
    except Exception as e:#capture ttes les erreurs possibles(e: contient msg derreur)
        print(f"\nERREUR : {e}")
        return None


# ════════════════════════════════════════════════════════════
#  4. PROMPT PHASE 6 — OPTIMISATION STOCKS
# ════════════════════════════════════════════════════════════

def prompt_phase6(df, res_reg):
    #verifier col mdl existe(si oui col_mod=modèle snn col_mod=modele)
    col_mod = "Modèle" if "Modèle" in res_reg.columns else "Modele"
    mae_best  = res_reg.iloc[0]["MAE"]#recup valeur mae du meilleur mdl
    mape_best = res_reg.iloc[0]["wMAPE"]#de mm pour wmape
#transformer un sous_df en texte
    def formater(sous_df):
        if len(sous_df) == 0:#si df vide fonct retourne aucun article
            return "  Aucun article."
        lignes = []#liste qui va stocker les lignes de texte
        for _, r in sous_df.iterrows():#boucle sur chaque article(_:index non utilisé, r: ligne act)
            #ajoute ligne texte dans la liste
            lignes.append(
                f"  - {r['identifiant_article']}"#identif darticle
                f" | stock={int(r['stock_disponible'])} unites"#stock disponible: int(enlève les décim)
                f" | couverture={r['couverture_sem']} semaines"#nbr de semaines couvertes par le stock
                #.0f arrondi sans décim
                f" | demande_moy={r['demande_moy']:.0f} u/sem"
                f" | prevision_S+1={int(r['prediction_s1'])} unites"
                #.0% transforme en %
                f" | proba_rupture={r['proba_rupture_max']:.0%}"
                f" | proba_anomalie={r['proba_anomalie_max']:.0%}"
                f" | lead_time={r['lead_time_hist_moy']:.1f} sem"
                f" | prix={r['prix_vente']:.2f} EUR"
                f" | MAE_modele={r['mae_article']:.0f} unites"
            )
        #fusionne ttes les lignes avec des retours à la ligne
        return "\n".join(lignes)
#exp: - ART_001 | stock=120 unites | couverture=3 semaines | demande_moy=35 u/sem | prevision_S+1=40 unites | proba_rupture=87% | proba_anomalie=20% | lead_time=2.4 sem | prix=15.90 EUR | MAE_modele=5 unites

    critiques = df[df["statut"]=="CRITIQUE"]
    attention = df[df["statut"]=="ATTENTION"]
    ok        = df[df["statut"]=="OK"]

    return f"""Tu es un expert en gestion de stocks. Reponds en francais.
Analyse ces donnees ML (3 modeles : regression demande, detection rupture, detection anomalie stock).

MODELE ML : MAE={mae_best:.0f} unites, wMAPE={mape_best:.1f}%
DATE : {datetime.now().strftime('%d/%m/%Y')}

ARTICLES CRITIQUES (rupture imminente) :
{formater(critiques)}

ARTICLES ATTENTION :
{formater(attention)}

ARTICLES OK :
{formater(ok)}

TOTAUX : {len(critiques)} critiques | {len(attention)} attention | {len(ok)} OK

Pour chaque article CRITIQUE et ATTENTION, dis :
1. Quantite a commander (calcule : demande_moy x lead_time - stock + marge securite)
2. Urgence (combien de jours avant rupture)
3. Si proba_anomalie > 0.55 : signaler aussi le risque de surstock/mauvaise anticipation
4. Risque financier si on ne commande pas (quantite x prix)

Termine par un RESUME de 3 lignes pour le responsable.
"""


# ════════════════════════════════════════════════════════════
#  5. PROMPT PHASE 7 — INTERPRETATION KPI
# ════════════════════════════════════════════════════════════
#fonct prend 4 jeux de données
def prompt_phase7(df, perf_rup_art, perf_ano_art, res_reg):
    col_mod   = "Modèle" if "Modèle" in res_reg.columns else "Modele"
    mae_best  = res_reg.iloc[0]["MAE"]
    mape_best = res_reg.iloc[0]["wMAPE"]
    #recup mae du mdl baseline
    mae_base  = res_reg[res_reg[col_mod].str.contains("Baseline")]["MAE"].values[0]
    gain      = (mae_base - mae_best) / mae_base * 100

    total_rup  = perf_rup_art["ruptures_reelles"].sum()
    total_det  = perf_rup_art["ruptures_detectees"].sum()
    total_ano  = perf_ano_art["anomalies_reelles"].sum()
    det_ano    = perf_ano_art["anomalies_detectees"].sum()
    valeur_stk = (df["stock_disponible"] * df["prix_vente"]).sum()

    return f"""Tu es un expert en data science. Reponds en francais, sans jargon technique.
Interprete ces resultats ML (3 modeles) pour un directeur des achats.

REGRESSION (prevision demande) :
  Meilleur modele : {res_reg.iloc[0][col_mod]}
  MAE = {mae_best:.0f} unites d'erreur moyenne
  wMAPE = {mape_best:.1f}%
  Gain vs methode classique = +{gain:.1f}%

CLASSIFICATION RUPTURE :
  Ruptures reelles    = {total_rup}
  Ruptures detectees  = {total_det}
  Taux detection      = {total_det/max(total_rup,1)*100:.1f}%

CLASSIFICATION ANOMALIE STOCK (NOUVEAU v3) :
  Definition : surstock (couverture > 8 sem.) OU commande fourn. > 2x demande reelle
  Anomalies reelles    = {total_ano}
  Anomalies detectees  = {det_ano}
  Taux detection       = {det_ano/max(total_ano,1)*100:.1f}%

ETAT STOCK :
  Valeur totale stock = {valeur_stk:,.0f} EUR
  Articles CRITIQUES  = {len(df[df['statut']=='CRITIQUE'])} / {len(df)}
  Articles anomalie   = {len(df[df['statut_anomalie']!='NORMAL'])} / {len(df)}

Reponds a ces 5 questions :
1. Le modele de prevision est-il fiable ? (utilise gain +{gain:.1f}%)
2. Que signifie detecter {total_det}/{total_rup} ruptures en pratique ?
3. Quel est l'interet du modele anomalie stock (surstock + mauvaise anticipation) ?
4. Quels articles ont des previsions peu fiables et pourquoi ?
5. 3 recommandations concretes pour ameliorer le systeme.
"""


# ════════════════════════════════════════════════════════════
#  6. PROMPT PHASE 3 — ANOMALIE STOCK 
# ════════════════════════════════════════════════════════════

def prompt_phase3_anomalie(df, pred_ano, perf_ano_art):
    """
    Analyse dédiée aux anomalies de stock :
    - Surstock (couverture > 8 semaines)
    - Mauvaise anticipation (commande fourn. > 2x demande réelle)
    """
    anomalies = df[df["statut_anomalie"]!="NORMAL"][[
        "identifiant_article","statut_anomalie","proba_anomalie_max",
        "couverture_sem","stock_disponible","valeur_stock","demande_moy"
    ]].copy()

    n_sur = pred_ano["surstock_reel"].sum() if "surstock_reel" in pred_ano.columns else "N/A"
    n_ant = pred_ano["mauvaise_anticipation_reel"].sum() if "mauvaise_anticipation_reel" in pred_ano.columns else "N/A"
    capital_immobilise = anomalies["valeur_stock"].sum()

    return f"""Tu es un expert en logistique et gestion de stocks. Reponds en francais, de facon precise.

ANALYSE DES ANOMALIES STOCK (modele LightGBM v3) :
  Definition surstock : couverture_semaines > 8 (stock_disponible / demande_moy > 8 sem.)
  Definition mauvaise anticipation : commande_fournisseur > 2 x demande_reelle (ou commande sans demande)

ARTICLES AVEC ANOMALIES DETECTEES :
{anomalies.to_string(index=False)}

STATISTIQUES GLOBALES :
  Semaines en surstock detectees      : {n_sur}
  Semaines mauvaise anticipation      : {n_ant}
  Capital immobilise en anomalies     : {capital_immobilise:,.0f} EUR
  Taux detection global               : {perf_ano_art['anomalies_detectees'].sum()/max(perf_ano_art['anomalies_reelles'].sum(),1)*100:.1f}%

Pour chaque article avec anomalie, dis :
1. Type d'anomalie principale (surstock ou mauvaise anticipation)
2. Impact financier concret (valeur immobilisee ou risque de surcout)
3. Action recommandee (reduire commandes, negocier avec fournisseur, promouvoir le stock, etc.)

Termine par :
- 2 mesures systematiques pour eviter ces anomalies a l'avenir
"""


# ════════════════════════════════════════════════════════════
#  7. CHATBOT INTERACTIF v3
# ════════════════════════════════════════════════════════════

def chatbot_interactif(df, perf_rup_art, perf_ano_art, res_reg):
    col_mod = "Modèle" if "Modèle" in res_reg.columns else "Modele"

    contexte = f"""Tu es un expert en gestion de stocks. Reponds en francais.
Tu connais les resultats de 3 modeles ML LightGBM :
  1. Regression demande : MAE={res_reg.iloc[0]['MAE']:.0f}, wMAPE={res_reg.iloc[0]['wMAPE']:.1f}%
  2. Classification rupture stock
  3. Classification anomalie stock (surstock + mauvaise anticipation) 

ARTICLES ET STOCKS :
{df[['identifiant_article','stock_disponible','couverture_sem',
     'demande_moy','prediction_s1','proba_rupture_max','proba_anomalie_max',
     'statut','statut_anomalie','wmape_%','prix_vente']].to_string(index=False)}

PERFORMANCES DETECTION RUPTURE :
{perf_rup_art.to_string(index=False)}

PERFORMANCES DETECTION ANOMALIE :
{perf_ano_art.to_string(index=False)}

COMPARAISON MODELES :
{res_reg.to_string(index=False)}

Reponds aux questions en utilisant uniquement ces donnees.
"""

    print("\n" + "="*55)
    print("  CHATBOT v3 — Questions libres")
    print("  Exemples :")
    print("    'Quel article est le plus urgent ?'")
    print("    'Quels articles sont en surstock ?'")
    print("    'Quel est le capital immobilise dans les anomalies ?'")
    print("    'Le modele anomalie est-il fiable ?'")
    print("  Tapez 'quitter' pour terminer")
    print("="*55)

    while True:#pg tourne jusqu'à lutilisateur sarrete
        question = input("\n Vous : ").strip()#attend lutili tape le qst et enleve les espaces inutiles(strip)
        #si lutili tape quitter, exit... chatbot sarrete(break)
        if question.lower() in ["quitter", "exit", "quit"]:
            break
        if not question:#si utili ne tape rien , ignorer et recommencer la boucle
            continue
#construction du txt envoyé au mdl ia
        prompt_complet = contexte + f"\n\nQuestion : {question}\nReponse :"
        reponse = appeler_llm(prompt_complet)
        if reponse:
            print(f"\n Agent : {reponse}")


# ════════════════════════════════════════════════════════════
#  8. VERIFICATION OLLAMA: si ollama est bien lancé avant denvoyer des requetes
# ════════════════════════════════════════════════════════════

def verifier_ollama():
    print("Verification Ollama...", end=" ")#end: empeche retour à la lig
    try:
        #essaie de contacter ollama max 5 sec dattente obj:verifier serveur répond
        requests.get("http://localhost:11434", timeout=5)
        print("OK")
        return True
    #si ne fonct pas
    except:
        print("\nOllama n'est pas démarré !")
        print("1. Ouvre le menu Démarrer → Ollama")
        print("2. Attends 10 secondes puis relance ce script")
        return False#indique sys nest pas pret


# ════════════════════════════════════════════════════════════
#  9. PIPELINE PRINCIPAL 
# ════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*55)
    print("  AGENT LLM LOCAL  — Stocks, Ruptures & Anomalies")
    print("  LLM : Ollama llama3.2:3b (gratuit, local)")
    print("  3 modèles ML : Régression + Rupture + Anomalie")
    print("="*55)
#si verif_ol retourne true not true devient false donc le return ne sexecute pas le pg continue
    if not verifier_ollama():
        return

    #fonct charger donnees: centralise tt ce qui est neces pour lanalyse
    pred_dem, pred_rup, pred_ano, perf_art, perf_rup_art, perf_ano_art, res_reg, base = charger_donnees()

    # construction Tableau articles final
    df = construire_tableau_articles(
        pred_dem, pred_rup, pred_ano, perf_art, perf_rup_art, perf_ano_art, base
    )

    print(f"\nArticles : {len(df[df['statut']=='CRITIQUE'])} CRITIQUES | "
          f"{len(df[df['statut']=='ATTENTION'])} ATTENTION | {len(df[df['statut']=='OK'])} OK")
    print(f"Anomalies : {len(df[df['statut_anomalie']!='NORMAL'])} articles en surstock/mauvaise anticipation")

    # ── Phase 6 : Optimisation stocks
    print("\n--- PHASE 6 : Optimisation des stocks ---")
    reponse_p6 = appeler_llm(prompt_phase6(df, res_reg))#2 fonct imbriquées
    if reponse_p6:
        print("\nRECOMMANDATIONS :\n")
        print(reponse_p6)
        #cree un fichier texte utf-8:garantir la lisibilité des données sur nimporte quel logi, sys ou naviga web
        with open("agent_recommandations.txt", "w", encoding="utf-8") as f:
            #recupere date actuelle avec format (exp:25/05/2026 14:30)
            f.write(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{'='*60}\n\n")
            f.write(reponse_p6)#ecriture des recommandations
        print("\n  -> agent_recommandations.txt sauvegardé")

    # ── Phase 7 : KPI
    print("\n--- PHASE 7 : Interprétation des KPI ---")
    reponse_p7 = appeler_llm(prompt_phase7(df, perf_rup_art, perf_ano_art, res_reg))
    if reponse_p7:
        print("\nINTERPRÉTATION KPI :\n")
        print(reponse_p7)
        with open("agent_kpi.txt", "w", encoding="utf-8") as f:
            f.write(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{'='*60}\n\n")
            f.write(reponse_p7)
        print("\n  -> agent_kpi.txt sauvegardé")

    # ── Phase 3 NOUVELLE : Anomalie stock
    print("\n--- PHASE ANOMALIE : Analyse surstock & mauvaise anticipation ---")
    reponse_ano = appeler_llm(prompt_phase3_anomalie(df, pred_ano, perf_ano_art))
    if reponse_ano:
        print("\nANALYSE ANOMALIE STOCK :\n")
        print(reponse_ano)
        with open("agent_anomalie.txt", "w", encoding="utf-8") as f:
            f.write(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{'='*60}\n\n")
            f.write(reponse_ano)
        print("\n  -> agent_anomalie.txt sauvegardé")

    # ── Phase 8 : Chatbot
    print("\n--- PHASE 8 : Chatbot interactif ---")
    choix = input("\n  Activer le chatbot ? (o/n) : ").strip().lower()
    if choix == "o":
        #lancement du chatbot:envoie plusieurs infos au chatbot
        chatbot_interactif(df, perf_rup_art, perf_ano_art, res_reg)

    print("\n" + "="*55)
    print("  TERMINÉ ")
    print("  Fichiers : agent_recommandations.txt | agent_kpi.txt | agent_anomalie.txt")
    print("="*55)

#lancer la fonct main si fichier est exécuté
if __name__ == "__main__":
    main()