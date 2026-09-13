import requests #permet denvoyer des requettes vers api ou sites web
import pandas as pd
import urllib3#gere les cnx https(ici utiliser pour désactiver les avertiss liés aux 
#certifi SSL lorsque les rqt sont envoyées à api)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# CONFIG
# =========================
BASE_URL = "https://localhost/Acumatica24R2"
USERNAME = "admin"
PASSWORD = "P@ssw0rd"
TENANT = "Acumatica24R2"

HEADERS = {
    "Content-Type": "application/json", #données que jenvoie au format json
    "Accept": "application/json"#donn recues au format json
}

ENDPOINTS = {
    "StockItem": "/entity/Default/23.200.001/StockItem/?$expand=WarehouseDetails,VendorDetails,UOMConversions,CrossReferences,Boxes,Categories,Attributes",
    "SalesOrder": "/entity/Default/23.200.001/SalesOrder/?$expand=Details",
    "Shipment": "/entity/Default/23.200.001/Shipment/?$expand=Details",
    "PurchaseOrder": "/entity/Default/23.200.001/PurchaseOrder/?$expand=Details,ShippingInstructions,TaxDetails",
    "PurchaseReceipt": "/entity/Default/23.200.001/PurchaseReceipt/?$expand=Details",
}

PAGE_SIZE = 200

# =========================
# 
# =========================
#fnct sert à extraire les enregis renvoyés par lapi
def extract_records(data):
    #verifie si data est un dict 
    if isinstance(data, dict):
        #si un dic on cherche la clé value et fonct retourne contenu
        return data.get("value", [data])#[data]= si value n'existe, retourne dict sous forme de liste
    elif isinstance(data, list):#si data déjà liste api renvoie direct une liste
        return data
    #si ni dic ni liste renvoie liste vide
    return []
#recu tte les donn dun endpoint
def get_all_data(session, endpoint):#session http pour envoyer les req à api
    #liste vide qui contient les enreg recup
    all_records = []
    #contruit ladr complète de lapi exp : https://api.cegid.com/stocks
    url = f"{BASE_URL}{endpoint}"
#boucle princ : continue tant quil existe une url
    while url:
        resp = session.get(url, headers=HEADERS, verify=False)
        resp.raise_for_status()#verifie si req s'est bien deroulée
        data = resp.json()#rep convertie en obj python
        #appelle au fonct pour extraitre la liste des enregistre
        records = extract_records(data)
#si ya plus de donn quitte la boucle
        if not records:
            break
#ajoute  les nv enregis à la liste existante
        all_records.extend(records)
        print(f"   → {len(all_records)} lignes récupérées...")

        # Pagination OData: ki yebda fama barcha donn tebath b chwy b chwy
        #à la fin de la 1ere page elle ajoute lien vers la page suivante
        #verifie que la rép est un dic
        if isinstance(data, dict):
            #on cherche @odata.nextLink
            url = data.get("@odata.nextLink")
            #si existe page suivante donc on continue
            if url:#on enléve lurl base_url bch nkhaliw juste chemin relatif exp: /stocks?$skip=1000
                url = url.replace(BASE_URL, "")
        #cas sans paginatio 
        else:
            url = None  # si c’est une liste simple, on arrête

    return all_records

# =========================
# EXPLOSION DES CHAMPS $EXPAND
# =========================
def explode_expand(records, expand_field):
    """Transforme les champs imbriqués en plusieurs lignes."""
    expanded_rows = []

    for record in records:
        nested_items = record.get(expand_field, [])

        # Base record (sans le champ imbriqué)
        base_record = {}
        #parcourt tous les champs
        for k, v in record.items():
            #ignorer le champ imbriqué
            if k != expand_field:
                #cas particulier certaines api odata renvoient 
                #{"nom":{"value":pc}} 
                if isinstance(v, dict) and "value" in v:
                    #on recupere juste "pc" 
                    base_record[k] = v["value"]
                else:
                    base_record[k] = v

        # Si pas de données imbriquées, garder la ligne
        if not nested_items:
            expanded_rows.append(base_record)
            continue

        # Créer une ligne par élément imbriqué
        for item in nested_items:
            if isinstance(item, dict):
                #copier les info principales
                new_row = base_record.copy()
                #ajouter les données imbriquées
                for key, value in item.items():
                    if isinstance(value, dict) and "value" in value:
                        new_row[key] = value["value"]
                    else:
                        new_row[key] = value
                expanded_rows.append(new_row)
            else:
                # élément non dict 
                new_row = base_record.copy()
                #ajoute direc la valeur 
                new_row[expand_field] = item
                expanded_rows.append(new_row)

    return expanded_rows

# =========================
# SESSION ACUMATICA
# =========================
#session permet de conserver les info de connexion 
session = requests.Session()
login_url = f"{BASE_URL}/entity/auth/login"
#contient les infos envoyées a lapi
login_payload = {"name": USERNAME, "password": PASSWORD, "tenant": TENANT}
session.post(login_url, json=login_payload, headers=HEADERS, verify=False).raise_for_status()
print(" Connexion Acumatica réussie")

# =========================
# EXPORT EXCEL
# =========================
with pd.ExcelWriter("acumatica_export.xlsx", engine="openpyxl") as writer:
    for sheet_name, endpoint in ENDPOINTS.items():
        print(f"\n Récupération {sheet_name}")
        records = get_all_data(session, endpoint)

        if not records:
            print(f" Aucune donnée pour {sheet_name}")
            continue

        # Exploser les champs imbriqués selon le sheet
        if sheet_name == "StockItem":
            for field in ["WarehouseDetails", "VendorDetails",
                          "UOMConversions", "CrossReferences",
                          "Boxes", "Categories", "Attributes"]:
                records = explode_expand(records, field)

        elif sheet_name == "PurchaseOrder":
            for field in ["Details", "ShippingInstructions", "TaxDetails"]:
                records = explode_expand(records, field)

        elif sheet_name in ["SalesOrder", "Shipment", "PurchaseReceipt"]:
            records = explode_expand(records, "Details")

        df = pd.DataFrame(records)
        #renommage des col certa col contiennent point donc devient _
        df.columns = [c.replace(".", "_") for c in df.columns]
        #ecrire le df en feuille excel
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f" {sheet_name} exporté ({len(df)} lignes)")

# =========================
# LOGOUT
# =========================
session.post(f"{BASE_URL}/entity/auth/logout", headers=HEADERS, verify=False)
print(" Déconnexion Acumatica")
print(" Fichier créé : acumatica_export.xlsx")

