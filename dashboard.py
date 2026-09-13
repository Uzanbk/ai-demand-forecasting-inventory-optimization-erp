"""
=============================================================
 DASHBOARD STREAMLIT v4 — Thème Cegid XRP Flex
 Prévision demande & Optimisation stocks
 LLM : Ollama llama3.2:3b
 Couleurs Cegid : Bleu #0057B8 · Rouge #E8332A · Fond #F5F6FA
=============================================================
 Lancer : streamlit run dashboard_cegid_v4.py
=============================================================
"""
#nrmlm pour cree app web il faut html css javasc+ backend avec streamlit faire une app web uni avec py
import streamlit as st#biblio qui cree des app web, dash ia etc...
import pandas as pd
import numpy as np
import plotly.express as px#biblio pour cree des graph interac(rapide et simple)
import plotly.graph_objects as go#de mm mais plus detaillé et puissant
import requests
from datetime import datetime

# ════════════════════════════════════════════════════════════
#  CONFIG PAGE
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Intelligence — Cegid",
    page_icon="📦",
    layout="wide",#largeur(wide: utilise tout lecran )
    initial_sidebar_state="expanded"#ouverte
)

# ── Thème Cegid complet ──────────────────────────────────────
#st.markdown sert à écrire texte sur page web avec mise en forme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

    /* ─── CACHER LE BOUTON DEPLOY ET MENU ─── */
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .stDeployButton { display: none !important; }
    footer { visibility: hidden !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* ─── VARIABLES CEGID ─── */
    :root {
        --cegid-blue:         #0057B8;
        --cegid-blue-dark:    #00418A;
        --cegid-blue-light:   #E8F0FB;
        --cegid-blue-mid:     #4A90D9;
        --cegid-red:          #E8332A;
        --cegid-red-light:    #FDE8E7;
        --cegid-gray-bg:      #F5F6FA;
        --cegid-gray-card:    #FFFFFF;
        --cegid-gray-border:  #E2E6EF;
        --cegid-gray-text:    #6B7280;
        --cegid-text:         #1A2340;
        --cegid-green:        #0A8A5C;
        --cegid-green-light:  #E6F5F0;
        --cegid-orange:       #E87D0D;
        --cegid-orange-light: #FEF3E5;
        --cegid-purple:       #5B3EBC;
        --cegid-purple-light: #EEE9FB;
    }

    /* ─── RESET STREAMLIT ─── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 15px !important;
    }
    p, li, span, td, th, label, div { font-size: 15px; }
    .main { background-color: #F5F6FA !important; }
    .block-container { padding-top: 0 !important; padding-bottom: 2rem; max-width: 100% !important; }
    section[data-testid="stSidebar"] { background-color: #f1f1ec !important; border-right: 1px solid #E2E6EF; }

    /* ─── TOPBAR CEGID ─── */
    .cegid-topbar {
        background: #f1f1ec;
        border-bottom: 3px solid #0057B8;
        padding: 14px 28px;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .cegid-logo {
        font-size: 32px;
        font-weight: 700;
        color: #0057B8;
        letter-spacing: -1px;
        line-height: 1;
    }
    .cegid-logo-dot { color: #E8332A; }
    .cegid-topbar-sep {
        width: 1px;
        height: 28px;
        background: #E2E6EF;
        margin: 0 4px;
    }
    .cegid-topbar-title {
        font-size: 15px;
        color: #6B7280;
        font-weight: 400;
    }
    .cegid-topbar-right {
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .cegid-badge {
        font-size: 13px;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 20px;
    }
    .cegid-badge-green {
        background: #E6F5F0;
        color: #0A8A5C;
        border: 1px solid #B3E0CF;
    }
    .cegid-badge-gray {
        background: #F5F6FA;
        color: #6B7280;
        border: 1px solid #E2E6EF;
    }
    .cegid-date {
        font-size: 13px;
        color: #6B7280;
    }

    /* ─── SIDEBAR ─── */
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 15px !important;
        color: #1A2340 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    .sidebar-logo-block {
        background: linear-gradient(135deg, #0057B8 0%, #003D8F 100%);
        padding: 22px 24px 18px 24px;
        margin: -1rem -1rem 1rem -1rem;
    }
    .sidebar-logo {
        font-size: 36px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -1px;
        line-height: 1;
    }
    .sidebar-logo-dot { color: #E8332A; }
    .sidebar-sub {
        font-size: 13px;
        color: rgba(255,255,255,0.75);
        margin-top: 5px;
        font-weight: 400;
    }

    /* ─── NAVIGATION ITEMS AVEC ICÔNES SVG ─── */
    /* Style de base pour les radio buttons de navigation */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 4px !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        padding: 10px 14px !important;
        border-radius: 8px !important;
        transition: background 0.15s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: #F0F5FF !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: #E8F0FB !important;
        color: #0057B8 !important;
        font-weight: 600 !important;
    }

    /* ─── METRICS CEGID ─── */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E6EF !important;
        border-top: 4px solid #0057B8 !important;
        border-radius: 10px !important;
        padding: 18px 20px 16px 20px !important;
        min-height: 110px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
        transition: box-shadow 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 16px rgba(0,87,184,0.10) !important;
    }
    div[data-testid="stMetric"] label {
        color: #6B7280 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1A2340 !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 13px !important;
    }
    /* Forcer toutes les colonnes de métriques à la même hauteur */
    [data-testid="column"] > div > div > div[data-testid="stMetric"] {
        height: 110px !important;
    }

    /* ─── METRIC COULEURS PAR STATUT ─── */
    .metric-critique div[data-testid="stMetric"] { border-top-color: #E8332A !important; }
    .metric-attention div[data-testid="stMetric"] { border-top-color: #E87D0D !important; }
    .metric-ok div[data-testid="stMetric"] { border-top-color: #0A8A5C !important; }
    .metric-anomalie div[data-testid="stMetric"] { border-top-color: #5B3EBC !important; }
    .metric-gain div[data-testid="stMetric"] { border-top-color: #0A8A5C !important; }

    /* ─── PANNEAUX IA ─── */
    .ai-panel {
        background: #F0F9F5;
        border: 1px solid #B3E0CF;
        border-left: 4px solid #0A8A5C;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0 16px 0;
        font-size: 15px;
        line-height: 1.7;
        color: #1A3530;
    }
    .ai-panel::before {
        content: "✦  Analyse IA";
        display: block;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #0A8A5C;
        margin-bottom: 8px;
    }
    .ai-panel-anomalie {
        background: #F5F0FD;
        border-color: #C9B8F0;
        border-left-color: #5B3EBC;
        color: #2A1A50;
    }
    .ai-panel-anomalie::before { content: "✦  Analyse Anomalie IA"; color: #5B3EBC; }
    .ai-panel-reco {
        background: #FEF8EE;
        border-color: #F5D9A8;
        border-left-color: #E87D0D;
        color: #3D2000;
    }
    .ai-panel-reco::before { content: "✦  Recommandations IA"; color: #E87D0D; }
    .ai-panel-kpi {
        background: #EBF3FD;
        border-color: #A8C8F0;
        border-left-color: #0057B8;
        color: #0D2A50;
    }
    .ai-panel-kpi::before { content: "✦  Interprétation KPI"; color: #0057B8; }

    /* ─── CHAT ─── */
    .chat-user {
        background: #F5F6FA;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #0057B8;
        font-size: 15px;
        color: #1A2340;
    }
    .chat-agent {
        background: #F0F9F5;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #0A8A5C;
        font-size: 15px;
        color: #1A3530;
    }

    /* ─── TABLEAUX ─── */
    .stDataFrame {
        border: 1px solid #E2E6EF !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* ─── DIVIDER ─── */
    hr { border-color: #E2E6EF !important; }

    /* ─── TITRES DE SECTION ─── */
    h1 { color: #1A2340 !important; font-size: 22px !important; font-weight: 700 !important; }
    h2 { color: #1A2340 !important; font-size: 18px !important; font-weight: 600 !important; }
    h3 { color: #1A2340 !important; font-size: 16px !important; font-weight: 600 !important; }

    /* ─── PILLS STATUT ─── */
    .pill-critique {
        background: #FDE8E7; color: #E8332A; font-size: 11px;
        font-weight: 700; padding: 2px 9px; border-radius: 12px;
        display: inline-block;
    }
    .pill-attention {
        background: #FEF3E5; color: #E87D0D; font-size: 11px;
        font-weight: 700; padding: 2px 9px; border-radius: 12px;
        display: inline-block;
    }
    .pill-ok {
        background: #E6F5F0; color: #0A8A5C; font-size: 11px;
        font-weight: 700; padding: 2px 9px; border-radius: 12px;
        display: inline-block;
    }
    .pill-anomalie {
        background: #EEE9FB; color: #5B3EBC; font-size: 11px;
        font-weight: 700; padding: 2px 9px; border-radius: 12px;
        display: inline-block;
    }

    /* ─── SPINNER ─── */
    .stSpinner > div { border-top-color: #0057B8 !important; }

    /* ─── BOUTONS ─── */
    .stButton > button {
        background: #FFFFFF !important;
        color: #0057B8 !important;
        border: 1px solid #0057B8 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #E8F0FB !important;
        box-shadow: 0 2px 8px rgba(0,87,184,0.15) !important;
    }

    /* ─── MULTISELECT / SELECTBOX ─── */
    .stMultiSelect [data-baseweb="tag"] {
        background: #E8F0FB !important;
        color: #0057B8 !important;
    }

    /* ─── ALIGNEMENT COLONNES MÉTRIQUES ─── */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] > div > div {
        flex: 1 !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] > div > div > div[data-testid="stMetric"] {
        height: 100% !important;
        min-height: 110px !important;
    }

    /* ─── CAPTION / SOUS-TITRES ─── */
    [data-testid="stCaptionContainer"] p {
        font-size: 13px !important;
        color: #6B7280 !important;
    }

    /* ─── SECTION HEADER AVEC ICÔNE SVG ─── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .section-header-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .section-header-icon-blue   { background: #E8F0FB; }
    .section-header-icon-green  { background: #E6F5F0; }
    .section-header-icon-orange { background: #FEF3E5; }
    .section-header-icon-purple { background: #EEE9FB; }
    .section-header-text {
        font-size: 17px;
        font-weight: 600;
        color: #1A2340;
    }

    /* ─── NAV ITEM LABELS avec icônes SVG inline ─── */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 12px;
        border-radius: 8px;
        margin: 2px 0;
        cursor: pointer;
        transition: background 0.15s;
        font-size: 14px;
        font-weight: 500;
        color: #374151;
    }
    .nav-item:hover { background: #F0F5FF; }
    .nav-item.active {
        background: #E8F0FB;
        color: #0057B8;
        font-weight: 600;
    }
    .nav-item svg { flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)#autorise streamlit à interpréter du html brut


# ════════════════════════════════════════════════════════════
#  ICÔNES SVG SOPHISTIQUÉES (remplace les emojis)
# ════════════════════════════════════════════════════════════
# Chaque icône est un SVG inline propre, style Lucide/Phosphor

def svg_icon(name, size=18, color="#0057B8"):
    """Retourne un SVG inline selon le nom de l'icône"""
    icons = {
        # Icône tableau de bord — grille de carrés
        "dashboard": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="3" width="7" height="7" rx="1.5" fill="{color}" opacity="0.9"/>
            <rect x="14" y="3" width="7" height="7" rx="1.5" fill="{color}" opacity="0.5"/>
            <rect x="3" y="14" width="7" height="7" rx="1.5" fill="{color}" opacity="0.5"/>
            <rect x="14" y="14" width="7" height="7" rx="1.5" fill="{color}" opacity="0.7"/>
        </svg>""",

        # Icône boîte / stock — cube 3D
        "stock": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L21 7V17L12 22L3 17V7L12 2Z" stroke="{color}" stroke-width="1.8" stroke-linejoin="round" fill="{color}" fill-opacity="0.08"/>
            <path d="M12 2L21 7M12 2L3 7M12 22V12M21 7L12 12M3 7L12 12" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""",

        # Icône anomalie / alerte — triangle avec point d'exclamation
        "anomaly": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="{color}" fill-opacity="0.08"/>
            <line x1="12" y1="9" x2="12" y2="13" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
            <circle cx="12" cy="17" r="1" fill="{color}"/>
        </svg>""",

        # Icône recommandations — ampoule / idée
        "reco": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 21h6M10 17.5A7 7 0 119 10c0 2.5 1 4 2 5.5" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M15 17.5A7 7 0 1115 10" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M9 17.5h6" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
        </svg>""",

        # Icône KPI — graphe en ligne montante
        "kpi": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <polyline points="22 12 18 8 13 13 9 9 2 16" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 8h6v6" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""",

        # Icône chatbot — bulle de dialogue avec éclairs IA
        "chat": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="{color}" fill-opacity="0.07"/>
            <path d="M10 10l1.5-3 1.5 3" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M14 10l1.5-3 1.5 3" stroke="{color}" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
        </svg>""",

        # Icône refresh — flèches circulaires
        "refresh": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M1 4v6h6" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M23 20v-6h-6" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M20.49 9A9 9 0 005.64 5.64L1 10M23 14l-4.64 4.36A9 9 0 013.51 15" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""",

        # Icône IA — cerveau stylisé
        "ai": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="3" fill="{color}" opacity="0.8"/>
            <path d="M12 3v2M12 19v2M3 12h2M19 12h2" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M5.64 5.64l1.41 1.41M16.95 16.95l1.41 1.41M5.64 18.36l1.41-1.41M16.95 7.05l1.41-1.41" stroke="{color}" stroke-width="1.6" stroke-linecap="round" opacity="0.6"/>
        </svg>""",

        # Icône calendrier
        "calendar": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="4" width="18" height="18" rx="2" stroke="{color}" stroke-width="1.8"/>
            <path d="M3 9h18" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M8 2v2M16 2v2" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
        </svg>""",

        # Icône article / tag produit
        "article": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="{color}" fill-opacity="0.07"/>
            <circle cx="7" cy="7" r="1.5" fill="{color}"/>
        </svg>""",
    }
    return icons.get(name, "")


# ════════════════════════════════════════════════════════════
#  creation du TOPBAR CEGID
# ════════════════════════════════════════════════════════════

def afficher_topbar(titre_page, ollama_ok):#ollama=ok: état de lia
    #si ia fonctionne badge vert, snn badge gris ,résultat visuel(● IA active)
    badge_ollama = (
        '<span class="cegid-badge cegid-badge-green">● IA active</span>'
        if ollama_ok else
        '<span class="cegid-badge cegid-badge-gray">○ IA inactive</span>'
    )
    #utilisation du html/css custom
    st.markdown(f"""
    <div class="cegid-topbar"> <!--conteneur principa-->
        <div class="cegid-logo">cegid<span class="cegid-logo-dot">.</span></div>
        <div class="cegid-topbar-sep"></div> <!-- ligne espace visuel entre logo et titre-->
        <div class="cegid-topbar-title">Stock Intelligence — {titre_page}</div>
        <div class="cegid-topbar-right">  <!-- partie droite contient badge et date-->
            {badge_ollama}
            <span class="cegid-date">{datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CHARGEMENT DONNÉES
# ════════════════════════════════════════════════════════════
#sans cache à chq clic dans l'app excel rechargé, lent
@st.cache_data#décorateur streamlit: évite de recharger les fich excel à chaque interaction utilisateur
def charger_donnees():
    pred_dem     = pd.read_excel("predictions_demande.xlsx")
    pred_rup     = pd.read_excel("predictions_rupture.xlsx")
    pred_ano     = pd.read_excel("predictions_anomalie_stock.xlsx")
    perf_art     = pd.read_excel("performances_par_article.xlsx")
    perf_rup_art = pd.read_excel("performances_rupture_par_article.xlsx")
    perf_ano_art = pd.read_excel("performances_anomalie_par_article.xlsx")
    res_reg      = pd.read_excel("resultats_regression.xlsx")
    base         = pd.read_excel("base_ml_finale.xlsx")
#conversion des dates
    for df in [pred_dem, pred_rup, pred_ano, base]:
        #avant: "2026-05-20" , après: datetime(2026, 5, 20)
        df["date_debut_semaine"] = pd.to_datetime(df["date_debut_semaine"])

    return pred_dem, pred_rup, pred_ano, perf_art, perf_rup_art, perf_ano_art, res_reg, base


@st.cache_data
#tableau comme si état acturel des articles
def construire_tableau(pred_dem, pred_rup, pred_ano, perf_art, perf_rup_art, perf_ano_art, base):
    articles_ml = pred_dem["identifiant_article"].unique()
    cols = ["identifiant_article", "stock_disponible", "prix_vente", "cout_achat", "lead_time_hist_moy"]
    statiques = (
        base[base["identifiant_article"].isin(articles_ml)][cols]#garder seul les col importantes
        .groupby("identifiant_article").last().reset_index()
        #last(): prend la dernière lig de chq article(dernier état connu)
    )
#résume demande par article
    stats_dem = pred_dem.groupby("identifiant_article").agg(
        demande_moy=("reel", "mean"),
        demande_std=("reel", "std"),
        prediction_s1=("prediction", "last"),
        derniere_semaine=("date_debut_semaine", "max"),
    ).reset_index()
#mettre les sem dans lordre et prend les 4 dernières sem et prend la plus forte proba recente
    proba_recente = (
        pred_rup.sort_values("date_debut_semaine")
        .groupby("identifiant_article")
        .apply(lambda x: x.tail(4)["proba_rupture"].max())
        .reset_index()
    )
    proba_recente.columns = ["identifiant_article", "proba_rupture_max"]

    proba_ano_recente = (
        pred_ano.sort_values("date_debut_semaine")
        .groupby("identifiant_article")
        .apply(lambda x: x.tail(4)["proba_anomalie"].max())
        .reset_index()
    )
    proba_ano_recente.columns = ["identifiant_article", "proba_anomalie_max"]

    df = (
        statiques
        .merge(stats_dem,          on="identifiant_article", how="left")
        .merge(proba_recente,      on="identifiant_article", how="left")
        .merge(proba_ano_recente,  on="identifiant_article", how="left")
        .merge(perf_art[["identifiant_article", "mae_article", "wmape_%"]], on="identifiant_article", how="left")
        .merge(perf_rup_art[["identifiant_article", "ruptures_reelles", "ruptures_detectees", "taux_detection_%"]], on="identifiant_article", how="left")
        .merge(perf_ano_art[["identifiant_article", "anomalies_reelles", "anomalies_detectees"]], on="identifiant_article", how="left")
    )

    df["couverture_sem"] = (df["stock_disponible"] / df["demande_moy"].replace(0, np.nan)).round(1)
    df["valeur_stock"]   = df["stock_disponible"] * df["prix_vente"]

    df["statut"] = df["proba_rupture_max"].apply(
        lambda x: "CRITIQUE" if x >= 0.90 else ("ATTENTION" if x >= 0.55 else "OK")
    )
    df["statut_anomalie"] = df["proba_anomalie_max"].apply(
        lambda x: "SURSTOCK/ANTICIPATION" if x >= 0.85 else ("À SURVEILLER" if x >= 0.55 else "NORMAL")
    )

    ordre = {"CRITIQUE": 0, "ATTENTION": 1, "OK": 2}
    df["_tri"] = df["statut"].map(ordre)
    df = df.sort_values(["_tri", "proba_rupture_max"], ascending=[True, False]).drop(columns=["_tri"])
    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
#  OLLAMA
# ════════════════════════════════════════════════════════════

FOUNDRY_URL    = "https://stock-foundry.services.ai.azure.com/openai/v1/chat/completions"
FOUNDRY_KEY    = "AYFRhQGFD3XO4vWtXBQOT1Qg8d00wuGe4mv67olAJULIK81jRlWqJQQJ99CFAC5T7U2XJ3w3AAAAACOG813S"
FOUNDRY_MODEL  = "Llama-3.3-70B-Instruct"

def verifier_ollama():
    # Vérifie que l'endpoint Azure Foundry est accessible
    try:
        r = requests.post(
            FOUNDRY_URL,
            headers={
                "api-key": FOUNDRY_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": FOUNDRY_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            timeout=10,
        )
        return r.status_code in (200, 201)
    except:
        return False

def appeler_ollama(prompt, max_tokens=800):
    # Appel Azure Foundry — Llama-3.3-70B-Instruct
    try:
        r = requests.post(
            FOUNDRY_URL,
            headers={
                "api-key": FOUNDRY_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": FOUNDRY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.25,
            },
            timeout=120,
        )
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ LLM indisponible : {e}"


# ─── Prompts ────────────────────────────────────────────────
#traducteur dataframe en langage naturel
def prompt_analyse_overview(df, gain, total_rup, total_det):
    critiques = df[df["statut"] == "CRITIQUE"][
        ["identifiant_article", "stock_disponible", "couverture_sem", "proba_rupture_max"]
    ].to_string(index=False)
    return f"""Tu es expert en gestion de stocks. Réponse en français, 4-5 phrases, ton professionnel.
Données : {len(df[df['statut']=='CRITIQUE'])} CRITIQUES, {len(df[df['statut']=='ATTENTION'])} ATTENTION, {len(df[df['statut']=='OK'])} OK.
Gain ML vs baseline : +{gain:.1f}%. Ruptures détectées : {total_det}/{total_rup}.
Articles critiques :
{critiques}
Donne une analyse synthétique pour un directeur des achats. Mets en avant les risques prioritaires."""


def prompt_analyse_rupture(article, row):
    return f"""Tu es expert en gestion de stocks. Réponse en français, 3-4 phrases.
Article : {article}
Stock : {int(row.get('stock_disponible', 0))} unités | Couverture : {row.get('couverture_sem', '?')} sem
Demande moy : {row.get('demande_moy', 0):.0f} u/sem | P(rupture) : {row.get('proba_rupture_max', 0):.0%}
wMAPE : {row.get('wmape_%', '?')}%
Explique le risque, la fiabilité ML et une recommandation concrète."""


def prompt_analyse_anomalie(df, pred_ano):
    anomalies = df[df["statut_anomalie"] != "NORMAL"][
        ["identifiant_article", "proba_anomalie_max", "statut_anomalie", "couverture_sem", "valeur_stock"]
    ].to_string(index=False)
    n_sur = pred_ano["surstock_reel"].sum() if "surstock_reel" in pred_ano.columns else "N/A"
    n_ant = pred_ano["mauvaise_anticipation_reel"].sum() if "mauvaise_anticipation_reel" in pred_ano.columns else "N/A"
    return f"""Tu es expert en logistique. Réponse en français, 4-5 phrases, ton décisionnel.
Anomalies (surstock > 8 sem OU commandes > 2× demande) :
{anomalies}
Surstock : {n_sur} semaines | Mauvaise anticipation : {n_ant} semaines
Impact financier, risques, 2 recommandations concrètes."""


def prompt_reco_commandes(df):
    df_reco = df[df["qte_commander"] > 0][
        ["identifiant_article", "statut", "qte_commander", "valeur_commande", "couverture_sem", "proba_rupture_max"]
    ].head(10)
    return f"""Tu es expert en optimisation de stocks. Réponds en français, max 200 mots.
Budget total : {df['valeur_commande'].sum():,.0f} €
Articles à commander :
{df_reco.to_string(index=False)}

3 parties COURTES :
1. PRIORITÉS (2-3 phrases)
2. LOGIQUE DE CALCUL (2 phrases)
3. CONSEILS BUDGET (2-3 phrases)"""


def prompt_kpi(mae_best, mape_best, gain, total_rup, total_det, f1_ano, rec_ano, df):
    return f"""Tu es data scientist expert. Réponse en français, 5-6 phrases, sans jargon.
Régression : MAE={mae_best:.0f} u, wwMAPE={mape_best:.1f}%, gain=+{gain:.1f}%
Rupture : {total_det}/{total_rup} détectées
Anomalie : F1={f1_ano:.3f}, Rappel={rec_ano:.1%}
Stock total : {df['valeur_stock'].sum():,.0f} €
Est-ce fiable ? Que signifie concrètement pour la gestion des stocks ? Quelles limites ?"""


def repondre_depuis_donnees(question, df, pred_ano):
    q = question.lower()

    # Detection article specifique mentionne dans la question
    article_mentionne = None
    for art in df["identifiant_article"].tolist():
        if art.lower() in q:
            article_mentionne = art
            break

    if article_mentionne is not None:
        r = df[df["identifiant_article"] == article_mentionne].iloc[0]

        # Détection de ce qui est demandé spécifiquement
        if any(k in q for k in ["anomalie", "surstock", "anticipation"]):
            return (
                f"**{article_mentionne}** — Probabilité anomalie : **{r['proba_anomalie_max']:.0%}** "
                f"| Statut anomalie : **{r['statut_anomalie']}** "
                f"| Couverture : {r['couverture_sem']} sem "
                f"| Stock : {int(r['stock_disponible'])} u"
            ), True

        elif any(k in q for k in ["rupture", "critique", "urgent"]):
            return (
                f"**{article_mentionne}** — Probabilité rupture : **{r['proba_rupture_max']:.0%}** "
                f"| Statut : **{r['statut']}** "
                f"| Couverture : {r['couverture_sem']} sem "
                f"| Stock : {int(r['stock_disponible'])} u "
                f"| Prévision S+1 : {int(r['prediction_s1'])} u"
            ), True

        elif any(k in q for k in ["stock", "disponible"]):
            return (
                f"**{article_mentionne}** — Stock disponible : **{int(r['stock_disponible'])} u** "
                f"| Couverture : {r['couverture_sem']} sem "
                f"| Demande moyenne : {r['demande_moy']:.0f} u/sem"
            ), True

        elif any(k in q for k in ["prévision", "prevision", "demande"]):
            return (
                f"**{article_mentionne}** — Prévision S+1 : **{int(r['prediction_s1'])} u** "
                f"| Demande moyenne : {r['demande_moy']:.0f} u/sem "
                f"| wMAPE : {r['wmape_%']:.1f}%"
            ), True

        elif any(k in q for k in ["prix", "valeur", "coût", "cout"]):
            return (
                f"**{article_mentionne}** — Prix vente : **{r['prix_vente']:.2f} €** "
                f"| Coût achat : {r['cout_achat']:.2f} € "
                f"| Valeur stock : {r['valeur_stock']:,.0f} €"
            ), True

        else:
            # Fiche complète si question générale sur l'article
            lignes = [
                f"**{article_mentionne}**",
                f"- Statut rupture : {r['statut']} (P={r['proba_rupture_max']:.0%})",
                f"- Statut anomalie : {r['statut_anomalie']} (P={r['proba_anomalie_max']:.0%})",
                f"- Stock disponible : {int(r['stock_disponible'])} u",
                f"- Couverture : {r['couverture_sem']} semaines",
                f"- Demande moyenne : {r['demande_moy']:.0f} u/sem",
                f"- Prévision S+1 : {int(r['prediction_s1'])} u",
                f"- wMAPE : {r['wmape_%']:.1f}%",
                f"- Prix vente : {r['prix_vente']:.2f} €",
                f"- Valeur stock : {r['valeur_stock']:,.0f} €",
                f"- Lead time : {r['lead_time_hist_moy']:.1f} sem",
            ]
            return "\n".join(lignes), True

    reponse = None

    if any(k in q for k in ["urgent", "critique", "commander en premier", "priorité", "prioritaires"]):
        critiques = df[df["statut"] == "CRITIQUE"].sort_values("proba_rupture_max", ascending=False)
        if len(critiques) == 0:
            return "Aucun article critique détecté actuellement.", True
        lignes = []
        for _, r in critiques.head(5).iterrows():
            qte = max(0, int(r["demande_moy"] * r["lead_time_hist_moy"] * 1.2 - r["stock_disponible"]))
            lignes.append(f"• **{r['identifiant_article']}** — stock={int(r['stock_disponible'])} u, couverture={r['couverture_sem']} sem, P(rupture)={r['proba_rupture_max']:.0%}, prévision S+1={int(r['prediction_s1'])} u, qté à commander≈{qte} u")
        return f"**{len(critiques)} articles CRITIQUES**, classés par urgence :\n\n" + "\n".join(lignes), True

    elif any(k in q for k in ["surstock", "anomalie", "mauvaise anticipation", "immobilisé", "capital"]):
        anomalies = df[df["statut_anomalie"] != "NORMAL"].sort_values("proba_anomalie_max", ascending=False)
        if len(anomalies) == 0:
            return "Aucune anomalie détectée.", True
        capital = anomalies["valeur_stock"].sum()
        lignes = [f"• **{r['identifiant_article']}** — {r['statut_anomalie']}, couverture={r['couverture_sem']} sem, valeur={r['valeur_stock']:,.0f} €, P(anomalie)={r['proba_anomalie_max']:.0%}" for _, r in anomalies.iterrows()]
        return f"**{len(anomalies)} articles** avec anomalie. Capital immobilisé : **{capital:,.0f} €**\n\n" + "\n".join(lignes), True

    elif any(k in q for k in ["risque financier", "risque", "perte", "si on ne commande"]):
        critiques = df[df["statut"] == "CRITIQUE"]
        risque = (critiques["prediction_s1"] * critiques["prix_vente"]).sum()
        lignes = [f"• **{r['identifiant_article']}** — {int(r['prediction_s1'])} u × {r['prix_vente']:.2f} € = {r['prediction_s1']*r['prix_vente']:,.0f} €" for _, r in critiques.iterrows()]
        return f"Risque financier total : **{risque:,.0f} €**\n\n" + "\n".join(lignes), True

    elif any(k in q for k in ["fiable", "wmape", "mape", "erreur", "peu fiable"]):
        peu_fiables = df[df["wmape_%"].notna()].sort_values("wmape_%", ascending=False).head(5)
        lignes = [f"• **{r['identifiant_article']}** — wMAPE={r['wmape_%']:.1f}%, demande_moy={r['demande_moy']:.0f} u/sem" for _, r in peu_fiables.iterrows()]
        return "Articles avec prévisions les moins fiables :\n\n" + "\n".join(lignes), True

    elif any(k in q for k in ["stock", "état", "etat", "disponible", "couverture"]):
        total_val = df["valeur_stock"].sum()
        bas = df.sort_values("couverture_sem").head(5)
        lignes = [f"• **{r['identifiant_article']}** — {r['couverture_sem']} sem, stock={int(r['stock_disponible'])} u" for _, r in bas.iterrows()]
        return (f"Valeur totale : {total_val:,.0f} € | {len(df[df['statut']=='CRITIQUE'])} CRITIQUES | {len(df[df['statut']=='ATTENTION'])} ATTENTION | {len(df[df['statut']=='OK'])} OK\n\n**Couverture la plus faible :**\n" + "\n".join(lignes)), True

    return None, False

def prompt_chatbot(contexte, question):
    return f"""{contexte}

INSTRUCTION : Réponds UNIQUEMENT avec les données du tableau fourni. Ne jamais inventer de chiffres.
Question : {question}
Réponse (français, précis) :"""


# ════════════════════════════════════════════════════════════
#  CHARGEMENT GLOBAL
# ════════════════════════════════════════════════════════════

pred_dem, pred_rup, pred_ano, perf_art, perf_rup_art, perf_ano_art, res_reg, base = charger_donnees()
df = construire_tableau(pred_dem, pred_rup, pred_ano, perf_art, perf_rup_art, perf_ano_art, base)

col_modele = "Modèle" if "Modèle" in res_reg.columns else "Modele"
mae_best   = res_reg.iloc[0]["MAE"]
mape_best  = res_reg.iloc[0]["wMAPE"]
mae_base   = res_reg[res_reg[col_modele].str.contains("Baseline")]["MAE"].values[0]
gain       = round((mae_base - mae_best) / mae_base * 100, 1)
total_rup  = int(perf_rup_art["ruptures_reelles"].sum())
total_det  = int(perf_rup_art["ruptures_detectees"].sum())
f1_ano     = perf_ano_art["taux_detection_%"].mean() / 100 if "taux_detection_%" in perf_ano_art.columns else 0
rec_ano    = perf_ano_art["anomalies_detectees"].sum() / max(perf_ano_art["anomalies_reelles"].sum(), 1)

OLLAMA_OK = verifier_ollama()

# ── Couleurs Plotly Cegid ────────────────────────────────────
CEGID_PLOTLY = {
    "bleu":    "#0057B8",
    "rouge":   "#E8332A",
    "vert":    "#0A8A5C",
    "orange":  "#E87D0D",
    "violet":  "#5B3EBC",
    "gris":    "#6B7280",
    "bleu2":   "#4A90D9",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFBFD",
    font=dict(family="DM Sans, system-ui, sans-serif", color="#1A2340"),
    margin=dict(l=8, r=8, t=32, b=8),
)


# ════════════════════════════════════════════════════════════
#  SIDEBAR — icônes SVG remplacent les emojis
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-block">
        <div class="sidebar-logo">cegid<span class="sidebar-logo-dot">.</span></div>
        <div class="sidebar-sub">Stock Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Labels de navigation avec icônes SVG intégrées dans le texte
    # Les icônes remplacent les emojis basiques par des SVG professionnels
    nav_labels = {
        "Vue d'ensemble":           svg_icon("dashboard",  18, "#0057B8") + "  Vue d'ensemble",
        "Stocks & Ruptures":        svg_icon("stock",      18, "#0057B8") + "  Stocks & Ruptures",
        "Anomalies stock":          svg_icon("anomaly",    18, "#0057B8") + "  Anomalies stock",
        "Recommandations":          svg_icon("reco",       18, "#0057B8") + "  Recommandations",
        "Indicateurs clés de performance":        svg_icon("kpi",        18, "#0057B8") + "  Indicateurs clés de performance",
        "Chatbot IA":               svg_icon("chat",       18, "#0057B8") + "  Chatbot IA",
    }

    # Streamlit radio — on utilise les clés simples pour la logique
    # et on affiche les SVG via format_func
    page_key = st.radio(
        "Navigation",
        list(nav_labels.keys()),
        format_func=lambda x: x,   # texte brut pour le label interne
        label_visibility="collapsed",
    )

    # Affiche les labels avec SVG via HTML custom sous le radio
    # (le radio Streamlit garde la logique, on injecte juste le visuel)
    st.markdown("""
    <style>
    /* Masquer les labels natifs du radio et les remplacer par nos SVG */
    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    section[data-testid="stSidebar"] .stRadio > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        display: flex !important;
        align-items: center !important;
        padding: 10px 12px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: background 0.15s !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: #F0F5FF !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: #E8F0FB !important;
        color: #0057B8 !important;
        font-weight: 600 !important;
    }
    /* Cacher les boutons radio natifs */
    section[data-testid="stSidebar"] .stRadio input[type="radio"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Raccourci de page pour le reste du code (garde la même logique qu'avant)
    page = "📊 Vue d'ensemble"      if page_key == "Vue d'ensemble"    else \
           "📦 Stocks & Ruptures"   if page_key == "Stocks & Ruptures" else \
           "⚠️ Anomalies stock"     if page_key == "Anomalies stock"   else \
           "🎯 Recommandations"     if page_key == "Recommandations"   else \
           "📈 Indicateurs clés de performance"   if page_key == "Indicateurs clés de performance" else \
           "💬 Chatbot IA"

    st.divider()
    # Icône calendrier + date dernière semaine ML
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;color:#6B7280;font-size:13px;">'
        f'{svg_icon("calendar", 14, "#6B7280")}'
        f'<span>Dernière semaine ML : {df["derniere_semaine"].max().strftime("%d/%m/%Y")}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.divider()

    if OLLAMA_OK:
        st.success("🟢 Azure Foundry actif — Llama 3.3 70B")
    else:
        st.warning("🔴 Azure Foundry inaccessible\nLes analyses IA seront désactivées.")
#bouton pour relancer les analyses ia
    refresh_ai = st.button("🔄 Rafraîchir analyse IA", disabled=not OLLAMA_OK)


# ════════════════════════════════════════════════════════════
#  MOTEUR d'affichage IA : affiche une rép de llm dans le dashboard streamlit
# ════════════════════════════════════════════════════════════
#classe="ai-panel": la classe utilisée pour afficher le bloc
#max_tokens=1200 : ne pas utiliser le cache meme si dispo
def bloc_ia(prompt, classe="ai-panel", key=None, force_refresh=False, max_tokens=1200):
    #créer une clé unique pour chaque prompt pour indexer résultat is dans un cache
    #si key est fourni et nest pas vide il est utilisé
    #snn on calcule à partir de prompt on prend les 80 premiers carac de prompt
    #hash():donne un entier unique abs: prend la val absolue pour éviter un hash négatif
    cache_key = f"ai_{key or abs(hash(prompt[:80]))}"
    if not OLLAMA_OK: #si ollama nest pas lancé
        st.markdown(
            f'<div class="{classe}">⚠️ Ollama non démarré — démarrez Ollama pour voir l\'analyse IA.</div>',
            unsafe_allow_html=True
        )
        return
    #cas où on appelle lia si boutton rafraichir cliqué ou cest la prem fois pas encore
    #de réponse stockée snn ne fait rien reutiliser la réponse
    if force_refresh or cache_key not in st.session_state:
        with st.spinner("Analyse IA en cours…"):#affiche msg pendant exécution du code
            #je demande à l'ia de réfléchir et je stocke la réponse
            st.session_state[cache_key] = appeler_ollama(prompt, max_tokens=max_tokens)
    st.markdown(
        f'<div class="{classe}">{st.session_state[cache_key]}</div>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════
#  HELPER : titre de section avec icône SVG
# ════════════════════════════════════════════════════════════

def section_title(label, icon_name, icon_color="#0057B8", bg_class="section-header-icon-blue"):
    """Affiche un titre h3 avec une icône SVG dans un badge coloré"""
    icon_svg = svg_icon(icon_name, 16, icon_color)
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-icon {bg_class}">{icon_svg}</div>
        <span class="section-header-text">{label}</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 1 — VUE D'ENSEMBLE
# ════════════════════════════════════════════════════════════
#affichage de la page vue densemble quand lutili selec dans le menu
if page == "📊 Vue d'ensemble":
    afficher_topbar("Vue d'ensemble", OLLAMA_OK)
    st.caption(f"{len(df)} articles · 3 modèles ML · Régression + Rupture + Anomalie")

    # ── KPI métriques ────────────────────────────────────────
    # Injection CSS ciblée pour colorier chaque carte par position
    st.markdown("""
    <style>
    /* Ligne de 6 métriques — colorisation par position */
    [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-child(1) div[data-testid="stMetric"] { border-top-color: #E8332A !important; }
    [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-child(2) div[data-testid="stMetric"] { border-top-color: #E87D0D !important; }
    [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-child(3) div[data-testid="stMetric"] { border-top-color: #0A8A5C !important; }
    [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-child(4) div[data-testid="stMetric"] { border-top-color: #5B3EBC !important; }
    [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-child(5) div[data-testid="stMetric"] { border-top-color: #0057B8 !important; }
    [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-child(6) div[data-testid="stMetric"] { border-top-color: #0A8A5C !important; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("CRITIQUES", len(df[df["statut"] == "CRITIQUE"]), delta="Rupture imminente", delta_color="inverse")
    with c2:
        st.metric("ATTENTION", len(df[df["statut"] == "ATTENTION"]), delta="À surveiller", delta_color="off")
    with c3:
        st.metric("OK", len(df[df["statut"] == "OK"]), delta="Stock suffisant", delta_color="normal")
    with c4:
        st.metric("Anomalies stock", len(df[df["statut_anomalie"] != "NORMAL"]), delta="Surstock / Antici.", delta_color="inverse")
    with c5:
        st.metric("Valeur stock", f"{df['valeur_stock'].sum():,.0f} €", delta=f"{len(df)} articles")
    with c6:
        st.metric("Gain ML", f"+{gain}%", delta="vs baseline MAE", delta_color="normal")

    st.divider()

    bloc_ia(
        prompt_analyse_overview(df, gain, total_rup, total_det),
        classe="ai-panel",
        key="overview_global",
        force_refresh=refresh_ai,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        section_title("Statut rupture des articles", "stock", "#0057B8", "section-header-icon-blue")
        fig_pie = px.pie(
            values=[
                len(df[df["statut"] == "CRITIQUE"]),
                len(df[df["statut"] == "ATTENTION"]),
                len(df[df["statut"] == "OK"]),
            ],
            names=["CRITIQUE", "ATTENTION", "OK"],
            color_discrete_sequence=[
                CEGID_PLOTLY["rouge"],
                CEGID_PLOTLY["orange"],
                CEGID_PLOTLY["vert"],
            ],
            hole=0.52,
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=300)
        fig_pie.update_traces(textfont_color="#1A2340")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        section_title("Statut anomalie des articles", "anomaly", "#5B3EBC", "section-header-icon-purple")
        fig_pie2 = px.pie(
            values=[
                len(df[df["statut_anomalie"] == "SURSTOCK/ANTICIPATION"]),
                len(df[df["statut_anomalie"] == "À SURVEILLER"]),
                len(df[df["statut_anomalie"] == "NORMAL"]),
            ],
            names=["SURSTOCK", "À SURVEILLER", "NORMAL"],
            color_discrete_sequence=[
                CEGID_PLOTLY["violet"],
                CEGID_PLOTLY["orange"],
                CEGID_PLOTLY["vert"],
            ],
            hole=0.52,
        )
        fig_pie2.update_layout(**PLOTLY_LAYOUT, height=300)
        fig_pie2.update_traces(textfont_color="#1A2340")
        st.plotly_chart(fig_pie2, use_container_width=True)

    # ── Tableau complet vue d'ensemble
    section_title("Tableau de bord — tous les articles", "article", "#0057B8", "section-header-icon-blue")
    df_display = df[[
        "identifiant_article", "statut", "statut_anomalie",
        "stock_disponible", "couverture_sem", "demande_moy",
        "proba_rupture_max", "proba_anomalie_max", "valeur_stock",
    ]].copy()
    df_display.columns = [
        "Article", "Statut rupture", "Statut anomalie",
        "Stock", "Couv. (sem)", "Demande moy",
        "P(rupture)", "P(anomalie)", "Valeur (€)",
    ]
    #transformer les proba en % affichables
    df_display["P(rupture)"]  = df_display["P(rupture)"].apply(lambda x: f"{x:.0%}")
    df_display["P(anomalie)"] = df_display["P(anomalie)"].apply(lambda x: f"{x:.0%}")
    df_display["Valeur (€)"]  = df_display["Valeur (€)"].apply(lambda x: f"{x:,.0f}")

    def color_statut(val):
        if val == "CRITIQUE":              return "background-color:#FDE8E7;color:#E8332A;font-weight:700"
        if val == "ATTENTION":             return "background-color:#FEF3E5;color:#E87D0D;font-weight:700"
        if val == "SURSTOCK/ANTICIPATION": return "background-color:#EEE9FB;color:#5B3EBC;font-weight:700"
        if val == "À SURVEILLER":          return "background-color:#FEF3E5;color:#E87D0D;font-weight:700"
        if val == "OK":                    return "background-color:#E6F5F0;color:#0A8A5C;font-weight:700"
        if val == "NORMAL":                return "background-color:#E6F5F0;color:#0A8A5C;font-weight:700"
        return ""

#applique fonct de sty sur les col concernés
    st.dataframe(
        df_display.style.map(color_statut, subset=["Statut rupture", "Statut anomalie"]),
        use_container_width=True,
        height=420,
    )


# ════════════════════════════════════════════════════════════
#  PAGE 2 — STOCKS & RUPTURES
# ════════════════════════════════════════════════════════════

elif page == "📦 Stocks & Ruptures":
    afficher_topbar("Stocks & Ruptures", OLLAMA_OK)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        statut_filtre = st.multiselect(
            "Filtrer par statut",
            ["CRITIQUE", "ATTENTION", "OK"],
            default=["CRITIQUE", "ATTENTION"],
        )
    with col_f2:
        article_select = st.selectbox("Analyse détaillée par article", df["identifiant_article"].tolist())

    df_filtre = df[df["statut"].isin(statut_filtre)] if statut_filtre else df
#si lutili choisit les stat cri, att,ok donc on filtre le df selon stat demandé
    #cherche la ligne corresp à larticle sélectionné(prem lig trouvée)
    row_art = df[df["identifiant_article"] == article_select].iloc[0]
    #demander à lia analyse le risque de rup
    bloc_ia(
        prompt_analyse_rupture(article_select, row_art.to_dict()),
        classe="ai-panel",
        key=f"rupture_{article_select}",
        force_refresh=refresh_ai,
    )

    col1, col2 = st.columns(2)

    with col1:
        section_title("Stock actuel par article", "stock", "#0057B8", "section-header-icon-blue")
        colors_bar = df_filtre["statut"].map({
            "CRITIQUE":  CEGID_PLOTLY["rouge"],
            "ATTENTION": CEGID_PLOTLY["orange"],
            "OK":        CEGID_PLOTLY["vert"],
        })
        fig_stock = go.Figure(go.Bar(
            x=df_filtre["identifiant_article"],
            y=df_filtre["stock_disponible"],
            marker_color=colors_bar,
            text=df_filtre["stock_disponible"],
            textposition="outside",
            marker_line_width=0,
        ))
        #plotly_layout: reutilise style glob défini, height : hauteur du graph
        #incline les labels de laxe x
        fig_stock.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_tickangle=-35)
        fig_stock.update_yaxes(gridcolor="#E2E6EF")
        #affiche graph dans dashboard, use_container_width: prend tte larg dispo
        st.plotly_chart(fig_stock, use_container_width=True)

    with col2:
        section_title("Couverture en semaines", "kpi", "#0A8A5C", "section-header-icon-green")
        df_couv = df_filtre[df_filtre["couverture_sem"] < 50].copy()
        colors_c = df_couv["couverture_sem"].apply(
            lambda x: CEGID_PLOTLY["rouge"] if x < 2 else (CEGID_PLOTLY["orange"] if x < 4 else CEGID_PLOTLY["vert"])
        )
        fig_couv = go.Figure(go.Bar(
            x=df_couv["identifiant_article"],
            y=df_couv["couverture_sem"],
            marker_color=colors_c,
            text=[f"{v:.1f}s" for v in df_couv["couverture_sem"]],
            textposition="outside",
            marker_line_width=0,
        ))
        fig_couv.add_hline(y=2, line_dash="dash", line_color=CEGID_PLOTLY["rouge"],
                           annotation_text="< 2 sem critique", annotation_font_color=CEGID_PLOTLY["rouge"])
        fig_couv.add_hline(y=4, line_dash="dash", line_color=CEGID_PLOTLY["orange"],
                           annotation_text="< 4 sem attention", annotation_font_color=CEGID_PLOTLY["orange"])
        fig_couv.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_tickangle=-35)
        fig_couv.update_yaxes(gridcolor="#E2E6EF")
        st.plotly_chart(fig_couv, use_container_width=True)

    section_title(f"Évolution probabilité rupture — {article_select}", "anomaly", "#E8332A", "section-header-icon-blue")
    rup_art = pred_rup[pred_rup["identifiant_article"] == article_select].sort_values("date_debut_semaine")
    #creer un graphe vide
    fig_evo = go.Figure()
    fig_evo.add_trace(go.Scatter(
        x=rup_art["date_debut_semaine"], y=rup_art["proba_rupture"],
        mode="lines+markers", name="P(rupture)",
        line=dict(color=CEGID_PLOTLY["bleu"], width=2),
        fill="tozeroy", fillcolor="rgba(0,87,184,0.08)",
    ))
    fig_evo.add_trace(go.Scatter(
        x=rup_art["date_debut_semaine"], y=rup_art["rupture_reelle"],
        mode="markers", name="Rupture réelle",
        marker=dict(color=CEGID_PLOTLY["rouge"], size=8, symbol="x"),
    ))
    fig_evo.add_hline(y=0.90, line_dash="dash", line_color=CEGID_PLOTLY["rouge"],
                      annotation_text="Seuil critique 90%", annotation_font_color=CEGID_PLOTLY["rouge"])
    fig_evo.add_hline(y=0.55, line_dash="dash", line_color=CEGID_PLOTLY["orange"],
                      annotation_text="Seuil attention 55%", annotation_font_color=CEGID_PLOTLY["orange"])
    fig_evo.update_layout(**PLOTLY_LAYOUT, height=300)
    fig_evo.update_yaxes(gridcolor="#E2E6EF")
    st.plotly_chart(fig_evo, use_container_width=True)

    section_title(f"Prévision demande vs Réel — {article_select}", "kpi", "#0A8A5C", "section-header-icon-green")
    dem_art = pred_dem[pred_dem["identifiant_article"] == article_select].sort_values("date_debut_semaine")
    fig_prev = go.Figure()
    fig_prev.add_trace(go.Scatter(
        x=dem_art["date_debut_semaine"], y=dem_art["reel"],
        mode="lines+markers", name="Réel",
        line=dict(color=CEGID_PLOTLY["vert"], width=2),
    ))
    fig_prev.add_trace(go.Scatter(
        x=dem_art["date_debut_semaine"], y=dem_art["prediction"],
        mode="lines+markers", name="Prédiction ML",
        line=dict(color=CEGID_PLOTLY["bleu"], width=2, dash="dash"),
    ))
    fig_prev.update_layout(**PLOTLY_LAYOUT, height=300)
    fig_prev.update_yaxes(gridcolor="#E2E6EF")
    st.plotly_chart(fig_prev, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALIES STOCK
# ════════════════════════════════════════════════════════════

elif page == "⚠️ Anomalies stock":
    afficher_topbar("Anomalies stock", OLLAMA_OK)
    st.caption("Surstock (couverture > 8 semaines) · Mauvaise anticipation (commande fourn. > 2× demande réelle)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Anomalies confirmées", len(df[df["statut_anomalie"] == "SURSTOCK/ANTICIPATION"]),
                  delta="Surstock/Mauvaise antici.", delta_color="inverse")
    with c2:
        st.metric("À surveiller", len(df[df["statut_anomalie"] == "À SURVEILLER"]), delta_color="off")
    with c3:
        val_surstock = df[df["statut_anomalie"] != "NORMAL"]["valeur_stock"].sum()
        st.metric("Capital immobilisé", f"{val_surstock:,.0f} €", delta="Anomalies stock")
    with c4:
        st.metric("Taux détection", f"{rec_ano:.0%}", delta="Rappel modèle", delta_color="normal")

    st.divider()

    bloc_ia(
        prompt_analyse_anomalie(df, pred_ano),
        classe="ai-panel ai-panel-anomalie",
        key="anomalie_global",
        force_refresh=refresh_ai,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        section_title("Probabilité anomalie par article", "anomaly", "#5B3EBC", "section-header-icon-purple")
        colors_ano = df["statut_anomalie"].map({
            "SURSTOCK/ANTICIPATION": CEGID_PLOTLY["violet"],
            "À SURVEILLER":          CEGID_PLOTLY["orange"],
            "NORMAL":                CEGID_PLOTLY["vert"],
        })
        fig_ano_bar = go.Figure(go.Bar(
            x=df["proba_anomalie_max"] * 100,
            y=df["identifiant_article"],
            orientation="h",#les barres horizontales
            marker_color=colors_ano,
            text=[f"{v:.0f}%" for v in df["proba_anomalie_max"] * 100],
            textposition="outside",
            marker_line_width=0,
        ))
        fig_ano_bar.add_vline(x=85, line_dash="dash", line_color=CEGID_PLOTLY["violet"],
                              annotation_text="Seuil 85%", annotation_font_color=CEGID_PLOTLY["violet"])
        fig_ano_bar.add_vline(x=55, line_dash="dash", line_color=CEGID_PLOTLY["orange"],
                              annotation_text="Attention 55%", annotation_font_color=CEGID_PLOTLY["orange"])
        fig_ano_bar.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Probabilité anomalie (%)")
        fig_ano_bar.update_xaxes(gridcolor="#E2E6EF")
        st.plotly_chart(fig_ano_bar, use_container_width=True)

    with col2:
        section_title("Détection anomalies par article", "ai", "#5B3EBC", "section-header-icon-purple")
        if "anomalies_reelles" in perf_ano_art.columns:
            #verifie si la col existe dans df(pref...) snn après groupby col peut devenir index
            #donc df ne contient pas col on transforme lindex
            perf_ano_d = perf_ano_art if "identifiant_article" in perf_ano_art.columns else perf_ano_art.reset_index()
            fig_ano_det = go.Figure()
            fig_ano_det.add_trace(go.Bar(
                name="Anomalies réelles",
                x=perf_ano_d["identifiant_article"], y=perf_ano_d["anomalies_reelles"],
                marker_color=CEGID_PLOTLY["violet"], marker_line_width=0,
            ))
            fig_ano_det.add_trace(go.Bar(
                name="Anomalies détectées",
                x=perf_ano_d["identifiant_article"], y=perf_ano_d["anomalies_detectees"],
                marker_color=CEGID_PLOTLY["vert"], opacity=0.80, marker_line_width=0,
            ))
            fig_ano_det.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=420, xaxis_tickangle=-35)
            fig_ano_det.update_yaxes(gridcolor="#E2E6EF")
            st.plotly_chart(fig_ano_det, use_container_width=True)

    article_ano = st.selectbox("Évolution anomalie — article", df["identifiant_article"].tolist(), key="ano_sel")
    ano_art = pred_ano[pred_ano["identifiant_article"] == article_ano].sort_values("date_debut_semaine")

    fig_ano_evo = go.Figure()
    fig_ano_evo.add_trace(go.Scatter(
        x=ano_art["date_debut_semaine"], y=ano_art["proba_anomalie"],
        mode="lines+markers", name="P(anomalie)",
        line=dict(color=CEGID_PLOTLY["violet"], width=2),
        fill="tozeroy", fillcolor="rgba(91,62,188,0.07)",
    ))
    fig_ano_evo.add_trace(go.Scatter(
        x=ano_art["date_debut_semaine"], y=ano_art["anomalie_reelle"],
        mode="markers", name="Anomalie réelle",
        marker=dict(color=CEGID_PLOTLY["orange"], size=8, symbol="diamond"),
    ))
    if "surstock_reel" in ano_art.columns:
        fig_ano_evo.add_trace(go.Scatter(
            x=ano_art["date_debut_semaine"], y=ano_art["surstock_reel"],
            mode="markers", name="Surstock réel",
            marker=dict(color=CEGID_PLOTLY["rouge"], size=7, symbol="square"),
        ))
    if "mauvaise_anticipation_reel" in ano_art.columns:
        fig_ano_evo.add_trace(go.Scatter(
            x=ano_art["date_debut_semaine"], y=ano_art["mauvaise_anticipation_reel"],
            mode="markers", name="Mauvaise anticipation",
            marker=dict(color=CEGID_PLOTLY["bleu2"], size=7, symbol="triangle-up"),
        ))
    fig_ano_evo.add_hline(y=0.85, line_dash="dash", line_color=CEGID_PLOTLY["violet"],
                          annotation_text="Seuil 85%", annotation_font_color=CEGID_PLOTLY["violet"])
    fig_ano_evo.update_layout(**PLOTLY_LAYOUT, height=300)
    fig_ano_evo.update_yaxes(gridcolor="#E2E6EF")
    st.plotly_chart(fig_ano_evo, use_container_width=True)

    section_title("Récapitulatif anomalies", "article", "#5B3EBC", "section-header-icon-purple")
    df_ano_table = df[[
        "identifiant_article", "statut_anomalie", "proba_anomalie_max",
        "couverture_sem", "stock_disponible", "valeur_stock",
    ]].copy()
    df_ano_table.columns = ["Article", "Statut anomalie", "P(anomalie)", "Couv. (sem)", "Stock", "Valeur stock (€)"]
    df_ano_table["P(anomalie)"]     = df_ano_table["P(anomalie)"].apply(lambda x: f"{x:.0%}")
    df_ano_table["Valeur stock (€)"] = df_ano_table["Valeur stock (€)"].apply(lambda x: f"{x:,.0f}")

    def color_ano(val):
        if val == "SURSTOCK/ANTICIPATION": return "background-color:#EEE9FB;color:#5B3EBC;font-weight:700"
        if val == "À SURVEILLER":          return "background-color:#FEF3E5;color:#E87D0D;font-weight:700"
        return "background-color:#E6F5F0;color:#0A8A5C;font-weight:700"

    st.dataframe(
        df_ano_table.style.map(color_ano, subset=["Statut anomalie"]),
        use_container_width=True, height=350,
    )


# ════════════════════════════════════════════════════════════
#  PAGE 4 — RECOMMANDATIONS
# ════════════════════════════════════════════════════════════

elif page == "🎯 Recommandations":
    afficher_topbar("Recommandations de commande", OLLAMA_OK)
    st.caption("Calcul ROP basé prévision ML · SS = 1.65 × MAE_article × √LT · ROP = pred_S1 × LT + SS")

    def calculer_recommandation(row):
        #recup délai moy d'appro: max(.,1): évite un délai nul et div inutile
        lt = max(row["lead_time_hist_moy"], 1)

        # si prév existe on lutilise snn on revient à la moy historique
        pred = row["prediction_s1"] if pd.notna(row["prediction_s1"]) and row["prediction_s1"] > 0 else row["demande_moy"]

        # Incertitude : MAE article > demande_std > 30% prévision ML
        std = row.get("mae_article", None)
        #si mae:std null ou 0 on prend demande_std snn approximation 30% de la prév
        if pd.isna(std) or std == 0:
            std = row.get("demande_std", pred * 0.3)
        if pd.isna(std) or std == 0:
            std = pred * 0.3

        # niv de service * incertitude de la dem * impact dèlai fourn
        ss = round(1.65 * std * np.sqrt(lt))

        # Point de réappro = prévision ML × lead time + stock de sécurité
        rop = round(pred * lt + ss)

        if row["stock_disponible"] < rop:
            # Qté = couvre lead time + 1sem(période de sécu supp) + SS - stock dispo
            qte = round(pred * (lt + 1) + ss - row["stock_disponible"])
            #évite les val négatives si pas besoin de commander
            return max(qte, 0)
        return 0

    df["qte_commander"]  = df.apply(calculer_recommandation, axis=1)
    df["valeur_commande"] = df["qte_commander"] * df["cout_achat"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Articles à commander", len(df[df["qte_commander"] > 0]))
    with c2:
        st.metric("Budget total", f"{df['valeur_commande'].sum():,.0f} €")
    with c3:
        #indic de risque financier
#garder que les art dont e stat est critique(donc fort risque de rup)
#estimation de la perte(predic*prix_vente)
        risque = (df[df["statut"] == "CRITIQUE"]["prediction_s1"] * df[df["statut"] == "CRITIQUE"]["prix_vente"]).sum()
        st.metric("Risque rupture / sem", f"{risque:,.0f} €/sem")
    with c4:
        #qté_commander=0 aucun besoin de commander(stock déjà suffisant)
        #et art classé comme surstock ou mauvaise anticipation
        n_surstock = len(df[(df["qte_commander"] == 0) & (df["statut_anomalie"] == "SURSTOCK/ANTICIPATION")])
        st.metric("Articles en surstock", n_surstock, delta="Pas de commande")

    st.divider()

    bloc_ia(
        prompt_reco_commandes(df),
        classe="ai-panel ai-panel-reco",
        key="reco_global",
        force_refresh=refresh_ai,
        max_tokens=1800,
    )

    st.divider()

    section_title("Commandes à passer", "reco", "#E87D0D", "section-header-icon-orange")
    df_reco = df[df["qte_commander"] > 0][[
        "identifiant_article", "statut", "statut_anomalie",
        "stock_disponible", "couverture_sem", "prediction_s1",
        "proba_rupture_max", "qte_commander", "valeur_commande",
    ]].copy()
    df_reco.columns = [
        "Article", "Statut", "Anomalie", "Stock", "Couv.(sem)",
        "Prévision ML S+1", "P(rupture)", "Qté commander", "Valeur (€)",
    ]
    df_reco["P(rupture)"]      = df_reco["P(rupture)"].apply(lambda x: f"{x:.0%}")
    df_reco["Valeur (€)"]      = df_reco["Valeur (€)"].apply(lambda x: f"{x:,.0f}")
    df_reco["Prévision ML S+1"] = df_reco["Prévision ML S+1"].apply(lambda x: f"{x:.0f}")

    def color_reco(val):
        if val == "CRITIQUE":              return "background-color:#FDE8E7;color:#E8332A;font-weight:700"
        if val == "ATTENTION":             return "background-color:#FEF3E5;color:#E87D0D;font-weight:700"
        if val == "SURSTOCK/ANTICIPATION": return "background-color:#EEE9FB;color:#5B3EBC"
        return ""

    st.dataframe(
        df_reco.style.map(color_reco, subset=["Statut", "Anomalie"]),
        use_container_width=True, height=380,
    )

    section_title("Budget par article", "kpi", "#E87D0D", "section-header-icon-orange")
    df_budget = df[df["qte_commander"] > 0].sort_values("valeur_commande", ascending=False)
    fig_budget = go.Figure(go.Bar(
        x=df_budget["identifiant_article"],
        y=df_budget["valeur_commande"],
        marker_color=df_budget["statut"].map({
            "CRITIQUE":  CEGID_PLOTLY["rouge"],
            "ATTENTION": CEGID_PLOTLY["orange"],
            "OK":        CEGID_PLOTLY["vert"],
        }),
        text=[f"{v:,.0f}€" for v in df_budget["valeur_commande"]],
        textposition="outside",
        marker_line_width=0,
    ))
    fig_budget.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_tickangle=-35)
    fig_budget.update_yaxes(gridcolor="#E2E6EF", title_text="Valeur (€)")
    st.plotly_chart(fig_budget, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 5 — Indicateurs clés de performance
# ════════════════════════════════════════════════════════════

elif page == "📈 Indicateurs clés de performance":
    afficher_topbar("Indicateurs clés de performance", OLLAMA_OK)
    st.caption("3 modèles : Régression demande · Classification rupture · Classification anomalie stock")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("MAE globale", f"{mae_best:.0f} u", delta=f"+{gain}% vs baseline", delta_color="normal")
    with c2:
        st.metric("wMAPE globale", f"{mape_best:.1f}%")
    with c3:
        st.metric("Ruptures détectées", f"{total_det}/{total_rup}", delta="Modèle rupture", delta_color="normal")
    with c4:
        n_ano_total = int(perf_ano_art["anomalies_reelles"].sum())
        n_ano_det   = int(perf_ano_art["anomalies_detectees"].sum())
        st.metric("Anomalies détectées", f"{n_ano_det}/{n_ano_total}", delta="Modèle anomalie", delta_color="normal")
    with c5:
        st.metric("Valeur stock", f"{df['valeur_stock'].sum():,.0f} €")

    st.divider()

    bloc_ia(
        prompt_kpi(mae_best, mape_best, gain, total_rup, total_det, f1_ano, rec_ano, df),
        classe="ai-panel ai-panel-kpi",
        key="kpi_global",
        force_refresh=refresh_ai,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        section_title("Comparaison MAE — modèles", "kpi", "#0057B8", "section-header-icon-blue")
        fig_mod = go.Figure(go.Bar(
            x=res_reg[col_modele],
            y=res_reg["MAE"],
            marker_color=[CEGID_PLOTLY["vert"], CEGID_PLOTLY["bleu"], CEGID_PLOTLY["rouge"]],
            text=[f"MAE={v:.0f}" for v in res_reg["MAE"]],
            textposition="outside",
            marker_line_width=0,
        ))
        fig_mod.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_tickangle=-10)
        fig_mod.update_yaxes(gridcolor="#E2E6EF", title_text="MAE (moins = meilleur)")
        st.plotly_chart(fig_mod, use_container_width=True)

        section_title("wMAPE par article", "kpi", "#0057B8", "section-header-icon-blue")
        perf_sort = perf_art.sort_values("wmape_%")
        colors_mape = perf_sort["wmape_%"].apply(
            lambda x: CEGID_PLOTLY["vert"] if x < 30 else (CEGID_PLOTLY["orange"] if x < 60 else CEGID_PLOTLY["rouge"])
        )
        fig_mape = go.Figure(go.Bar(
            x=perf_sort["wmape_%"],
            y=perf_sort["identifiant_article"],
            orientation="h",
            marker_color=colors_mape,
            text=[f"{v:.1f}%" for v in perf_sort["wmape_%"]],
            textposition="outside",
            marker_line_width=0,
        ))
        fig_mape.add_vline(x=perf_art["wmape_%"].mean(), line_dash="dash",
                           line_color=CEGID_PLOTLY["bleu"], annotation_text="Moyenne",
                           annotation_font_color=CEGID_PLOTLY["bleu"])
        fig_mape.update_layout(**PLOTLY_LAYOUT, height=420)
        fig_mape.update_xaxes(gridcolor="#E2E6EF", title_text="wMAPE % (moins = meilleur)")
        st.plotly_chart(fig_mape, use_container_width=True)

    with col2:
        section_title("Détection ruptures par article", "anomaly", "#E8332A", "section-header-icon-blue")
        fig_rup = go.Figure()
        fig_rup.add_trace(go.Bar(
            name="Ruptures réelles",
            x=perf_rup_art["identifiant_article"], y=perf_rup_art["ruptures_reelles"],
            marker_color=CEGID_PLOTLY["rouge"], marker_line_width=0,
        ))
        fig_rup.add_trace(go.Bar(
            name="Ruptures détectées",
            x=perf_rup_art["identifiant_article"], y=perf_rup_art["ruptures_detectees"],
            marker_color=CEGID_PLOTLY["vert"], opacity=0.80, marker_line_width=0,
        ))
        fig_rup.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=300, xaxis_tickangle=-35)
        fig_rup.update_yaxes(gridcolor="#E2E6EF")
        st.plotly_chart(fig_rup, use_container_width=True)

        section_title("Détection anomalies par article", "anomaly", "#5B3EBC", "section-header-icon-purple")
        perf_ano_d = perf_ano_art if "identifiant_article" in perf_ano_art.columns else perf_ano_art.reset_index()
        fig_ano2 = go.Figure()
        fig_ano2.add_trace(go.Bar(
            name="Anomalies réelles",
            x=perf_ano_d["identifiant_article"], y=perf_ano_d["anomalies_reelles"],
            marker_color=CEGID_PLOTLY["violet"], marker_line_width=0,
        ))
        fig_ano2.add_trace(go.Bar(
            name="Anomalies détectées",
            x=perf_ano_d["identifiant_article"], y=perf_ano_d["anomalies_detectees"],
            marker_color=CEGID_PLOTLY["vert"], opacity=0.80, marker_line_width=0,
        ))
        fig_ano2.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=320, xaxis_tickangle=-35)
        fig_ano2.update_yaxes(gridcolor="#E2E6EF")
        st.plotly_chart(fig_ano2, use_container_width=True)

    section_title("MAE vs wMAPE par article", "ai", "#0057B8", "section-header-icon-blue")
    fig_sc = px.scatter(
        perf_art, x="mae_article", y="wmape_%",
        text="identifiant_article", size="mae_article", color="wmape_%",
        color_continuous_scale=[CEGID_PLOTLY["vert"], CEGID_PLOTLY["orange"], CEGID_PLOTLY["rouge"]],
    )
    fig_sc.update_traces(textposition="top center")
    fig_sc.update_layout(**PLOTLY_LAYOUT, height=360)
    fig_sc.update_xaxes(gridcolor="#E2E6EF", title_text="MAE (unités)")
    fig_sc.update_yaxes(gridcolor="#E2E6EF", title_text="wMAPE (%)")
    st.plotly_chart(fig_sc, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 6 — CHATBOT IA
# ════════════════════════════════════════════════════════════

elif page == "💬 Chatbot IA":
    afficher_topbar("Chatbot IA — Assistant stocks", OLLAMA_OK)
    st.caption("Questions libres · Contexte complet : régression + rupture + anomalie")

#si ollama nest pas lancé(ollama_ok: variable indique si ollama actif)
#affiche msg derreur, on arrete lapp
    if not OLLAMA_OK:
        st.error("Azure Foundry inaccessible ! Vérifiez l'endpoint et la clé API.")
        st.stop()

    st.success("Azure Foundry actif — Llama-3.3-70B-Instruct · Contexte v3 chargé (3 modèles ML)")
#création de la mémoire du chat
#si lutili ouvre lapp pour la prem fois donc on crée liste vide pour stocker les conversations
    if "messages" not in st.session_state:
        st.session_state.messages = []
        #donner lidentité et contexte
        st.session_state.contexte_v3 = f"""Tu es un expert en gestion de stocks et data science. Réponds en français, de façon concise et précise.
Tu analyses les résultats de 3 modèles ML LightGBM :
  1. Régression demande (MAE={mae_best:.0f}, wMAPE={mape_best:.1f}%, gain=+{gain:.1f}%)
  2. Classification rupture stock (détecté {total_det}/{total_rup} ruptures)
  3. Classification anomalie stock (surstock + mauvaise anticipation, rappel={rec_ano:.0%})

ARTICLES ET STOCKS :
{df[['identifiant_article','stock_disponible','couverture_sem','demande_moy',
     'prediction_s1','proba_rupture_max','proba_anomalie_max',
     'statut','statut_anomalie','wmape_%','prix_vente','cout_achat','lead_time_hist_moy','valeur_stock']].to_string(index=False)}

PERFORMANCES RUPTURE PAR ARTICLE :
{perf_rup_art[['identifiant_article','ruptures_reelles','ruptures_detectees','taux_detection_%']].to_string(index=False)}

PERFORMANCES ANOMALIE PAR ARTICLE :
{perf_ano_art[['identifiant_article','anomalies_reelles','anomalies_detectees']].to_string(index=False)}

COMPARAISON MODÈLES ML :
{res_reg.to_string(index=False)}

Réponds uniquement avec ces données, sans inventer."""

    ##########################################################
    # bloc d'affichage d'histo dune conversation puis saisie dune nvl qst via un champ chat
    ##########################################################
#boucle parcourt chaque msg psq: st.session_state.msg contient liste de dic comme ca
#{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Comment puis-je vous aider ?"}
    for msg in st.session_state.messages:
        #si le msg vient de lutili on affiche
        # 〉 Vous : question..... ?
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">〉 <b>Vous :</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    #msg vient de lagent: ✦ Agent : La rupture est due à une hausse soudaine de la demande.
        else:
            st.markdown(
                f'<div class="chat-agent">✦ <b>Agent IA :</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
#crée une zone de chat en bas de la page
    question = st.chat_input("Posez votre question sur les stocks, les anomalies ou les modèles ML…")
    ####################################
#bloc gère lenvoi de la qst à lagent
####################################
#verifie si qst a été pré-remplie dans le session_state
    if "q_auto" in st.session_state:
        question = st.session_state.q_auto
#supp cette var pour éviter que la qst soit reposée à chaque exécution du script
        del st.session_state.q_auto
#si qst existe
    if question:
    #sauvegarde du msg utilisateur pour conserver l'historique
        st.session_state.messages.append({"role": "user", "content": question})
        #affiche immédiat du msg avant que lia répond
        st.markdown(
            f'<div class="chat-user">〉 <b>Vous :</b> {question}</div>',
            unsafe_allow_html=True,
        )
        #pendant que lappel au llm sexécute , streamlit affiche agent réfléchit
        with st.spinner("L'agent réfléchit…"):
            reponse_det, handled = repondre_depuis_donnees(question, df, pred_ano)
            if handled:
                reponse = reponse_det
            else:
                reponse = appeler_ollama(
                    prompt_chatbot(st.session_state.contexte_v3, question),max_tokens=800,)
        #sauvegarde de la réponse
        st.session_state.messages.append({"role": "assistant", "content": reponse})
        #affichage de la rép
        st.markdown(
            f'<div class="chat-agent">✦ <b>Agent IA :</b> {reponse}</div>',
            unsafe_allow_html=True,
        )

    if st.button("✕  Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()