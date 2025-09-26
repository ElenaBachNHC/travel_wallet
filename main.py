# travel_wallet_streamlit_app_supabase_v2.py
import os
import time
import json
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import pytz
import streamlit as st
import altair as alt
import bcrypt
from supabase import create_client

APP_TITLE = "Travel Wallet — Supabase"
JST = pytz.timezone("Asia/Tokyo")

# ----------------------
# Supabase
# ----------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase URL ou KEY manquante")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------
# Helpers
# ----------------------
def jst_now(): return datetime.now(JST)
def jst_today(): return jst_now().date()
def from_date_str(s): return datetime.strptime(s, "%Y-%m-%d").date() if isinstance(s, str) else s
def daterange(d0, d1): return [d0 + timedelta(days=i) for i in range((d1-d0).days+1)]

def load_dataframe(table, filters=None):
    query = supabase.table(table)
    if filters:
        for k,v in filters.items(): query = query.eq(k,v)
    data = query.execute().data
    return pd.DataFrame(data) if data else pd.DataFrame()

# ----------------------
# Domain logic
# ----------------------
def create_voyage(nom, d0, d1, budget_global, personnes, categories_init):
    data = {"nom":nom,"date_debut":str(d0),"date_fin":str(d1),"tz":"JST","etat":"actif","budget_global_initial":budget_global,"last_consolidation":None}
    res = supabase.table("voyages").insert(data).execute()
    voyage_id = res.data[0]["id"]
    for p in personnes or []:
        if p.strip(): supabase.table("personnes").insert({"voyage_id":voyage_id,"nom":p.strip()}).execute()
    sum_cat=0
    for c in categories_init or []:
        b=int(c["budget"])
        supabase.table("categories").insert({"voyage_id":voyage_id,"nom":c["nom"].strip(),"budget_initial":b,"est_autres":False}).execute()
        sum_cat+=b
    reste = budget_global-sum_cat
    if reste>0: supabase.table("categories").insert({"voyage_id":voyage_id,"nom":"Autres","budget_initial":reste,"est_autres":True}).execute()
    N=(d1-d0).days+1
    cats = supabase.table("categories").select("id,budget_initial").eq("voyage_id",voyage_id).order("id").execute().data
    for cat in cats:
        base, rem = divmod(cat["budget_initial"],N)
        for i, day in enumerate(daterange(d0,d1)):
            amt = base+(1 if i<rem else 0)
            supabase.table("perdiem").insert({"voyage_id":voyage_id,"date":str(day),"categorie_id":cat["id"],"montant":amt,"consolidee":False}).execute()
    return voyage_id

def add_depense(voyage_id, dt, montant, categorie_id, ville_id=None, personne_id=None, libelle=None):
    if montant<=0: raise ValueError("Montant doit être > 0")
    supabase.table("depenses").insert({"voyage_id":voyage_id,"date":str(dt),"montant":montant,"categorie_id":categorie_id,"ville_id":ville_id,"personne_id":personne_id,"libelle":libelle}).execute()
    r = supabase.table("perdiem").update({"montant":f"montant-{montant}"}).eq("voyage_id",voyage_id).eq("date",str(dt)).eq("categorie_id",categorie_id).execute()
    if not r.data:
        supabase.table("perdiem").insert({"voyage_id":voyage_id,"date":str(dt),"categorie_id":categorie_id,"montant":-montant,"consolidee":False}).execute()

def get_voyages(active_only=True):
    df=load_dataframe("voyages")
    if active_only: df=df[df["etat"]=="actif"]
    return df.sort_values("id",ascending=False)

def get_entities(voyage_id):
    people = load_dataframe("personnes",{"voyage_id":voyage_id})
    cities = load_dataframe("villes",{"voyage_id":voyage_id})
    cats = load_dataframe("categories",{"voyage_id":voyage_id}).sort_values(["est_autres","nom"])
    return people, cities, cats

def add_city_if_needed(voyage_id, nom_ville):
    if not nom_ville.strip(): return None
    df = load_dataframe("villes",{"voyage_id":voyage_id,"nom":nom_ville.strip()})
    if not df.empty: return int(df.iloc[0]["id"])
    res = supabase.table("villes").insert({"voyage_id":voyage_id,"nom":nom_ville.strip()}).execute()
    return res.data[0]["id"]

# ----------------------
# Auth
# ----------------------
def verify_password(pw):
    if not pw: return False
    hash_from_secrets = st.secrets.get("APP_PASSWORD_HASH") if hasattr(st,"secrets") else os.getenv("APP_PASSWORD_HASH")
    if not hash_from_secrets: return True
    return bcrypt.checkpw(pw.encode(),hash_from_secrets.encode())

def ui_login():
    st.title(APP_TITLE)
    st.caption("Accès protégé")
    pw = st.text_input("Mot de passe",type="password")
    if st.button("Se connecter"):
        if verify_password(pw):
            st.session_state["auth_ok"]=True
            st.experimental_rerun()
        else:
            time.sleep(1); st.error("Mot de passe incorrect")

# ----------------------
# UI: création, saisie, dashboard
# ----------------------
def ui_create_voyage():
    st.subheader("Créer un voyage")
    with st.form("create_voyage_form"):
        nom = st.text_input("Nom du voyage","Japon 2025")
        d0 = st.date_input("Date début",value=jst_today())
        d1 = st.date_input("Date fin",value=max(jst_today()+timedelta(days=6),d0),min_value=d0)
        budget_global = st.number_input("Budget global",min_value=1,step=1)
        personnes_raw = st.text_input("Personnes (virgule)","Alice,Bob")
        cats_raw = st.text_input("Catégories Nom:Budget","Bouffe:300000,Logement:600000")
        submitted=st.form_submit_button("Créer")
        if submitted:
            try:
                personnes=[x.strip() for x in personnes_raw.split(",") if x.strip()]
                cats=[]; total_cats=0
                for chunk in [x for x in cats_raw.split(",") if x.strip()]:
                    n,b=chunk.split(":",1)
                    b=int(b)
                    cats.append({"nom":n.strip(),"budget":b})
                    total_cats+=b
                if total_cats>budget_global: st.error("Somme des catégories > budget global"); st.stop()
                vid=create_voyage(nom,d0,d1,int(budget_global),personnes,cats)
                st.success(f"Voyage créé id={vid}")
                st.session_state["voyage_id"]=vid; st.experimental_rerun()
            except Exception as e: st.exception(e)

def ui_saisie(voyage_id):
    st.subheader("Saisie dépense")
    people,cities,cats=get_entities(voyage_id)
    with st.form("dep_form"):
        dt=st.date_input("Date",value=jst_today())
        montant=st.number_input("Montant",min_value=1,step=1)
        cat=st.selectbox("Catégorie",options=cats["id"].tolist(),format_func=lambda i: cats.set_index("id").loc[i,"nom"])
        ville_existing=st.selectbox("Ville existante",[None]+cities["id"].tolist(),format_func=lambda x:"—" if x is None else cities.set_index("id").loc[x,"nom"])
        ville_new=st.text_input("Nouvelle ville")
        pers=st.selectbox("Personne",[None]+people["id"].tolist(),format_func=lambda x:"—" if x is None else people.set_index("id").loc[x,"nom"])
        lib=st.text_input("Libellé")
        ok=st.form_submit_button("Ajouter")
        if ok:
            try:
                ville_id=ville_existing
                if ville_new.strip(): ville_id=add_city_if_needed(voyage_id,ville_new)
                add_depense(voyage_id,dt,int(montant),int(cat),ville_id,pers,lib)
                st.success("Dépense ajoutée"); st.experimental_rerun()
            except Exception as e: st.error(str(e))

def ui_dashboard(voyage_id):
    st.subheader("Dashboard")
    per=load_dataframe("perdiem",{"voyage_id":voyage_id})
    dep=load_dataframe("depenses",{"voyage_id":voyage_id})
    if not per.empty: per["date"]=pd.to_datetime(per["date"]); per_sum=per.groupby("date")["montant"].sum().cumsum().reset_index()
    else: per_sum=pd.DataFrame({"date":[],"montant":[]})
    if not dep.empty: dep["date"]=pd.to_datetime(dep["date"]); dep_sum=dep.groupby("date")["montant"].sum().cumsum().reset_index()
    else: dep_sum=pd.DataFrame({"date":[],"montant":[]})
    ch1=alt.Chart(per_sum).mark_line(color="blue").encode(x="date:T",y="montant:Q",tooltip=["date","montant"])
    ch2=alt.Chart(dep_sum).mark_line(color="red").encode(x="date:T",y="montant:Q",tooltip=["date","montant"])
    st.altair_chart(alt.layer(ch1,ch2).resolve_scale(y="independent").properties(height=300),use_container_width=True)

# ----------------------
# Main
# ----------------------
def main():
    st.set_page_config(APP_TITLE,page_icon="🧾",layout="wide")
    if not st.session_state.get("auth_ok"): ui_login(); return
    with st.sidebar:
        st.header("Voyages")
        voyages=get_voyages(active_only=False)
        if voyages.empty: st.info("Aucun voyage"); ui_create_voyage()
        else:
            idx=st.selectbox("Sélection voyage",options=voyages.index,format_func=lambda i:f"#{voyages.loc[i,'id']}—{voyages.loc[i,'nom']} ({voyages.loc[i,'etat']})")
            st.session_state["voyage_id"]=voyages.loc[idx,"id"]
            ui_create_voyage()
    vid=st.session_state.get("voyage_id")
    if not vid: st.info("Sélectionner un voyage"); return
    ui_saisie(vid)
    ui_dashboard(vid)

if __name__=="__main__":
    main()
