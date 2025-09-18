# travel_wallet_streamlit_app.py — version Postgres (Supabase)
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
from sqlalchemy import create_engine, text

# ----------------------
# Configuration
# ----------------------
APP_TITLE = "Travel Wallet — Per-day / Per-category (Postgres)"
JST = pytz.timezone("Asia/Tokyo")

DB_URL = os.getenv("DATABASE_URL", st.secrets.get("DATABASE_URL", ""))
if not DB_URL:
    st.error("❌ DATABASE_URL manquant dans secrets Streamlit ou variable d'environnement")
engine = create_engine(DB_URL, future=True)

# ----------------------
# Helpers (dates / DB)
# ----------------------
def jst_now() -> datetime:
    return datetime.now(JST)

def jst_today() -> date:
    return jst_now().date()

def to_date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def from_date_str(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def run_query(query: str, params: dict = None, fetch: bool = False):
    with engine.begin() as conn:
        result = conn.execute(text(query), params or {})
        if fetch:
            return result.fetchall()
        return None

def load_dataframe(query: str, params: dict = None):
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)

def init_db():
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS voyages (
            id SERIAL PRIMARY KEY,
            nom TEXT NOT NULL,
            date_debut DATE NOT NULL,
            date_fin DATE NOT NULL,
            tz TEXT NOT NULL DEFAULT 'JST',
            etat TEXT NOT NULL DEFAULT 'actif',
            budget_global_initial INTEGER NOT NULL,
            last_consolidation DATE
        );""",
        """
        CREATE TABLE IF NOT EXISTS personnes (
            id SERIAL PRIMARY KEY,
            voyage_id INT NOT NULL REFERENCES voyages(id) ON DELETE CASCADE,
            nom TEXT NOT NULL
        );""",
        """
        CREATE TABLE IF NOT EXISTS villes (
            id SERIAL PRIMARY KEY,
            voyage_id INT NOT NULL REFERENCES voyages(id) ON DELETE CASCADE,
            nom TEXT NOT NULL,
            UNIQUE(voyage_id, nom)
        );""",
        """
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            voyage_id INT NOT NULL REFERENCES voyages(id) ON DELETE CASCADE,
            nom TEXT NOT NULL,
            couleur TEXT,
            icone TEXT,
            budget_initial INT NOT NULL,
            est_autres BOOLEAN NOT NULL DEFAULT false,
            UNIQUE(voyage_id, nom)
        );""",
        """
        CREATE TABLE IF NOT EXISTS perdiem (
            id SERIAL PRIMARY KEY,
            voyage_id INT NOT NULL REFERENCES voyages(id) ON DELETE CASCADE,
            date DATE NOT NULL,
            categorie_id INT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            montant INT NOT NULL,
            consolidee BOOLEAN NOT NULL DEFAULT false,
            UNIQUE(voyage_id, date, categorie_id)
        );""",
        """
        CREATE TABLE IF NOT EXISTS depenses (
            id SERIAL PRIMARY KEY,
            voyage_id INT NOT NULL REFERENCES voyages(id) ON DELETE CASCADE,
            date DATE NOT NULL,
            montant INT NOT NULL,
            categorie_id INT NOT NULL REFERENCES categories(id),
            ville_id INT REFERENCES villes(id),
            personne_id INT REFERENCES personnes(id),
            libelle TEXT
        );"""
    ]
    for s in stmts:
        run_query(s)

# ----------------------
# Domain logic
# ----------------------
def daterange(d0: date, d1: date):
    for n in range((d1 - d0).days + 1):
        yield d0 + timedelta(days=n)

def create_voyage(nom: str, d0: date, d1: date, budget_global: int, personnes: list[str], categories_init: list[dict]) -> int:
    res = run_query(
        """INSERT INTO voyages(nom, date_debut, date_fin, tz, etat, budget_global_initial, last_consolidation)
            VALUES(:nom,:d0,:d1,'JST','actif',:budget,NULL) RETURNING id""",
        {"nom": nom, "d0": d0, "d1": d1, "budget": budget_global},
        fetch=True,
    )
    voyage_id = res[0][0]
    for p in personnes:
        if p.strip():
            run_query("INSERT INTO personnes(voyage_id, nom) VALUES(:vid,:nom)", {"vid": voyage_id, "nom": p.strip()})
    sum_cat = 0
    for c in categories_init:
        nom_cat = c.get("nom").strip()
        b = int(c.get("budget", 0))
        run_query("INSERT INTO categories(voyage_id, nom, budget_initial, est_autres) VALUES(:vid,:nom,:b,false)",
                  {"vid": voyage_id, "nom": nom_cat, "b": b})
        sum_cat += b
    reste = int(budget_global) - sum_cat
    if reste > 0:
        run_query("INSERT INTO categories(voyage_id, nom, budget_initial, est_autres) VALUES(:vid,'Autres',:b,true)", {"vid": voyage_id, "b": reste})
    N = (d1 - d0).days + 1
    cats = load_dataframe("SELECT id, budget_initial FROM categories WHERE voyage_id=:vid", {"vid": voyage_id})
    for _, cat in cats.iterrows():
        cat_id, Bc = cat["id"], int(cat["budget_initial"])
        base, rem = Bc // N, Bc % N
        for i, day in enumerate(daterange(d0, d1)):
            amt = base + (1 if i < rem else 0)
            run_query("INSERT INTO perdiem(voyage_id, date, categorie_id, montant, consolidee) VALUES(:vid,:d,:c,:m,false)",
                      {"vid": voyage_id, "d": day, "c": cat_id, "m": amt})
    return voyage_id

def get_voyages(active_only=True):
    q = "SELECT * FROM voyages"
    if active_only:
        q += " WHERE etat='actif'"
    q += " ORDER BY id DESC"
    return load_dataframe(q)

def get_entities(voyage_id: int):
    people = load_dataframe("SELECT id, nom FROM personnes WHERE voyage_id=:vid", {"vid": voyage_id})
    cities = load_dataframe("SELECT id, nom FROM villes WHERE voyage_id=:vid", {"vid": voyage_id})
    cats = load_dataframe("SELECT id, nom, budget_initial, est_autres FROM categories WHERE voyage_id=:vid", {"vid": voyage_id})
    return people, cities, cats

def add_city_if_needed(voyage_id: int, nom_ville: str) -> Optional[int]:
    if not nom_ville or not nom_ville.strip():
        return None
    rows = run_query("SELECT id FROM villes WHERE voyage_id=:vid AND nom=:nom", {"vid": voyage_id, "nom": nom_ville.strip()}, fetch=True)
    if rows:
        return rows[0][0]
    res = run_query("INSERT INTO villes(voyage_id, nom) VALUES(:vid,:nom) RETURNING id", {"vid": voyage_id, "nom": nom_ville.strip()}, fetch=True)
    return res[0][0]

def add_depense(voyage_id: int, dt: date, montant: int, categorie_id: int, ville_id: Optional[int], personne_id: Optional[int], libelle: Optional[str]):
    run_query("INSERT INTO depenses(voyage_id, date, montant, categorie_id, ville_id, personne_id, libelle) VALUES(:vid,:d,:m,:c,:v,:p,:l)",
              {"vid": voyage_id, "d": dt, "m": montant, "c": categorie_id, "v": ville_id, "p": personne_id, "l": libelle})
    run_query("UPDATE perdiem SET montant = montant - :m WHERE voyage_id=:vid AND date=:d AND categorie_id=:c",
              {"m": montant, "vid": voyage_id, "d": dt, "c": categorie_id})

def edit_depense(depense_id: int, new_dt: date, new_montant: int, new_categorie_id: int, new_ville_id: Optional[int], new_personne_id: Optional[int], new_libelle: Optional[str]):
    old = run_query("SELECT voyage_id, date, montant, categorie_id FROM depenses WHERE id=:id", {"id": depense_id}, fetch=True)
    if not old:
        return
    voyage_id, old_date, old_montant, old_cat = old[0]
    run_query("UPDATE perdiem SET montant = montant + :m WHERE voyage_id=:vid AND date=:d AND categorie_id=:c",
              {"m": old_montant, "vid": voyage_id, "d": old_date, "c": old_cat})
    run_query("UPDATE depenses SET date=:d, montant=:m, categorie_id=:c, ville_id=:v, personne_id=:p, libelle=:l WHERE id=:id",
              {"d": new_dt, "m": new_montant, "c": new_categorie_id, "v": new_ville_id, "p": new_personne_id, "l": new_libelle, "id": depense_id})
    run_query("UPDATE perdiem SET montant = montant - :m WHERE voyage_id=:vid AND date=:d AND categorie_id=:c",
              {"m": new_montant, "vid": voyage_id, "d": new_dt, "c": new_categorie_id})

def delete_depense(depense_id: int):
    row = run_query("SELECT voyage_id, date, montant, categorie_id FROM depenses WHERE id=:id", {"id": depense_id}, fetch=True)
    if not row:
        return
    voyage_id, d, m, c = row[0]
    run_query("UPDATE perdiem SET montant = montant + :m WHERE voyage_id=:vid AND date=:d AND categorie_id=:c",
              {"m": m, "vid": voyage_id, "d": d, "c": c})
    run_query("DELETE FROM depenses WHERE id=:id", {"id": depense_id})

def consolidate_until_today(voyage_id: int):
    today = jst_today()
    rows = run_query("SELECT date, categorie_id, montant, consolidee FROM perdiem WHERE voyage_id=:vid ORDER BY date, categorie_id",
                     {"vid": voyage_id}, fetch=True)
    for date_str, cat_id, montant, consolidee in rows:
        d = date_str
        if d >= today:
            break
        if consolidee:
            continue
        remainder = int(montant)
        next_d = d + timedelta(days=1)
        run_query("UPDATE perdiem SET montant = montant + :r WHERE voyage_id=:vid AND date=:d AND categorie_id=:c",
                  {"r": remainder, "vid": voyage_id, "d": next_d, "c": cat_id})
        run_query("UPDATE perdiem SET consolidee=true WHERE voyage_id=:vid AND date=:d AND categorie_id=:c",
                  {"vid": voyage_id, "d": d, "c": cat_id})

def compute_kpis(voyage_id: int):
    per = load_dataframe("SELECT date, montant FROM perdiem WHERE voyage_id=:vid", {"vid": voyage_id})
    dep = load_dataframe("SELECT date, montant FROM depenses WHERE voyage_id=:vid", {"vid": voyage_id})
    total_depenses = int(dep["montant"].sum()) if len(dep) else 0
    budget_init = load_dataframe("SELECT budget_global_initial FROM voyages WHERE id=:vid", {"vid": voyage_id}).iloc[0,0]
    today = jst_today()
    D_today = int(per[per["date"] == pd.to_datetime(today)]["montant"].sum()) if len(per) else 0
    planned_until_today = int(per[per["date"] <= pd.to_datetime(today)]["montant"].sum()) if len(per) else 0
    real_until_today = int(dep[dep["date"] <= pd.to_datetime(today)]["montant"].sum()) if len(dep) else 0
    avance = planned_until_today - real_until_today
    return {"budget_courant": budget_init - total_depenses, "perdiem_du_jour": D_today, "avance_retard": avance, "depenses_totales": total_depenses}

def category_totals(voyage_id: int):
    q = """
    SELECT c.id as categorie_id, c.nom as categorie, c.budget_initial,
           COALESCE(SUM(d.montant),0) as depense
    FROM categories c
    LEFT JOIN depenses d ON d.categorie_id = c.id AND d.voyage_id=c.voyage_id
    WHERE c.voyage_id=:vid
    GROUP BY c.id, c.nom, c.budget_initial
    ORDER BY c.nom
    """
    return load_dataframe(q, {"vid": voyage_id})

def export_json(voyage_id: int) -> str:
    v = run_query("SELECT * FROM voyages WHERE id=:vid", {"vid": voyage_id}, fetch=True)[0]
    people = load_dataframe("SELECT id, nom FROM personnes WHERE voyage_id=:vid", {"vid": voyage_id})
    cities = load_dataframe("SELECT id, nom FROM villes WHERE voyage_id=:vid", {"vid": voyage_id})
    cats = load_dataframe("SELECT id, nom, budget_initial, est_autres FROM categories WHERE voyage_id=:vid", {"vid": voyage_id})
    deps = load_dataframe("SELECT id, date, montant, categorie_id, ville_id, personne_id, libelle FROM depenses WHERE voyage_id=:vid ORDER BY date, id", {"vid": voyage_id})
    per = load_dataframe("SELECT date, categorie_id, montant FROM perdiem WHERE voyage_id=:vid ORDER BY date, categorie_id", {"vid": voyage_id})
    payload = {"voyage": dict(v), "personnes": people.to_dict(orient="records"), "villes": cities.to_dict(orient="records"), "categories": cats.to_dict(orient="records"), "depenses": deps.to_dict(orient="records"), "per_diem": per.to_dict(orient="records")}
    path = f"export_voyage_{voyage_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

# ----------------------
# UI et main identiques (à réutiliser)
# ----------------------


# ----------------------
# UI
# ----------------------
def verify_password(pw: str) -> bool:
    if not pw:
        return False
    hash_from_secrets = None
    try:
        hash_from_secrets = st.secrets.get("APP_PASSWORD_HASH")
    except Exception:
        pass
    if not hash_from_secrets:
        hash_from_secrets = os.getenv("APP_PASSWORD_HASH")
    if not hash_from_secrets:
        # no hash configured -> allow (dev)
        return True
    try:
        return bcrypt.checkpw(pw.encode(), hash_from_secrets.encode())
    except Exception:
        return False

def ui_login():
    st.title(APP_TITLE)
    st.caption("Accès protégé (optionnel) — mot de passe via secrets ou variable d'env")
    pw = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if verify_password(pw):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            time.sleep(1)
            st.error("Mot de passe incorrect")

def ui_create_voyage():
    st.subheader("Créer un voyage")
    with st.form("create_voyage_form"):
        nom = st.text_input("Nom du voyage", placeholder="Japon 2025")
        col1, col2 = st.columns(2)
        with col1:
            d0 = st.date_input("Date de début (JST)", value=jst_today())
        with col2:
            default_end = max(jst_today() + timedelta(days=6), d0)
            d1 = st.date_input("Date de fin (JST)", value=default_end, min_value=d0)
        budget_global = st.number_input("Budget global initial (¥)", min_value=1, step=1)
        st.markdown("**Personnes (liste fermée, optionnel)**")
        personnes_raw = st.text_input("Noms séparés par des virgules", placeholder="Alice, Bob")
        st.markdown("**Catégories (format: Nom:Budget, séparées par des virgules)**")
        cats_raw = st.text_input("Ex.: Bouffe:300000, Logement:600000, Transport:200000")
        submitted = st.form_submit_button("Créer")
        if submitted:
            try:
                personnes = [x.strip() for x in personnes_raw.split(",") if x.strip()]
                cats = []
                total_cats = 0
                for chunk in [x for x in cats_raw.split(",") if x.strip()]:
                    if ":" not in chunk:
                        st.error(f"Catégorie invalide: '{chunk}' (format Nom:Budget)")
                        st.stop()
                    n, b = chunk.split(":", 1)
                    b = int(b)
                    cats.append({"nom": n.strip(), "budget": b})
                    total_cats += b
                if total_cats > budget_global:
                    st.error("Somme des budgets de catégories > budget global")
                    st.stop()
                voyage_id = create_voyage(nom, d0, d1, int(budget_global), personnes, cats)
                st.success(f"Voyage créé (id={voyage_id})")
                st.session_state["voyage_id"] = voyage_id
                st.rerun()
            except Exception as e:
                st.exception(e)

def ui_voyage_header(v: pd.Series):
    st.markdown(f"### {v['nom']}  ·  {v['date_debut']} → {v['date_fin']}  (JST)")
    st.caption("Per-diem = solde journalier **par catégorie**. Les dépenses diminuent immédiatement le solde; à minuit JST le reste est reporté au lendemain (même catégorie).")

def ui_kpis(voyage_id: int):
    consolidate_until_today(voyage_id)
    k = compute_kpis(voyage_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget global estimé", f"¥{k['budget_courant']:,}")
    c2.metric("Per-diem total aujourd'hui (toutes catégories)", f"¥{k['perdiem_du_jour']:,}")
    c3.metric("Avance (+) / Retard (−) (cumul)", f"¥{k['avance_retard']:,}")
    c4.metric("Dépenses cumulées", f"¥{k['depenses_totales']:,}")

    # show cumulative curves
    per = load_dataframe("SELECT date, montant FROM perdiem WHERE voyage_id=? ORDER BY date", (voyage_id,))
    if len(per):
        per['date'] = pd.to_datetime(per['date'])
        per_sum = per.groupby('date', as_index=False)['montant'].sum().rename(columns={'montant': 'cumule'})
        per_sum['cumule'] = per_sum['cumule'].cumsum()
    else:
        per_sum = pd.DataFrame({'date': [], 'cumule': []})
    dep = load_dataframe("SELECT date, montant FROM depenses WHERE voyage_id=? ORDER BY date", (voyage_id,))
    if len(dep):
        dep['date'] = pd.to_datetime(dep['date'])
        actual = dep.groupby('date', as_index=False)['montant'].sum().rename(columns={'montant': 'journalier'})
        actual['cumule'] = actual['journalier'].cumsum()
    else:
        actual = pd.DataFrame({'date': [], 'cumule': []})
    ch1 = alt.Chart(per_sum).mark_line().encode(x='date:T', y='cumule:Q', tooltip=['date:T','cumule:Q'])
    ch2 = alt.Chart(actual).mark_line().encode(x='date:T', y='cumule:Q', tooltip=['date:T','cumule:Q'])
    st.altair_chart(alt.layer(ch1, ch2).resolve_scale(y='independent').properties(height=260), use_container_width=True)

    # per-diem today by category
    today = jst_today()
    per_today = load_dataframe("SELECT p.categorie_id, c.nom, p.montant FROM perdiem p JOIN categories c ON c.id=p.categorie_id WHERE p.voyage_id=? AND p.date=? ORDER BY c.nom", (voyage_id, to_date_str(today)))
    if len(per_today):
        st.markdown("**Per-diem du jour par catégorie**")
        st.dataframe(per_today.rename(columns={'categorie_id':'id','nom':'categorie','montant':'montant (¥)'}), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun per-diem pour aujourd'hui (vérifie les dates du voyage).")

def ui_saisie(voyage_id: int):
    st.subheader("Saisie")
    people, cities, cats = get_entities(voyage_id)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dépense**")
        with st.form("dep_form"):
            dt = st.date_input("Date", value=jst_today())
            montant = st.number_input("Montant (¥)", min_value=1, step=1)
            cat = st.selectbox("Catégorie", options=cats['id'], format_func=lambda i: cats.set_index('id').loc[i, 'nom'])
            ville_existing = st.selectbox("Ville (existante)", options=[None] + cities['id'].tolist(), format_func=lambda x: '—' if x is None else cities.set_index('id').loc[x, 'nom'])
            ville_new = st.text_input("… ou nouvelle ville")
            pers = st.selectbox("Personne", options=[None] + people['id'].tolist(), format_func=lambda x: '—' if x is None else people.set_index('id').loc[x, 'nom'])
            lib = st.text_input("Libellé (optionnel)")
            ok = st.form_submit_button("Ajouter")
            if ok:
                try:
                    ville_id = ville_existing
                    if ville_new.strip():
                        ville_id = add_city_if_needed(voyage_id, ville_new)
                    add_depense(voyage_id, dt, int(montant), int(cat), ville_id, pers, lib)
                    st.success("Dépense ajoutée — le solde du jour pour la catégorie a été mis à jour.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with col2:
        st.markdown("**Historique rapide**")
        recent = load_dataframe("SELECT d.id, d.date, d.montant, c.nom as categorie, v.nom as ville, p.nom as personne, d.libelle FROM depenses d JOIN categories c ON c.id=d.categorie_id LEFT JOIN villes v ON v.id=d.ville_id LEFT JOIN personnes p ON p.id=d.personne_id WHERE d.voyage_id=? ORDER BY d.date DESC, d.id DESC LIMIT 20", (voyage_id,))
        st.dataframe(recent, use_container_width=True, hide_index=True)

def ui_listes(voyage_id: int):
    st.subheader("Listes & Filtres")
    dep = load_dataframe("""
        SELECT d.id, d.date, d.montant, c.nom AS categorie, v.nom AS ville, p.nom AS personne, d.libelle
        FROM depenses d
        JOIN categories c ON c.id=d.categorie_id
        LEFT JOIN villes v ON v.id=d.ville_id
        LEFT JOIN personnes p ON p.id=d.personne_id
        WHERE d.voyage_id=?
        ORDER BY d.date DESC, d.id DESC
    """, (voyage_id,))
    st.dataframe(dep, use_container_width=True, hide_index=True)

def ui_categories(voyage_id: int):
    st.subheader("Par catégorie")
    df = category_totals(voyage_id)
    df["dépassement"] = df["depense"] > df["budget_initial"]
    def fmt_badge(row):
        if row["dépassement"]:
            return f"⚠️ Dépassement de ¥{row['depense']-row['budget_initial']:,}"
        return "OK"
    st.dataframe(df.assign(status=df.apply(fmt_badge, axis=1))[["categorie","budget_initial","depense","status"]], use_container_width=True, hide_index=True)
    chart = alt.Chart(df).mark_bar().encode(x=alt.X("categorie:N", sort='-y'), y=alt.Y("depense:Q"), tooltip=["categorie","depense","budget_initial"]).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

def ui_par_vue(voyage_id: int):
    st.subheader("Vues par jour / catégorie")
    # pivot table
    per = load_dataframe("SELECT p.date, c.nom AS categorie, p.montant FROM perdiem p JOIN categories c ON c.id=p.categorie_id WHERE p.voyage_id=? ORDER BY p.date, c.nom", (voyage_id,))
    if per.empty:
        st.info("Aucun per-diem initialisé.")
        return
    per['date'] = pd.to_datetime(per['date']).dt.date
    pivot = per.pivot(index='date', columns='categorie', values='montant').fillna(0).astype(int)
    st.markdown("**Matrice date × catégorie (solde restant)**")
    st.dataframe(pivot, use_container_width=True)
    st.markdown("**Graphique : total par jour (somme des catégories)**")
    tot = pivot.sum(axis=1).reset_index().rename(columns={0:'total'}) if len(pivot) else pd.DataFrame()
    if not tot.empty:
        tot.columns = ['date','total']
        tot['date'] = pd.to_datetime(tot['date'])
        chart = alt.Chart(tot).mark_line().encode(x='date:T', y='total:Q', tooltip=['date:T','total:Q'])
        st.altair_chart(chart, use_container_width=True)

def ui_admin(voyage_id: int):
    st.subheader("Administration")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Archiver (lecture seule)"):
            with get_conn() as conn:
                conn.execute("UPDATE voyages SET etat='archivé' WHERE id=?", (voyage_id,))
            st.success("Voyage archivé.")
            st.rerun()
    with col2:
        path = export_json(voyage_id)
        with open(path, "rb") as f:
            st.download_button("Exporter JSON", data=f, file_name=os.path.basename(path), mime="application/json")
    with col3:
        if st.button("Supprimer le voyage"):
            st.session_state["confirm_delete"] = True
    if st.session_state.get("confirm_delete"):
        st.warning("Tapez le nom exact du voyage pour confirmer la suppression")
        v = load_dataframe("SELECT nom FROM voyages WHERE id=?", (voyage_id,)).iloc[0]
        name = st.text_input("Nom du voyage")
        if st.button("Confirmer suppression"):
            if name == v["nom"]:
                with get_conn() as conn:
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("DELETE FROM voyages WHERE id=?", (voyage_id,))
                st.success("Voyage supprimé.")
                st.session_state.pop("confirm_delete")
                st.rerun()
            else:
                st.error("Nom incorrect.")

# ----------------------
# Main
# ----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")
    init_db()

    # auth
    if not st.session_state.get("auth_ok"):
        ui_login()
        if not st.session_state.get("auth_ok"):
            return

    # sidebar voyages + creation
    with st.sidebar:
        st.header("Voyages")
        voyages = get_voyages(active_only=False)
        if voyages.empty:
            st.info("Aucun voyage — créez-en un.")
        else:
            idx = st.selectbox(
                "Sélectionnez un voyage",
                options=voyages.index,
                format_func=lambda i: f"#{voyages.loc[i,'id']} — {voyages.loc[i,'nom']} ({voyages.loc[i,'etat']})",
            )
            st.session_state["voyage_id"] = int(voyages.loc[idx, "id"])
        st.divider()
        ui_create_voyage()

    voyage_id = st.session_state.get("voyage_id")
    if not voyage_id:
        st.info("Sélectionnez un voyage à gauche.")
        return

    vdf = get_voyages(active_only=False)
    v = vdf[vdf["id"] == voyage_id].iloc[0]

    ui_voyage_header(v)
    ui_kpis(voyage_id)

    tab1, tab2, tab3, tab4 = st.tabs(["Saisie", "Listes", "Catégories", "Analyses"])
    with tab1:
        ui_saisie(voyage_id)
    with tab2:
        ui_listes(voyage_id)
    with tab3:
        ui_categories(voyage_id)
    with tab4:
        ui_par_vue(voyage_id)

    st.divider()
    ui_admin(voyage_id)

if __name__ == "__main__":
    main()
