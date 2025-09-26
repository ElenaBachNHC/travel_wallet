# travel_wallet_streamlit_app.py
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

# DATABASE_URL doit être dans st.secrets (Streamlit Cloud) ou variable d'env
DATABASE_URL = st.secrets["DATABASE_URL"] if hasattr(st, "secrets") else os.getenv("DATABASE_URL")
if not DATABASE_URL:
    st.error("DATABASE_URL manquant dans st.secrets ou variable d'environnement. Ajoute la chaîne de connexion Postgres (Supabase).")
    st.stop()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

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
    # accepte date/datetime/str
    if s is None:
        return None
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    return datetime.strptime(s, "%Y-%m-%d").date()

def run_query(query: str, params: dict = None, fetch: bool = False):
    """Exécute une requête SQL. Si fetch==True retourne une liste de dicts (mappings)."""
    with engine.begin() as conn:
        result = conn.execute(text(query), params or {})
        if fetch:
            # result.mappings().all() -> list[RowMapping] convertible en dicts
            rows = result.mappings().all()
            return [dict(r) for r in rows]
        return None

def load_dataframe(query: str, params: dict = None) -> pd.DataFrame:
    """Retourne un DataFrame pandas pour une requête SELECT."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})

# ----------------------
# Init DB (Postgres)
# ----------------------
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
            budget_global_initial BIGINT NOT NULL,
            last_consolidation DATE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS personnes (
            id SERIAL PRIMARY KEY,
            voyage_id INT REFERENCES voyages(id) ON DELETE CASCADE,
            nom TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS villes (
            id SERIAL PRIMARY KEY,
            voyage_id INT REFERENCES voyages(id) ON DELETE CASCADE,
            nom TEXT NOT NULL,
            UNIQUE(voyage_id, nom)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            voyage_id INT REFERENCES voyages(id) ON DELETE CASCADE,
            nom TEXT NOT NULL,
            couleur TEXT,
            icone TEXT,
            budget_initial BIGINT NOT NULL,
            est_autres BOOLEAN NOT NULL DEFAULT FALSE,
            UNIQUE(voyage_id, nom)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS perdiem (
            id SERIAL PRIMARY KEY,
            voyage_id INT REFERENCES voyages(id) ON DELETE CASCADE,
            date DATE NOT NULL,
            categorie_id INT REFERENCES categories(id) ON DELETE CASCADE,
            montant BIGINT NOT NULL,
            consolidee BOOLEAN NOT NULL DEFAULT FALSE,
            UNIQUE(voyage_id, date, categorie_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS depenses (
            id SERIAL PRIMARY KEY,
            voyage_id INT REFERENCES voyages(id) ON DELETE CASCADE,
            date DATE NOT NULL,
            montant BIGINT NOT NULL,
            categorie_id INT REFERENCES categories(id),
            ville_id INT REFERENCES villes(id),
            personne_id INT REFERENCES personnes(id),
            libelle TEXT
        );
        """
    ]
    for s in stmts:
        run_query(s)

# ----------------------
# Domain logic
# ----------------------
def daterange(d0: date, d1: date):
    for n in range((d1 - d0).days + 1):
        yield d0 + timedelta(days=n)

def create_voyage(nom: str, d0: date, d1: date, budget_global: int, personnes: list, categories_init: list) -> int:
    """
    Crée le voyage, personnes, catégories et initialise le per-diem par catégorie / jour.
    categories_init = list of dicts { 'nom':..., 'budget': int }
    """
    if d1 < d0:
        raise ValueError("Date de fin avant date de début")
    if budget_global <= 0:
        raise ValueError("Budget global doit être > 0")
    with engine.begin() as conn:
        res = conn.execute(
            text("""INSERT INTO voyages
                   (nom,date_debut,date_fin,tz,etat,budget_global_initial,last_consolidation)
                   VALUES(:nom,:d0,:d1,'JST','actif',:bg,NULL)
                   RETURNING id"""),
            {"nom": nom, "d0": d0, "d1": d1, "bg": int(budget_global)}
        )
        voyage_id = int(res.scalar())

        # personnes
        for p in (personnes or []):
            if p and p.strip():
                conn.execute(text("INSERT INTO personnes(voyage_id, nom) VALUES(:v,:n)"), {"v": voyage_id, "n": p.strip()})

        # categories
        sum_cat = 0
        for c in (categories_init or []):
            nom_cat = c.get("nom").strip()
            b = int(c.get("budget", 0))
            if b < 0:
                raise ValueError("Budget de catégorie négatif")
            conn.execute(text(
                "INSERT INTO categories(voyage_id, nom, couleur, icone, budget_initial, est_autres) VALUES(:v,:n,NULL,NULL,:b,false)"
            ), {"v": voyage_id, "n": nom_cat, "b": b})
            sum_cat += b

        reste = int(budget_global) - sum_cat
        if reste > 0:
            conn.execute(text(
                "INSERT INTO categories(voyage_id, nom, budget_initial, est_autres) VALUES(:v,'Autres',:b,true)"
            ), {"v": voyage_id, "b": reste})

        # initialize perdiem per category per day (integer division + remainder on earliest days)
        N = (d1 - d0).days + 1
        cats = conn.execute(text("SELECT id, budget_initial FROM categories WHERE voyage_id=:v ORDER BY id"), {"v": voyage_id}).mappings().all()
        for cat in cats:
            cat_id = int(cat["id"])
            Bc = int(cat["budget_initial"])
            base, rem = divmod(Bc, N)
            for i, day in enumerate(daterange(d0, d1)):
                amt = base + (1 if i < rem else 0)
                conn.execute(text(
                    "INSERT INTO perdiem(voyage_id, date, categorie_id, montant, consolidee) VALUES(:v,:d,:c,:m,false)"
                ), {"v": voyage_id, "d": day, "c": cat_id, "m": int(amt)})

    return voyage_id

def get_voyages(active_only=True) -> pd.DataFrame:
    q = "SELECT * FROM voyages"
    if active_only:
        q += " WHERE etat='actif'"
    q += " ORDER BY id DESC"
    return load_dataframe(q)

def get_entities(voyage_id: int):
    people = load_dataframe("SELECT id, nom FROM personnes WHERE voyage_id=:v ORDER BY id", {"v": voyage_id})
    cities = load_dataframe("SELECT id, nom FROM villes WHERE voyage_id=:v ORDER BY nom", {"v": voyage_id})
    cats = load_dataframe("SELECT id, nom, budget_initial, est_autres FROM categories WHERE voyage_id=:v ORDER BY est_autres, nom", {"v": voyage_id})
    return people, cities, cats

def add_city_if_needed(voyage_id: int, nom_ville: str) -> Optional[int]:
    if not nom_ville or not nom_ville.strip():
        return None
    rows = run_query("SELECT id FROM villes WHERE voyage_id=:v AND nom=:n", {"v": voyage_id, "n": nom_ville.strip()}, fetch=True)
    if rows:
        return int(rows[0]["id"])
    res = run_query("INSERT INTO villes(voyage_id, nom) VALUES(:v,:n) RETURNING id", {"v": voyage_id, "n": nom_ville.strip()}, fetch=True)
    return int(res[0]["id"])

def add_depense(voyage_id: int, dt: date, montant: int, categorie_id: int, ville_id: Optional[int], personne_id: Optional[int], libelle: Optional[str]):
    """
    Ajoute une dépense et décrémente immédiatement le perdiem du jour+catégorie.
    """
    if montant <= 0:
        raise ValueError("Montant doit être > 0")
    # vérifie fenêtre voyage
    vdf = load_dataframe("SELECT date_debut, date_fin FROM voyages WHERE id=:v", {"v": voyage_id})
    if vdf.empty:
        raise ValueError("Voyage introuvable")
    d0 = from_date_str(vdf.iloc[0]["date_debut"])
    d1 = from_date_str(vdf.iloc[0]["date_fin"])
    today_jst = jst_today()
    if dt < d0 or dt > d1:
        raise ValueError("Date hors période de voyage")
    if dt > today_jst:
        raise ValueError("Saisie future interdite")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO depenses(voyage_id, date, montant, categorie_id, ville_id, personne_id, libelle) VALUES(:v,:d,:m,:c,:vi,:p,:l)"
        ), {"v": voyage_id, "d": dt, "m": int(montant), "c": int(categorie_id), "vi": ville_id, "p": personne_id, "l": libelle})
        r = conn.execute(text(
            "UPDATE perdiem SET montant = montant - :m WHERE voyage_id=:v AND date=:d AND categorie_id=:c"
        ), {"m": int(montant), "v": voyage_id, "d": dt, "c": int(categorie_id)})
        if r.rowcount == 0:
            # crée ligne perdiem négative si absente (failsafe)
            conn.execute(text(
                "INSERT INTO perdiem(voyage_id, date, categorie_id, montant, consolidee) VALUES(:v,:d,:c,:m,false)"
            ), {"v": voyage_id, "d": dt, "c": int(categorie_id), "m": -int(montant)})

def edit_depense(depense_id: int, new_dt: date, new_montant: int, new_categorie_id: int, new_ville_id: Optional[int], new_personne_id: Optional[int], new_libelle: Optional[str]):
    with engine.begin() as conn:
        old = conn.execute(text("SELECT voyage_id, date, montant, categorie_id FROM depenses WHERE id=:id"), {"id": depense_id}).mappings().first()
        if not old:
            raise ValueError("Dépense introuvable")
        voyage_id = int(old["voyage_id"])
        old_date = from_date_str(old["date"])
        old_amount = int(old["montant"])
        old_cat = int(old["categorie_id"])
        # restore old to perdiem
        conn.execute(text("UPDATE perdiem SET montant = montant + :m WHERE voyage_id=:v AND date=:d AND categorie_id=:c"),
                     {"m": old_amount, "v": voyage_id, "d": old_date, "c": old_cat})
        # update depense row
        conn.execute(text("UPDATE depenses SET date=:d, montant=:m, categorie_id=:c, ville_id=:vi, personne_id=:p, libelle=:l WHERE id=:id"),
                     {"d": new_dt, "m": int(new_montant), "c": int(new_categorie_id), "vi": new_ville_id, "p": new_personne_id, "l": new_libelle, "id": depense_id})
        # apply new to perdiem
        r = conn.execute(text("UPDATE perdiem SET montant = montant - :m WHERE voyage_id=:v AND date=:d AND categorie_id=:c"),
                        {"m": int(new_montant), "v": voyage_id, "d": new_dt, "c": int(new_categorie_id)})
        if r.rowcount == 0:
            conn.execute(text("INSERT INTO perdiem(voyage_id, date, categorie_id, montant, consolidee) VALUES(:v,:d,:c,:m,false)"),
                         {"v": voyage_id, "d": new_dt, "c": int(new_categorie_id), "m": -int(new_montant)})

def delete_depense(depense_id: int):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT voyage_id, date, montant, categorie_id FROM depenses WHERE id=:id"), {"id": depense_id}).mappings().first()
        if not row:
            return
        voyage_id = int(row["voyage_id"])
        d = from_date_str(row["date"])
        m = int(row["montant"])
        c = int(row["categorie_id"])
        # restore perdiem
        conn.execute(text("UPDATE perdiem SET montant = montant + :m WHERE voyage_id=:v AND date=:d AND categorie_id=:c"),
                     {"m": m, "v": voyage_id, "d": d, "c": c})
        conn.execute(text("DELETE FROM depenses WHERE id=:id"), {"id": depense_id})

def get_trip_window(voyage_id: int) -> tuple:
    df = load_dataframe("SELECT date_debut, date_fin FROM voyages WHERE id=:v", {"v": voyage_id})
    if df.empty:
        raise ValueError("Voyage introuvable")
    d0 = pd.to_datetime(df.iloc[0]["date_debut"]).date()
    d1 = pd.to_datetime(df.iloc[0]["date_fin"]).date()
    return d0, d1

def consolidate_until_today(voyage_id: int):
    """
    Pour chaque (date < aujourd'hui, catégorie) non consolidé :
      remainder = montant (déjà diminution par dépenses)
      on ajoute remainder au lendemain (même catégorie) et on marque consolidee
      si next day hors trip -> add back to budget_global_initial
    """
    today = jst_today()
    d0, d1 = get_trip_window(voyage_id)
    rows = run_query("SELECT date, categorie_id, montant, consolidee FROM perdiem WHERE voyage_id=:v ORDER BY date, categorie_id", {"v": voyage_id}, fetch=True)
    with engine.begin() as conn:
        for r in rows:
            d = r["date"]
            # r["date"] est déjà datetime.date (psycopg2) ou string; convertir si besoin
            if isinstance(d, str):
                d = from_date_str(d)
            if d >= today:
                break
            if r["consolidee"]:
                continue
            remainder = int(r["montant"])
            next_d = d + timedelta(days=1)
            if next_d <= d1:
                u = conn.execute(text("UPDATE perdiem SET montant = montant + :m WHERE voyage_id=:v AND date=:d AND categorie_id=:c"),
                                 {"m": remainder, "v": voyage_id, "d": next_d, "c": r["categorie_id"]})
                if u.rowcount == 0:
                    conn.execute(text("INSERT INTO perdiem(voyage_id,date,categorie_id,montant,consolidee) VALUES(:v,:d,:c,:m,false)"),
                                 {"v": voyage_id, "d": next_d, "c": r["categorie_id"], "m": remainder})
            else:
                # fin du voyage
                conn.execute(text("UPDATE voyages SET budget_global_initial = budget_global_initial + :m WHERE id=:v"),
                             {"m": remainder, "v": voyage_id})
            conn.execute(text("UPDATE perdiem SET consolidee = TRUE WHERE voyage_id=:v AND date=:d AND categorie_id=:c"),
                         {"v": voyage_id, "d": d, "c": r["categorie_id"]})

# ----------------------
# Metrics / Exports
# ----------------------
def compute_kpis(voyage_id: int):
    per = load_dataframe("SELECT date, montant FROM perdiem WHERE voyage_id=:v ORDER BY date", {"v": voyage_id})
    dep = load_dataframe("SELECT date, montant FROM depenses WHERE voyage_id=:v ORDER BY date", {"v": voyage_id})
    # normalise colonnes date en date
    if not per.empty:
        per["date"] = pd.to_datetime(per["date"]).dt.date
    if not dep.empty:
        dep["date"] = pd.to_datetime(dep["date"]).dt.date
    B0 = int(load_dataframe("SELECT budget_global_initial FROM voyages WHERE id=:v", {"v": voyage_id}).iloc[0, 0])
    total_depenses = int(dep["montant"].sum()) if not dep.empty else 0
    Bc = B0 - total_depenses
    today = jst_today()
    D_today = int(per[per["date"] == today]["montant"].sum()) if not per.empty else 0
    planned_until_today = int(per[per["date"] <= today]["montant"].sum()) if not per.empty else 0
    real_until_today = int(dep[dep["date"] <= today]["montant"].sum()) if not dep.empty else 0
    avance = planned_until_today - real_until_today
    return {
        "budget_courant": Bc,
        "perdiem_du_jour": D_today,
        "avance_retard": avance,
        "depenses_totales": total_depenses,
        "planned_until_today": planned_until_today,
        "real_until_today": real_until_today,
    }

def category_totals(voyage_id: int) -> pd.DataFrame:
    q = """
    SELECT c.id as categorie_id, c.nom as categorie, c.budget_initial,
           COALESCE(SUM(d.montant),0) as depense
    FROM categories c
    LEFT JOIN depenses d ON d.categorie_id = c.id AND d.voyage_id = c.voyage_id
    WHERE c.voyage_id = :v
    GROUP BY c.id, c.nom, c.budget_initial
    ORDER BY c.nom
    """
    return load_dataframe(q, {"v": voyage_id})

def export_json(voyage_id: int) -> str:
    voyage = run_query("SELECT * FROM voyages WHERE id=:v", {"v": voyage_id}, fetch=True)[0]
    people = load_dataframe("SELECT id, nom FROM personnes WHERE voyage_id=:v", {"v": voyage_id})
    cities = load_dataframe("SELECT id, nom FROM villes WHERE voyage_id=:v", {"v": voyage_id})
    cats = load_dataframe("SELECT id, nom, couleur, icone, budget_initial, est_autres FROM categories WHERE voyage_id=:v", {"v": voyage_id})
    deps = load_dataframe("SELECT id, date, montant, categorie_id, ville_id, personne_id, libelle FROM depenses WHERE voyage_id=:v ORDER BY date, id", {"v": voyage_id})
    per = load_dataframe("SELECT date, categorie_id, montant FROM perdiem WHERE voyage_id=:v ORDER BY date, categorie_id", {"v": voyage_id})
    payload = {
        "voyage": voyage,
        "personnes": people.to_dict(orient="records"),
        "villes": cities.to_dict(orient="records"),
        "categories": cats.to_dict(orient="records"),
        "depenses": deps.to_dict(orient="records"),
        "per_diem": per.to_dict(orient="records"),
    }
    path = f"export_voyage_{voyage_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

# ----------------------
# Auth / UI helpers
# ----------------------
def verify_password(pw: str) -> bool:
    if not pw:
        return False
    hash_from_secrets = st.secrets.get("APP_PASSWORD_HASH") if hasattr(st, "secrets") else os.getenv("APP_PASSWORD_HASH")
    if not hash_from_secrets:
        return True
    try:
        return bcrypt.checkpw(pw.encode(), hash_from_secrets.encode())
    except Exception:
        return False

def ui_login():
    st.title(APP_TITLE)
    st.caption("Accès protégé (optionnel) — mot de passe via st.secrets")
    pw = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if verify_password(pw):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            time.sleep(1)
            st.error("Mot de passe incorrect")

# ----------------------
# UI: création voyage, saisie, tableaux...
# ----------------------
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
        personnes_raw = st.text_input("Personnes (séparées par des virgules)", placeholder="Alice, Bob")
        cats_raw = st.text_input("Catégories (format: Nom:Budget, séparées par des virgules)", placeholder="Bouffe:300000, Logement:600000")
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
    st.markdown(f"### {v['nom']}  ·  {pd.to_datetime(v['date_debut']).date()} → {pd.to_datetime(v['date_fin']).date()}  (JST)")
    st.caption("Per-diem = solde journalier **par catégorie**. Les dépenses diminuent immédiatement le solde; à minuit JST le reste est reporté au lendemain (même catégorie).")

def ui_kpis(voyage_id: int):
    # consolide les jours passés
    consolidate_until_today(voyage_id)
    k = compute_kpis(voyage_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget global estimé", f"¥{k['budget_courant']:,}")
    c2.metric("Per-diem total aujourd'hui (toutes catégories)", f"¥{k['perdiem_du_jour']:,}")
    c3.metric("Avance (+) / Retard (−) (cumul)", f"¥{k['avance_retard']:,}")
    c4.metric("Dépenses cumulées", f"¥{k['depenses_totales']:,}")

    # courbes cumulées prévu vs réel
    per = load_dataframe("SELECT date, montant FROM perdiem WHERE voyage_id=:v ORDER BY date", {"v": voyage_id})
    dep = load_dataframe("SELECT date, montant FROM depenses WHERE voyage_id=:v ORDER BY date", {"v": voyage_id})
    if not per.empty:
        per["date"] = pd.to_datetime(per["date"])
        per_sum = per.groupby("date", as_index=False)["montant"].sum().rename(columns={"montant": "cumule"})
        per_sum["cumule"] = per_sum["cumule"].cumsum()
    else:
        per_sum = pd.DataFrame({"date": [], "cumule": []})
    if not dep.empty:
        dep["date"] = pd.to_datetime(dep["date"])
        actual = dep.groupby("date", as_index=False)["montant"].sum().rename(columns={"montant": "journalier"})
        actual["cumule"] = actual["journalier"].cumsum()
    else:
        actual = pd.DataFrame({"date": [], "cumule": []})
    ch1 = alt.Chart(per_sum).mark_line().encode(x="date:T", y="cumule:Q", tooltip=["date:T", "cumule:Q"])
    ch2 = alt.Chart(actual).mark_line().encode(x="date:T", y="cumule:Q", tooltip=["date:T", "cumule:Q"])
    st.altair_chart(alt.layer(ch1, ch2).resolve_scale(y="independent").properties(height=260), use_container_width=True)

    # per-diem today by category
    today = jst_today()
    per_today = load_dataframe("SELECT p.categorie_id, c.nom, p.montant FROM perdiem p JOIN categories c ON c.id=p.categorie_id WHERE p.voyage_id=:v AND p.date=:d ORDER BY c.nom", {"v": voyage_id, "d": today})
    if not per_today.empty:
        st.markdown("**Per-diem du jour par catégorie**")
        st.dataframe(per_today.rename(columns={"categorie_id": "id", "nom": "categorie", "montant": "montant (¥)"}), use_container_width=True, hide_index=True)
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
            cat = st.selectbox("Catégorie", options=cats["id"].tolist(), format_func=lambda i: cats.set_index("id").loc[i, "nom"])
            ville_existing = st.selectbox("Ville (existante)", options=[None] + cities["id"].tolist() if not cities.empty else [None], format_func=lambda x: "—" if x is None else cities.set_index("id").loc[x, "nom"])
            ville_new = st.text_input("… ou nouvelle ville")
            pers = st.selectbox("Personne", options=[None] + people["id"].tolist() if not people.empty else [None], format_func=lambda x: "—" if x is None else people.set_index("id").loc[x, "nom"])
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
        recent = load_dataframe("""
            SELECT d.id, d.date, d.montant, c.nom as categorie, v.nom as ville, p.nom as personne, d.libelle
            FROM depenses d JOIN categories c ON c.id=d.categorie_id
            LEFT JOIN villes v ON v.id=d.ville_id
            LEFT JOIN personnes p ON p.id=d.personne_id
            WHERE d.voyage_id=:v
            ORDER BY d.date DESC, d.id DESC
            LIMIT 20
        """, {"v": voyage_id})
        st.dataframe(recent, use_container_width=True, hide_index=True)

def ui_listes(voyage_id: int):
    st.subheader("Listes & Filtres")
    dep = load_dataframe("""
        SELECT d.id, d.date, d.montant, c.nom AS categorie, v.nom AS ville, p.nom AS personne, d.libelle
        FROM depenses d
        JOIN categories c ON c.id=d.categorie_id
        LEFT JOIN villes v ON v.id=d.ville_id
        LEFT JOIN personnes p ON p.id=d.personne_id
        WHERE d.voyage_id=:v
        ORDER BY d.date DESC, d.id DESC
    """, {"v": voyage_id})
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
    per = load_dataframe("SELECT p.date, c.nom AS categorie, p.montant FROM perdiem p JOIN categories c ON c.id=p.categorie_id WHERE p.voyage_id=:v ORDER BY p.date, c.nom", {"v": voyage_id})
    if per.empty:
        st.info("Aucun per-diem initialisé.")
        return
    per["date"] = pd.to_datetime(per["date"]).dt.date
    pivot = per.pivot(index="date", columns="categorie", values="montant").fillna(0).astype(int)
    st.markdown("**Matrice date × catégorie (solde restant)**")
    st.dataframe(pivot, use_container_width=True)
    st.markdown("**Graphique : total par jour (somme des catégories)**")
    tot = pivot.sum(axis=1).reset_index().rename(columns={0: "total"}) if len(pivot) else pd.DataFrame()
    if not tot.empty:
        tot.columns = ["date", "total"]
        tot["date"] = pd.to_datetime(tot["date"])
        chart = alt.Chart(tot).mark_line().encode(x="date:T", y="total:Q", tooltip=["date:T", "total:Q"])
        st.altair_chart(chart, use_container_width=True)

def ui_admin(voyage_id: int):
    st.subheader("Administration")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Archiver (lecture seule)"):
            run_query("UPDATE voyages SET etat='archivé' WHERE id=:v", {"v": voyage_id})
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
        v = load_dataframe("SELECT nom FROM voyages WHERE id=:v", {"v": voyage_id}).iloc[0]
        name = st.text_input("Nom du voyage")
        if st.button("Confirmer suppression"):
            if name == v["nom"]:
                run_query("DELETE FROM voyages WHERE id=:v", {"v": voyage_id})
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

    # sidebar voyages + création
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
