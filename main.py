import os
import time
import json
import bcrypt
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import pytz
import streamlit as st
import altair as alt

# =============================
# ---- Configuration -----
# =============================
APP_TITLE = "Travel Wallet — MVP"
JST = pytz.timezone("Asia/Tokyo")  # Fixed timezone (UTC+9)
DB_PATH = os.getenv("TRAVEL_WALLET_DB", "travel_wallet.db")

# Password: store a bcrypt hash in Streamlit secrets or env var
#   st.secrets["APP_PASSWORD_HASH"] or env "APP_PASSWORD_HASH"

# =============================
# ---- Utilities -----
# =============================

def jst_today():
    return datetime.now(JST).date()


def jst_now():
    return datetime.now(JST)


def to_date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def from_date_str(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS voyages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nom TEXT NOT NULL,
                date_debut TEXT NOT NULL,
                date_fin TEXT NOT NULL,
                tz TEXT NOT NULL DEFAULT 'JST',
                etat TEXT NOT NULL DEFAULT 'actif',
                budget_global_initial INTEGER NOT NULL,
                last_consolidation TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS personnes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voyage_id INTEGER NOT NULL,
                nom TEXT NOT NULL,
                FOREIGN KEY (voyage_id) REFERENCES voyages(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS villes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voyage_id INTEGER NOT NULL,
                nom TEXT NOT NULL,
                UNIQUE(voyage_id, nom),
                FOREIGN KEY (voyage_id) REFERENCES voyages(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voyage_id INTEGER NOT NULL,
                nom TEXT NOT NULL,
                couleur TEXT,
                icone TEXT,
                budget_initial INTEGER NOT NULL,
                est_autres INTEGER NOT NULL DEFAULT 0,
                UNIQUE(voyage_id, nom),
                FOREIGN KEY (voyage_id) REFERENCES voyages(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS depenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voyage_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                montant INTEGER NOT NULL,
                categorie_id INTEGER NOT NULL,
                ville_id INTEGER,
                personne_id INTEGER,
                libelle TEXT,
                FOREIGN KEY (voyage_id) REFERENCES voyages(id) ON DELETE CASCADE,
                FOREIGN KEY (categorie_id) REFERENCES categories(id),
                FOREIGN KEY (ville_id) REFERENCES villes(id),
                FOREIGN KEY (personne_id) REFERENCES personnes(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recettes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voyage_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                montant INTEGER NOT NULL,
                libelle TEXT,
                FOREIGN KEY (voyage_id) REFERENCES voyages(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS perdiem (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voyage_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                montant INTEGER NOT NULL,
                consolidee INTEGER NOT NULL DEFAULT 0,
                UNIQUE(voyage_id, date),
                FOREIGN KEY (voyage_id) REFERENCES voyages(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voyage_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                amount INTEGER NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (voyage_id) REFERENCES voyages(id) ON DELETE CASCADE
            );
            """
        )


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
        st.warning("Aucun mot de passe configuré. Définissez APP_PASSWORD_HASH (bcrypt). Accès libre pour le dev.")
        return True
    try:
        return bcrypt.checkpw(pw.encode(), hash_from_secrets.encode())
    except Exception:
        st.error("Hash bcrypt invalide dans la config.")
        return False


# =============================
# ---- Domain logic -----
# =============================

def daterange(d0: date, d1: date):
    for n in range((d1 - d0).days + 1):
        yield d0 + timedelta(days=n)


def create_voyage(nom: str, d0: date, d1: date, budget_global: int,
                   personnes: list[str], categories_init: list[dict]):
    assert budget_global > 0
    assert d1 >= d0
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO voyages(nom, date_debut, date_fin, tz, etat, budget_global_initial, last_consolidation) VALUES(?,?,?,?,?,?,?)",
            (nom, to_date_str(d0), to_date_str(d1), "JST", "actif", budget_global, None),
        )
        voyage_id = cur.lastrowid

        # personnes (closed list)
        for p in personnes:
            cur.execute("INSERT INTO personnes(voyage_id, nom) VALUES(?,?)", (voyage_id, p.strip()))

        # categories (user + Autres if remainder > 0)
        sum_cat = sum(int(c.get("budget", 0)) for c in categories_init)
        if sum_cat > budget_global:
            raise ValueError("Somme des budgets de catégories > budget global")
        for c in categories_init:
            cur.execute(
                "INSERT INTO categories(voyage_id, nom, couleur, icone, budget_initial, est_autres) VALUES(?,?,?,?,?,0)",
                (voyage_id, c["nom"].strip(), c.get("couleur"), c.get("icone"), int(c.get("budget", 0))),
            )
        reste = budget_global - sum_cat
        if reste > 0:
            cur.execute(
                "INSERT INTO categories(voyage_id, nom, budget_initial, est_autres) VALUES(?,?,?,1)",
                (voyage_id, "Autres", reste),
            )

        # initial per-diem distribution
        N = (d1 - d0).days + 1
        B = budget_global
        q = B // N
        r = B % N
        amounts = [q + 1 if i < r else q for i in range(N)]
        for i, dt in enumerate(daterange(d0, d1)):
            cur.execute(
                "INSERT INTO perdiem(voyage_id, date, montant, consolidee) VALUES(?,?,?,0)",
                (voyage_id, to_date_str(dt), int(amounts[i]), 0),
            )
        return voyage_id


def load_dataframe(query: str, params: tuple = ()):  # tiny helper
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_voyages(active_only=True) -> pd.DataFrame:
    q = "SELECT * FROM voyages" + (" WHERE etat='actif'" if active_only else "") + " ORDER BY id DESC"
    return load_dataframe(q)


def get_entities(voyage_id: int):
    people = load_dataframe("SELECT id, nom FROM personnes WHERE voyage_id=? ORDER BY id", (voyage_id,))
    cities = load_dataframe("SELECT id, nom FROM villes WHERE voyage_id=? ORDER BY nom", (voyage_id,))
    cats = load_dataframe("SELECT id, nom, budget_initial, est_autres FROM categories WHERE voyage_id=? ORDER BY est_autres, nom", (voyage_id,))
    return people, cities, cats


def add_city_if_needed(voyage_id: int, nom_ville: str) -> int:
    if not nom_ville:
        return None
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM villes WHERE voyage_id=? AND nom=?", (voyage_id, nom_ville.strip()))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("INSERT INTO villes(voyage_id, nom) VALUES(?,?)", (voyage_id, nom_ville.strip()))
        return cur.lastrowid


def add_depense(voyage_id: int, dt: date, montant: int, categorie_id: int, ville_id: int, personne_id: int, libelle: str):
    if montant <= 0:
        raise ValueError("Montant doit être > 0")
    # Validate date window and not in future (JST)
    v = load_dataframe("SELECT date_debut, date_fin FROM voyages WHERE id=?", (voyage_id,)).iloc[0]
    d0, d1 = from_date_str(v["date_debut"]), from_date_str(v["date_fin"]) 
    today_jst = jst_today()
    if dt < d0 or dt > d1:
        raise ValueError("Date hors période de voyage")
    if dt > today_jst:
        raise ValueError("Saisie future interdite")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO depenses(voyage_id, date, montant, categorie_id, ville_id, personne_id, libelle) VALUES(?,?,?,?,?,?,?)",
            (voyage_id, to_date_str(dt), int(montant), int(categorie_id), ville_id, personne_id, libelle),
        )
        # Register a negative adjustment to future days if the expense is on a past day (strictly < today)
        if dt < today_jst:
            cur.execute(
                "INSERT INTO pending_adjustments(voyage_id, date, amount, processed, created_at) VALUES(?,?,?,?,?)",
                (voyage_id, to_date_str(dt), -int(montant), 0, jst_now().isoformat()),
            )


def edit_depense(depense_id: int, new_dt: date, new_montant: int, new_categorie_id: int, new_ville_id: int, new_personne_id: int, new_libelle: str):
    with get_conn() as conn:
        cur = conn.cursor()
        old = cur.execute("SELECT voyage_id, date, montant FROM depenses WHERE id=?", (depense_id,)).fetchone()
        if not old:
            raise ValueError("Dépense introuvable")
        voyage_id, old_dt_str, old_amount = old[0], old[1], int(old[2])
        v = load_dataframe("SELECT date_debut, date_fin FROM voyages WHERE id=?", (voyage_id,)).iloc[0]
        d0, d1 = from_date_str(v["date_debut"]), from_date_str(v["date_fin"]) 
        today_jst = jst_today()
        if new_dt < d0 or new_dt > d1:
            raise ValueError("Date hors période de voyage")
        if new_dt > today_jst:
            raise ValueError("Saisie future interdite")
        cur.execute(
            "UPDATE depenses SET date=?, montant=?, categorie_id=?, ville_id=?, personne_id=?, libelle=? WHERE id=?",
            (to_date_str(new_dt), int(new_montant), int(new_categorie_id), new_ville_id, new_personne_id, new_libelle, depense_id),
        )
        # Variation amount vs old; if effective date is in the past relative to today JST, enqueue adjustment
        variation = int(new_montant) - old_amount
        old_dt = from_date_str(old_dt_str)
        # Any change that affects a past day (either old or new date < today) should be applied at next midnight to days > affected date
        affected_date = min(old_dt, new_dt)
        if affected_date < today_jst and variation != 0:
            cur.execute(
                "INSERT INTO pending_adjustments(voyage_id, date, amount, processed, created_at) VALUES(?,?,?,?,?)",
                (voyage_id, to_date_str(affected_date), -variation, 0, jst_now().isoformat()),
            )


def delete_depense(depense_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        row = cur.execute("SELECT voyage_id, date, montant FROM depenses WHERE id=?", (depense_id,)).fetchone()
        if not row:
            return
        voyage_id, dt_str, montant = row[0], row[1], int(row[2])
        cur.execute("DELETE FROM depenses WHERE id=?", (depense_id,))
        # Deleting a past expense is a positive variation
        if from_date_str(dt_str) < jst_today():
            cur.execute(
                "INSERT INTO pending_adjustments(voyage_id, date, amount, processed, created_at) VALUES(?,?,?,?,?)",
                (voyage_id, dt_str, int(montant), 0, jst_now().isoformat()),
            )


def add_recette(voyage_id: int, dt: date, montant: int, libelle: str):
    if montant <= 0:
        raise ValueError("Montant doit être > 0")
    v = load_dataframe("SELECT date_debut, date_fin FROM voyages WHERE id=?", (voyage_id,)).iloc[0]
    d0, d1 = from_date_str(v["date_debut"]), from_date_str(v["date_fin"]) 
    today_jst = jst_today()
    if dt < d0 or dt > d1:
        raise ValueError("Date hors période de voyage")
    if dt > today_jst:
        raise ValueError("Saisie future interdite")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO recettes(voyage_id, date, montant, libelle) VALUES(?,?,?,?)", (voyage_id, to_date_str(dt), int(montant), libelle))
        # Recette impacts only the global; redistribute to days > dt at next midnight
        cur.execute(
            "INSERT INTO pending_adjustments(voyage_id, date, amount, processed, created_at) VALUES(?,?,?,?,?)",
            (voyage_id, to_date_str(dt), int(montant), 0, jst_now().isoformat()),
        )


def get_trip_window(voyage_id: int) -> tuple[date, date]:
    v = load_dataframe("SELECT date_debut, date_fin FROM voyages WHERE id=?", (voyage_id,)).iloc[0]
    return from_date_str(v["date_debut"]), from_date_str(v["date_fin"]) 


def redistribute(voyage_id: int, start_exclusive: date, total_delta: int):
    """Redistribute delta across remaining days strictly > start_exclusive using equal-split with remainder to earliest days, floor at 0.
    Positive delta increases future per-diem; negative reduces. Never drop a day below 0.
    """
    if total_delta == 0:
        return
    d0, d1 = get_trip_window(voyage_id)
    with get_conn() as conn:
        cur = conn.cursor()
        # Candidate target dates
        targets = [to_date_str(d) for d in daterange(max(start_exclusive + timedelta(days=1), d0), d1)]
        if not targets:
            return
        R = len(targets)
        a = abs(total_delta) // R
        r = abs(total_delta) % R
        if total_delta > 0:
            # add a to all, +1 to first r
            for i, ds in enumerate(targets):
                inc = a + (1 if i < r else 0)
                cur.execute("UPDATE perdiem SET montant = montant + ? WHERE voyage_id=? AND date=?", (inc, voyage_id, ds))
        else:
            # subtract a from all, -1 from first r, but never below 0
            # Do two passes to respect floor 0:
            # 1) compute tentative decrements per day
            decrements = [(a + (1 if i < r else 0)) for i in range(R)]
            for ds, dec in zip(targets, decrements):
                # fetch current
                row = cur.execute("SELECT montant FROM perdiem WHERE voyage_id=? AND date=?", (voyage_id, ds)).fetchone()
                if not row:
                    continue
                current = int(row[0])
                new_val = max(0, current - dec)
                cur.execute("UPDATE perdiem SET montant=? WHERE voyage_id=? AND date=?", (new_val, voyage_id, ds))


def consolidate_until_today(voyage_id: int):
    """Run consolidation for all unconsolidated past days, and process pending adjustments whose date < today JST.
    This function is idempotent per day thanks to flags and will be executed on each app run."""
    today = jst_today()
    d0, d1 = get_trip_window(voyage_id)
    with get_conn() as conn:
        cur = conn.cursor()
        # Consolidate day by day for dates < today
        rows = cur.execute(
            "SELECT date, montant, consolidee FROM perdiem WHERE voyage_id=? ORDER BY date",
            (voyage_id,),
        ).fetchall()
        for row in rows:
            d = from_date_str(row[0])
            if d >= today:
                break
            if int(row[2]) == 1:
                continue  # already consolidated
            D_t = int(row[1])
            # Spend of the day
            spent = cur.execute(
                "SELECT COALESCE(SUM(montant),0) FROM depenses WHERE voyage_id=? AND date=?",
                (voyage_id, row[0]),
            ).fetchone()[0]
            spent = int(spent)
            E_t = D_t - spent
            # Redistribute Et over remaining days
            redistribute(voyage_id, start_exclusive=d, total_delta=E_t)
            # mark consolidated
            cur.execute("UPDATE perdiem SET consolidee=1 WHERE voyage_id=? AND date=?", (voyage_id, row[0]))

        # Process pending adjustments dated < today
        adj_rows = cur.execute(
            "SELECT id, date, amount FROM pending_adjustments WHERE voyage_id=? AND processed=0 AND date < ? ORDER BY id",
            (voyage_id, to_date_str(today)),
        ).fetchall()
        for aid, ds, amount in adj_rows:
            d_adj = from_date_str(ds)
            redistribute(voyage_id, start_exclusive=d_adj, total_delta=int(amount))
            cur.execute("UPDATE pending_adjustments SET processed=1 WHERE id=?", (aid,))


# =============================
# ---- Metrics / Queries -----
# =============================

def compute_kpis(voyage_id: int):
    d0, d1 = get_trip_window(voyage_id)
    today = jst_today()
    per = load_dataframe("SELECT date, montant FROM perdiem WHERE voyage_id=? ORDER BY date", (voyage_id,))
    per["date"] = pd.to_datetime(per["date"]).dt.date
    dep = load_dataframe("SELECT date, montant FROM depenses WHERE voyage_id=?", (voyage_id,))
    dep["date"] = pd.to_datetime(dep["date"]).dt.date if len(dep) else pd.Series(dtype="datetime64[ns]")
    rec = load_dataframe("SELECT date, montant FROM recettes WHERE voyage_id=?", (voyage_id,))
    rec["date"] = pd.to_datetime(rec["date"]).dt.date if len(rec) else pd.Series(dtype="datetime64[ns]")

    B0 = int(load_dataframe("SELECT budget_global_initial FROM voyages WHERE id=?", (voyage_id,)).iloc[0,0])
    total_recettes = int(rec["montant"].sum()) if len(rec) else 0
    total_depenses = int(dep["montant"].sum()) if len(dep) else 0

    # Global current budget = initial + recettes - depenses
    Bc = B0 + total_recettes - total_depenses

    # Today's per-diem (planned)
    today_row = per[per["date"] == today]
    D_today = int(today_row["montant"].iloc[0]) if len(today_row) else 0

    # Cumulative planned vs real up to today
    planned_until_today = int(per[per["date"] <= today]["montant"].sum())
    real_until_today = int(dep[dep["date"] <= today]["montant"].sum()) if len(dep) else 0
    avance = planned_until_today - real_until_today  # positive = ahead (underspent)

    return {
        "budget_courant": Bc,
        "perdiem_du_jour": D_today,
        "avance_retard": avance,
        "depenses_totales": total_depenses,
        "recettes_totales": total_recettes,
        "planned_until_today": planned_until_today,
        "real_until_today": real_until_today,
    }


def category_totals(voyage_id: int) -> pd.DataFrame:
    q = """
    SELECT c.id as categorie_id, c.nom as categorie, c.budget_initial, c.est_autres,
           COALESCE(SUM(d.montant),0) as depense
    FROM categories c
    LEFT JOIN depenses d ON d.categorie_id = c.id AND d.voyage_id=c.voyage_id
    WHERE c.voyage_id=?
    GROUP BY c.id, c.nom, c.budget_initial, c.est_autres
    ORDER BY c.est_autres DESC, c.nom
    """
    return load_dataframe(q, (voyage_id,))


# =============================
# ---- Export -----
# =============================

def export_json(voyage_id: int) -> str:
    with get_conn() as conn:
        cur = conn.cursor()
        v = cur.execute("SELECT * FROM voyages WHERE id=?", (voyage_id,)).fetchone()
        people = cur.execute("SELECT id, nom FROM personnes WHERE voyage_id=?", (voyage_id,)).fetchall()
        cities = cur.execute("SELECT id, nom FROM villes WHERE voyage_id=?", (voyage_id,)).fetchall()
        cats = cur.execute("SELECT id, nom, couleur, icone, budget_initial, est_autres FROM categories WHERE voyage_id=?", (voyage_id,)).fetchall()
        deps = cur.execute("SELECT id, date, montant, categorie_id, ville_id, personne_id, libelle FROM depenses WHERE voyage_id=? ORDER BY date, id", (voyage_id,)).fetchall()
        recs = cur.execute("SELECT id, date, montant, libelle FROM recettes WHERE voyage_id=? ORDER BY date, id", (voyage_id,)).fetchall()
        per = cur.execute("SELECT date, montant FROM perdiem WHERE voyage_id=? ORDER BY date", (voyage_id,)).fetchall()

    payload = {
        "voyage": {
            "id": v["id"],
            "nom": v["nom"],
            "date_debut": v["date_debut"],
            "date_fin": v["date_fin"],
            "fuseau": v["tz"],
            "etat": v["etat"],
        },
        "personnes": [dict(x) for x in people],
        "villes": [dict(x) for x in cities],
        "categories": [dict(x) for x in cats],
        "budgets": {
            "global_initial": v["budget_global_initial"],
            "global_courant": compute_kpis(voyage_id)["budget_courant"],
        },
        "depenses": [dict(x) for x in deps],
        "recettes": [dict(x) for x in recs],
        "per_diem": [dict(date=row["date"], montant=row["montant"]) for row in per],
        "parametres": {
            "interdiction_futur": True,
            "plancher_perdiem": 0,
            "tz_fixe": "JST",
        },
    }
    path = f"export_voyage_{voyage_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# =============================
# ---- UI -----
# =============================

def ui_login():
    st.title(APP_TITLE)
    st.caption("Accès protégé par mot de passe (familial)")
    pw = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter", type="primary"):
        if verify_password(pw):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            time.sleep(5)  # délai après mauvais mot de passe
            st.error("Mot de passe incorrect.")


def ui_create_voyage():
    st.subheader("Créer un voyage")
    with st.form("create_voyage_form"):
        nom = st.text_input("Nom du voyage", placeholder="Japon 2025")
        col1, col2 = st.columns(2)
        with col1:
            d0 = st.date_input("Date de début (JST)", value=jst_today())
        with col2:
            d1 = st.date_input("Date de fin (JST)", value=jst_today() + timedelta(days=6), min_value=d0)
        budget_global = st.number_input("Budget global initial (¥)", min_value=1, step=1)

        st.markdown("**Personnes** (liste fermée)")
        personnes_raw = st.text_input("Noms séparés par des virgules", placeholder="Ugo, Partenaire")

        st.markdown("**Catégories** (nom:budget, séparés par des virgules)")
        cats_raw = st.text_input("Ex.: Logement:300000, Repas:120000, Transport:100000")

        submitted = st.form_submit_button("Créer", type="primary")
        if submitted:
            try:
                personnes = [x.strip() for x in personnes_raw.split(",") if x.strip()]
                cats = []
                total_cats = 0
                for chunk in [x for x in cats_raw.split(",") if x.strip()]:
                    if ":" not in chunk:
                        st.error(f"Catégorie invalide: '{chunk}' (format attendu nom:budget)")
                        return
                    n, b = chunk.split(":", 1)
                    b = int(b)
                    if b < 0:
                        st.error("Budget de catégorie négatif interdit")
                        return
                    cats.append({"nom": n.strip(), "budget": b})
                    total_cats += b
                if total_cats > budget_global:
                    st.error("Somme des budgets de catégories > budget global")
                    return
                voyage_id = create_voyage(nom, d0, d1, int(budget_global), personnes, cats)
                st.success(f"Voyage créé (id={voyage_id}). Distribution per-diem initialisée.")
                st.rerun()
            except Exception as e:
                st.exception(e)


def ui_voyage_header(v: pd.Series):
    st.markdown(f"### {v['nom']}  ·  {v['date_debut']} → {v['date_fin']}  (JST)")
    st.caption("Saisie autorisée pour aujourd'hui et le passé uniquement. Les consolidations s'exécutent chaque minuit JST.")


def ui_kpis(voyage_id: int):
    consolidate_until_today(voyage_id)
    k = compute_kpis(voyage_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget global restant", f"¥{k['budget_courant']:,}")
    c2.metric("Per-diem du jour", f"¥{k['perdiem_du_jour']:,}")
    c3.metric("Avance (+) / Retard (−)", f"¥{k['avance_retard']:,}")
    c4.metric("Dépenses cumulées", f"¥{k['depenses_totales']:,}")

    # Courbe prévu vs réel
    per = load_dataframe("SELECT date, montant FROM perdiem WHERE voyage_id=? ORDER BY date", (voyage_id,))
    per["date"] = pd.to_datetime(per["date"]) 
    dep = load_dataframe("SELECT date, montant FROM depenses WHERE voyage_id=? ORDER BY date", (voyage_id,))
    if len(dep):
        dep["date"] = pd.to_datetime(dep["date"]) 
        actual = dep.groupby("date", as_index=False)["montant"].sum().rename(columns={"montant": "journalier"})
        actual["cumule"] = actual["journalier"].cumsum()
    else:
        actual = pd.DataFrame({"date": [], "cumule": []})
    per["cumule"] = per["montant"].cumsum()
    base = alt.Chart(per).mark_line().encode(x="date:T")
    ch1 = base.encode(y=alt.Y("cumule:Q", title="Prévu (cumul)"))
    ch2 = alt.Chart(actual).mark_line().encode(x="date:T", y=alt.Y("cumule:Q", title="Réel (cumul)"))
    st.altair_chart(alt.layer(ch1, ch2).resolve_scale(y='independent').properties(height=260), use_container_width=True)


def ui_saisie(voyage_id: int):
    st.subheader("Saisie")
    people, cities, cats = get_entities(voyage_id)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dépense**")
        with st.form("dep_form"):
            dt = st.date_input("Date", value=jst_today())
            montant = st.number_input("Montant (¥)", min_value=1, step=1)
            cat = st.selectbox("Catégorie", options=cats["id"], format_func=lambda i: cats.set_index("id").loc[i, "nom"])
            ville_existing = st.selectbox("Ville (existante)", options=[None] + cities["id"].tolist(), format_func=lambda x: "—" if x is None else cities.set_index("id").loc[x, "nom"])
            ville_new = st.text_input("… ou nouvelle ville")
            pers = st.selectbox("Personne", options=[None] + people["id"].tolist(), format_func=lambda x: "—" if x is None else people.set_index("id").loc[x, "nom"])
            lib = st.text_input("Libellé (optionnel)")
            ok = st.form_submit_button("Ajouter", type="primary")
        if ok:
            try:
                ville_id = ville_existing
                if ville_new.strip():
                    ville_id = add_city_if_needed(voyage_id, ville_new)
                add_depense(voyage_id, dt, int(montant), int(cat), ville_id, pers, lib)
                st.success("Dépense ajoutée.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with col2:
        st.markdown("**Recette (ajout d'argent)**")
        with st.form("rec_form"):
            dt2 = st.date_input("Date ", value=jst_today(), key="r_date")
            montant2 = st.number_input("Montant (¥) ", min_value=1, step=1, key="r_amt")
            lib2 = st.text_input("Libellé (optionnel)", key="r_lbl")
            ok2 = st.form_submit_button("Ajouter la recette", type="secondary")
        if ok2:
            try:
                add_recette(voyage_id, dt2, int(montant2), lib2)
                st.success("Recette ajoutée (répercutée au prochain minuit JST).")
                st.rerun()
            except Exception as e:
                st.error(str(e))


def ui_listes(voyage_id: int):
    st.subheader("Listes & Filtres")
    dep = load_dataframe(
        """
        SELECT d.id, d.date, d.montant, c.nom AS categorie, v.nom AS ville, p.nom AS personne, d.libelle
        FROM depenses d
        JOIN categories c ON c.id=d.categorie_id
        LEFT JOIN villes v ON v.id=d.ville_id
        LEFT JOIN personnes p ON p.id=d.personne_id
        WHERE d.voyage_id=?
        ORDER BY d.date DESC, d.id DESC
        """,
        (voyage_id,)
    )
    st.dataframe(dep, use_container_width=True, hide_index=True)

    rec = load_dataframe("SELECT id, date, montant, libelle FROM recettes WHERE voyage_id=? ORDER BY date DESC, id DESC", (voyage_id,))
    with st.expander("Recettes"):
        st.dataframe(rec, use_container_width=True, hide_index=True)


def ui_categories(voyage_id: int):
    st.subheader("Par catégorie")
    df = category_totals(voyage_id)
    df["dépassement"] = df["depense"] > df["budget_initial"]
    def fmt_badge(row):
        if row["dépassement"]:
            return f"⚠️ Dépassement de ¥{row['depense']-row['budget_initial']:,}"
        return "OK"
    st.dataframe(
        df.assign(status=df.apply(fmt_badge, axis=1))[["categorie", "budget_initial", "depense", "status"]],
        use_container_width=True,
        hide_index=True
    )

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("categorie:N", sort='-y'), y=alt.Y("depense:Q"), tooltip=["categorie", "depense", "budget_initial"],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)


def ui_par_vue(voyage_id: int):
    st.subheader("Vues par ville / personne / jour")
    dep = load_dataframe(
        """
        SELECT d.date, COALESCE(v.nom,'(sans ville)') AS ville, COALESCE(p.nom,'(sans personne)') AS personne, c.nom AS categorie, d.montant
        FROM depenses d
        JOIN categories c ON c.id=d.categorie_id
        LEFT JOIN villes v ON v.id=d.ville_id
        LEFT JOIN personnes p ON p.id=d.personne_id
        WHERE d.voyage_id=?
        """,
        (voyage_id,),
    )
    if len(dep) == 0:
        st.info("Aucune dépense encore.")
        return
    dep["date"] = pd.to_datetime(dep["date"]).dt.date

    # Heatmap par jour
    heat_day = dep.groupby("date", as_index=False)["montant"].sum()
    st.markdown("**Heatmap — ¥/jour**")
    ch1 = alt.Chart(heat_day).mark_rect().encode(x="date:T", y=alt.value(20), color="montant:Q", tooltip=["date:T", "montant:Q"]).properties(height=60)
    st.altair_chart(ch1, use_container_width=True)

    # Heatmap par catégorie
    heat_cat = dep.groupby("categorie", as_index=False)["montant"].sum()
    st.markdown("**Heatmap — ¥/catégorie**")
    ch2 = alt.Chart(heat_cat).mark_bar().encode(x="categorie:N", y="montant:Q", tooltip=["categorie", "montant"]).properties(height=220)
    st.altair_chart(ch2, use_container_width=True)

    # Par ville / personne (totaux)
    col1, col2 = st.columns(2)
    with col1:
        city = dep.groupby("ville", as_index=False)["montant"].sum().sort_values("montant", ascending=False)
        st.markdown("**Par ville**")
        st.dataframe(city, use_container_width=True, hide_index=True)
    with col2:
        ppl = dep.groupby("personne", as_index=False)["montant"].sum().sort_values("montant", ascending=False)
        st.markdown("**Par personne**")
        st.dataframe(ppl, use_container_width=True, hide_index=True)


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
        if st.button("Supprimer le voyage", help="Action irréversible — la confirmation vous sera demandée"):
            st.session_state["confirm_delete"] = True
    if st.session_state.get("confirm_delete"):
        st.warning("Tapez le nom exact du voyage pour confirmer la suppression.")
        v = load_dataframe("SELECT nom FROM voyages WHERE id=?", (voyage_id,)).iloc[0]
        name = st.text_input("Nom du voyage")
        if st.button("Confirmer la suppression", type="primary"):
            if name == v["nom"]:
                with get_conn() as conn:
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("DELETE FROM voyages WHERE id=?", (voyage_id,))
                st.success("Voyage supprimé.")
                st.session_state.pop("confirm_delete")
                st.experimental_rerun()
            else:
                st.error("Nom incorrect.")


# =============================
# ---- Main App -----
# =============================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧳", layout="wide")
    init_db()

    if not st.session_state.get("auth_ok"):
        ui_login()
        return

    # Sidebar: select voyage or create new
    with st.sidebar:
        st.header("Voyages")
        voyages = get_voyages(active_only=False)
        if len(voyages) == 0:
            st.info("Aucun voyage. Créez-en un.")
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
    if not voyage_id and len(get_voyages(active_only=False)):
        voyage_id = int(get_voyages(active_only=False).iloc[0]["id"])  # fallback

    if not voyage_id:
        return

    v = get_voyages(active_only=False)
    v = v[v["id"] == voyage_id].iloc[0]

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
