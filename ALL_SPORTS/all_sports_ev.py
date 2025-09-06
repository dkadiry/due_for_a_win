# fanduel_all_sports_ev.py
import os, time, requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------
# Config (tune these)
# ---------------------------
load_dotenv()
API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("Put THE_ODDS_API_KEY in your .env")

REGIONS_CONSENSUS = "us,uk,eu,au"   # wide net for consensus
TARGET_BOOK = "fanduel"             # what you actually bet
MARKETS = ["h2h", "spreads", "totals"]
ODDS_FORMAT = "decimal"

# Rate-limit/credit safety
MAX_SPORTS = 12           # cap how many sports to scan (set None to scan all active)
MAX_EVENTS_PER_SPORT = 40 # cap events per sport (None for all)
SLEEP_BETWEEN_CALLS = 0.35

# Edge thresholds
MIN_EV_PRIMARY = 0.02     # show picks with EV >= 2%
MIN_EV_FALLBACK = 0.01    # fallback to 1% if none for a sport

# ---------------------------
# Helpers
# ---------------------------
def implied_from_decimal(d: float) -> float:
    return 1.0 / float(d)

def devig_proportional(probs):
    s = float(sum(probs))
    if s <= 0: return probs
    return [p / s for p in probs]

def ev_per_dollar(p_true: float, dec_odds: float) -> float:
    return p_true * (dec_odds - 1.0) - (1.0 - p_true)

def kelly_fraction(p_true: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    if b <= 0: return 0.0
    num = p_true * b - (1.0 - p_true)
    return max(0.0, num / b)

def fetch_json(url, params):
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining"), r.headers.get("x-requests-used")


def list_active_sports():
    url = "https://api.the-odds-api.com/v4/sports"
    data, _, _ = fetch_json(url, {"apiKey": API_KEY})
    # Keep only active sports; stable order by group/title
    active = [s for s in data if s.get("active")]
    active.sort(key=lambda x: (x.get("group",""), x.get("title","")))
    return active

def get_market(bookmaker, key):
    for m in bookmaker.get("markets", []):
        if m.get("key") == key:
            return m
    return None

def is_three_way_h2h(sport_key: str) -> bool:
    # Soccer, hockey OT markets, some esports use 3-way h2h (Home/Draw/Away).
    # The Odds API doesn’t label it explicitly, so detect dynamically per event (count outcomes),
    # but a prior hint helps avoid assumptions:
    return sport_key.startswith("soccer_")

# ---------------------------
# Consensus & EV per event/market
# ---------------------------
def consensus_rows_for_event(event, sport_key, market_key, exclude_book=TARGET_BOOK):
    """
    Build consensus vs FanDuel rows for a single event & market.
    - h2h: supports 2-way or 3-way automatically (based on FanDuel’s outcomes).
    - spreads/totals: exact point match only (sides at same line).
    """
    game_label = f"{event['away_team']}@{event['home_team']}"
    commence = event["commence_time"]

    # Find FanDuel market (we only price selections FD actually offers)
    fd = next((b for b in event.get("bookmakers", []) if b.get("key","").lower() == exclude_book), None)
    if not fd: return []
    fd_mkt = get_market(fd, market_key)
    if not fd_mkt or not fd_mkt.get("outcomes"): return []

    rows = []

    if market_key == "h2h":
        # Use FD’s outcomes to fix ordering (2-way or 3-way)
        selections = [o["name"] for o in fd_mkt["outcomes"]]
        n = len(selections)
        if n < 2: return []

        # Build price table only for these selections
        price_table = {sel: {} for sel in selections}
        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key","").lower()
            m = get_market(bk, "h2h")
            if not m: continue
            name_to_price = {o["name"]: float(o["price"]) for o in m.get("outcomes", [])}
            if any(s not in name_to_price for s in selections):
                continue
            for s in selections:
                price_table[s][bk_key] = name_to_price[s]

        # Build de-vigged vector per non-FD book
        books_for_consensus = sorted({bk for d in price_table.values() for bk in d if bk != exclude_book})
        fair_per_book, used_books = [], []
        for bk in books_for_consensus:
            if any(bk not in price_table[s] for s in selections):
                continue
            implied = [implied_from_decimal(price_table[s][bk]) for s in selections]
            fair = devig_proportional(implied)  # N-way normalize
            fair_per_book.append(fair); used_books.append(bk)
        if not fair_per_book: return []

        consensus = np.mean(np.array(fair_per_book), axis=0).tolist()

        # Emit rows vs FanDuel
        for i, s in enumerate(selections):
            fd_dec = price_table[s].get(exclude_book)
            if not fd_dec: continue
            p = consensus[i]
            rows.append({
                "Sport": sport_key, "Game": game_label, "Start (UTC)": commence,
                "Market": market_key, "Selection": s,
                "Consensus P": round(p,4), "Fair Odds": round(1.0/p,3),
                "FanDuel (dec)": round(fd_dec,3),
                "EV per $1": round(ev_per_dollar(p, fd_dec),4),
                "Kelly": round(kelly_fraction(p, fd_dec),4),
                "Consensus Books Used": ", ".join(used_books)
            })
        return rows

    # spreads/totals (2-way) — exact point match
    fd_outcomes = []
    for o in fd_mkt.get("outcomes", []):
        side = o["name"]
        point = o.get("point")
        if point is None:  # we need the line/total value
            continue
        fd_outcomes.append((side, point, float(o["price"])))

    for side, point, fd_dec in fd_outcomes:
        # find partner side at same point
        partner = next((o for o in fd_mkt["outcomes"] if o.get("point")==point and o["name"]!=side), None)
        if not partner: continue
        other_side = partner["name"]

        # Collect matching books with BOTH sides at SAME point
        fair_pairs, used_books = [], []
        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key","").lower()
            if bk_key == exclude_book: continue
            m = get_market(bk, market_key)
            if not m: continue
            name_to_price = {}
            for oo in m.get("outcomes", []):
                if oo.get("point") == point:
                    name_to_price[oo["name"]] = float(oo["price"])
            if side in name_to_price and other_side in name_to_price:
                imp_side  = implied_from_decimal(name_to_price[side])
                imp_other = implied_from_decimal(name_to_price[other_side])
                ssum = imp_side + imp_other
                if ssum <= 0: continue
                fair_pairs.append([imp_side/ssum, imp_other/ssum])
                used_books.append(bk_key)

        if not fair_pairs: continue

        fair_arr = np.array(fair_pairs)
        p_consensus_side = float(fair_arr[:,0].mean())

        label = f"{side} {point}"
        p = p_consensus_side
        rows.append({
            "Sport": sport_key, "Game": game_label, "Start (UTC)": commence,
            "Market": market_key, "Selection": label,
            "Consensus P": round(p,4), "Win Prob (Consensus)": round(p * 100, 1), "Fair Odds": round(1.0/p,3),
            "FanDuel (dec)": round(fd_dec,3),
            "EV per $1": round(ev_per_dollar(p, fd_dec),4),
            "Kelly": round(kelly_fraction(p, fd_dec),4),
            "Consensus Books Used": ", ".join(sorted(set(used_books)))
        })
    return rows

# ---------------------------
# Main driver
# ---------------------------
def run():
    # 1) Discover sports
    sports = list_active_sports()
    if MAX_SPORTS:
        sports = sports[:MAX_SPORTS]

    all_rows = []
    checked_sports = [] 

    for s in sports:
        sport_key = s["key"]
        sport_title = s.get("title","?")
        got_rows_for_this = False

        # 2) For each market, fetch odds for this sport
        for mk in MARKETS:
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            params = dict(apiKey=API_KEY, regions=REGIONS_CONSENSUS, markets=mk, oddsFormat=ODDS_FORMAT)
            try:
                data, rem, used = fetch_json(url, params)
            except requests.HTTPError as e:
                print(f"[skip] {sport_key} {mk}: {e}")
                continue
            print(f"Number of requests used: {used}, Number of requests remaining: {rem}")

            # Cap events per sport to save credits
            events = data[:MAX_EVENTS_PER_SPORT] if MAX_EVENTS_PER_SPORT else data

            for ev in events:
                rows = consensus_rows_for_event(ev, sport_key, mk, exclude_book=TARGET_BOOK)
                all_rows.extend(rows)

            time.sleep(SLEEP_BETWEEN_CALLS)

        if got_rows_for_this:
            checked_sports.append(f"{sport_title} ({sport_key})")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data/edges produced. Try different regions, timing, or thresholds.")
        return df

    # rank by EV inside each game; sort overall
    df["EV Rank (game)"] = df.groupby(["Sport","Game"])["EV per $1"].rank(ascending=False, method="dense")
    df = df.sort_values(["EV per $1","Sport","Game"], ascending=[False, True, True]).reset_index(drop=True)

    # 3) Build ready-to-bet singles (positive EV)
    singles = df[df["EV per $1"] >= MIN_EV_PRIMARY].copy()
    if singles.empty:
        singles = df[df["EV per $1"] >= MIN_EV_FALLBACK].copy()

    # best pick per sport
    best_per_sport = (
        singles.sort_values("EV per $1", ascending=False)
        .groupby("Sport", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # best overall
    best_overall = singles.sort_values("EV per $1", ascending=False).head(1).copy()

    # write outputs
    df.to_csv("ALL_SPORTS/all_candidates.csv", index=False)
    singles.to_csv("ALL_SPORTS/ready_singles_all.csv", index=False)
    best_per_sport.to_csv("ALL_SPORTS/best_per_sport.csv", index=False)
    best_overall.to_csv("ALL_SPORTS/best_overall.csv", index=False)

    #print("\nTop 10 by EV:")
    #print(df.head(10).to_string(index=False))
    print("\nChecked sports:")
    for s in checked_sports:
        print(" -", s)

    print("\nWrote: all_candidates.csv, ready_singles_all.csv, best_per_sport.csv, best_overall.csv")
    return df

if __name__ == "__main__":
    run()
