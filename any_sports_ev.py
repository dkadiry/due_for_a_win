# fanduel_selected_sports_ev.py
import os, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  
from dotenv import load_dotenv

# ---------------------------
# Config (tune these)
# ---------------------------
load_dotenv()
API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("Put THE_ODDS_API_KEY in your .env")

# Only these sports:
ALLOWED_SPORT_KEYS = [
    #"soccer_epl",            # Premier League
    #"baseball_mlb",          # MLB
    #"basketball_ncaab",      # NCAA Men's Basketball
    "americanfootball_nfl",  # NFL
    #"americanfootball_ncaaf"  # College Football
]

REGIONS_CONSENSUS = "us"   # wide net for consensus
TARGET_BOOK = "fanduel"             # the book we are betting at
MARKETS = ["h2h", "spreads", "totals"]
ODDS_FORMAT = "decimal"

# Rate-limit/credit safety
MAX_EVENTS_PER_SPORT = None  # cap events per sport (None for all)
SLEEP_BETWEEN_CALLS = 0.35
RATE_LIMIT_MIN_REMAINING = 25  # stop early if quota dips below this

# Edge thresholds
MIN_EV_PRIMARY = 0.02     # show picks with EV >= 2%
MIN_EV_FALLBACK = 0.01    # fallback to 1% if none

# ---------------------------
# Helpers
# ---------------------------

ET = ZoneInfo("America/New_York")

def now_et():
    return datetime.now(tz=ET)

def parse_iso_to_et(s: str) -> datetime:
    # The Odds API returns ISO-8601 UTC (e.g., '2025-09-06T18:05:00Z')
    # Normalize 'Z' to '+00:00' for fromisoformat
    dt_utc = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt_utc.astimezone(ET)

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

def list_selected_sports():
    """Fetch all sports and return only the allowed ones, preserving title for printing."""
    url = "https://api.the-odds-api.com/v4/sports"
    data, rem, used = fetch_json(url, {"apiKey": API_KEY})
    selected = []
    for s in data:
        if not s.get("active"):
            continue
        if s.get("key") in ALLOWED_SPORT_KEYS:
            selected.append(s)
    # stable order by group/title
    selected.sort(key=lambda x: (x.get("group",""), x.get("title","")))
    return selected, rem, used

def get_market(bookmaker, key):
    for m in bookmaker.get("markets", []):
        if m.get("key") == key:
            return m
    return None

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
    start_et = parse_iso_to_et(event["commence_time"])
    # only future games (strict: must be later than 'now')
    if start_et <= now_et():
        return []

    # Find FanDuel market (we only price selections FD actually offers)
    fd = next((b for b in event.get("bookmakers", []) if b.get("key","").lower() == exclude_book), None)
    if not fd: 
        return []
    fd_mkt = get_market(fd, market_key)
    if not fd_mkt or not fd_mkt.get("outcomes"): 
        return []

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
        if not fair_per_book: 
            return []

        consensus = np.mean(np.array(fair_per_book), axis=0).tolist()

        # Emit rows vs FanDuel
        for i, s in enumerate(selections):
            fd_dec = price_table[s].get(exclude_book)
            if not fd_dec: 
                continue
            p = consensus[i]
            rows.append({
                "Sport": sport_key, "Game": game_label, "Start (ET)": start_et.strftime("%Y-%m-%d %H:%M"),
                "Market": market_key, "Selection": s,
                "Consensus P": round(p,4), "Win Prob (Consensus)": round(p*100,1),
                "Fair Odds": round(1.0/p,3),
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
        if point is None:
            continue
        fd_outcomes.append((side, point, float(o["price"])))

    for side, point, fd_dec in fd_outcomes:
        # find partner side at same point
        partner = next((o for o in fd_mkt["outcomes"] if o.get("point")==point and o["name"]!=side), None)
        if not partner: 
            continue
        other_side = partner["name"]

        # Collect matching books with BOTH sides at SAME point
        fair_pairs, used_books = [], []
        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key","").lower()
            if bk_key == exclude_book: 
                continue
            m = get_market(bk, market_key)
            if not m: 
                continue
            name_to_price = {}
            for oo in m.get("outcomes", []):
                if oo.get("point") == point:
                    name_to_price[oo["name"]] = float(oo["price"])
            if side in name_to_price and other_side in name_to_price:
                imp_side  = implied_from_decimal(name_to_price[side])
                imp_other = implied_from_decimal(name_to_price[other_side])
                ssum = imp_side + imp_other
                if ssum <= 0: 
                    continue
                fair_pairs.append([imp_side/ssum, imp_other/ssum])
                used_books.append(bk_key)

        if not fair_pairs: 
            continue

        fair_arr = np.array(fair_pairs)
        p_consensus_side = float(fair_arr[:,0].mean())

        label = f"{side} {point}"
        p = p_consensus_side
        rows.append({
            "Sport": sport_key, "Game": game_label, "Start (ET)": start_et.strftime("%Y-%m-%d %H:%M"),
            "Market": market_key, "Selection": label,
            "Consensus P": round(p,4), "Win Prob (Consensus)": round(p*100,1),
            "Fair Odds": round(1.0/p,3),
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
    # Only our selected sports
    sports, rem0, used0 = list_selected_sports()
    if used0 and rem0:
        print(f"Start quota — used: {used0}, remaining: {rem0}")

    all_rows = []
    checked_sports = []

    for s in sports:
        sport_key   = s["key"]
        sport_title = s.get("title","?")
        got_rows_for_this = False

        # Optional: trim markets per sport if you want to be conservative
        markets_plan = MARKETS[:]   # ["h2h","spreads","totals"]

        # Check quota before hammering a sport
        _, rem, used = fetch_json("https://api.the-odds-api.com/v4/sports", {"apiKey": API_KEY})
        try:
            rem_int = int(rem or 0)
        except:
            rem_int = 0
        print(f"Quota — used: {used}, remaining: {rem}")
        if rem_int < RATE_LIMIT_MIN_REMAINING:
            print("Stopping early to preserve credits.")
            break

        for mk in markets_plan:
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            params = dict(apiKey=API_KEY, regions=REGIONS_CONSENSUS, markets=mk, oddsFormat=ODDS_FORMAT)
            try:
                data, rem, used = fetch_json(url, params)
            except requests.HTTPError as e:
                print(f"[skip] {sport_key} {mk}: {e}")
                continue

            print(f"Requests used: {used}, remaining: {rem} | {sport_key}:{mk}")

            # Cap events per sport to save credits
            events = data[:MAX_EVENTS_PER_SPORT] if MAX_EVENTS_PER_SPORT else data

            for ev in events:
                rows = consensus_rows_for_event(ev, sport_key, mk, exclude_book=TARGET_BOOK)
                if rows:
                    got_rows_for_this = True
                    all_rows.extend(rows)

            time.sleep(SLEEP_BETWEEN_CALLS)

        if got_rows_for_this:
            checked_sports.append(f"{sport_title} ({sport_key})")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data/edges produced. Try different timing/thresholds.")
        print("\nChecked sports (with results):")
        for s in checked_sports:
            print(" -", s)
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

    # write outputs (keep your folder layout)
    #os.makedirs("FOUR_SPORTS", exist_ok=True)
    df.to_csv("just_college_football_candidates.csv", index=False)
    singles.to_csv("ready_singles_college_football.csv", index=False)
    #best_per_sport.to_csv("FOUR_SPORTS/best_per_sport.csv", index=False)
    best_overall.to_csv("best_overall_college_football.csv", index=False)

    print("\nChecked sports (with results):")
    for s in checked_sports:
        print(" -", s)
    print("\nWrote csv files")
    return df

if __name__ == "__main__":
    run()
