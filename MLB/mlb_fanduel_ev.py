# mlb_ev_fanduel_et.py
import os, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  
from dotenv import load_dotenv

# ---------------------------
# Config
# ---------------------------
load_dotenv()
API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("Put THE_ODDS_API_KEY in your .env")

SPORT_KEY = "baseball_mlb"
REGIONS_CONSENSUS = "us,uk,eu"  # wide for baseball consensus
TARGET_BOOK = "fanduel"
MARKETS = ["h2h", "spreads", "totals"] # options: "h2h", "spreads", "totals"
ODDS_FORMAT = "decimal"

# Rate-limit/credit safety
MAX_EVENTS = None          # cap number of MLB games; set e.g. 50 while testing
SLEEP_BETWEEN_CALLS = 0.35
RATE_LIMIT_MIN_REMAINING = 25

# Edges
MIN_EV_PRIMARY = 0.02      # >= +2% EV
MIN_EV_FALLBACK = 0.01

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

def get_market(bookmaker, key):
    for m in bookmaker.get("markets", []):
        if m.get("key") == key:
            return m
    return None

# ---------------------------
# Consensus & EV per game/market
# ---------------------------
def consensus_rows_for_event(event, market_key, exclude_book=TARGET_BOOK):
    """
    Build consensus vs FanDuel rows for a single MLB game & market.
    Skips if FD lacks the market. Assumes 2-way for spreads/totals; h2h is 2-way in MLB.
    """
    game_label = f"{event['away_team']}@{event['home_team']}"
    start_et = parse_iso_to_et(event["commence_time"])
    # only future games (strict: must be later than 'now')
    if start_et <= now_et():
        return []

    fd = next((b for b in event.get("bookmakers", []) if b.get("key","").lower() == exclude_book), None)
    if not fd: 
        return []
    fd_mkt = get_market(fd, market_key)
    if not fd_mkt or not fd_mkt.get("outcomes"): 
        return []

    rows = []

    if market_key == "h2h":
        # moneyline 2-way
        selections = [o["name"] for o in fd_mkt["outcomes"]]
        if len(selections) != 2:
            return []

        price_table = {sel: {} for sel in selections}
        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key","").lower()
            m = get_market(bk, "h2h")
            if not m: 
                continue
            name_to_price = {o["name"]: float(o["price"]) for o in m.get("outcomes", [])}
            if any(s not in name_to_price for s in selections):
                continue
            for s in selections:
                price_table[s][bk_key] = name_to_price[s]

        books_for_consensus = sorted({bk for d in price_table.values() for bk in d if bk != exclude_book})
        fair_per_book, used_books = [], []
        for bk in books_for_consensus:
            if any(bk not in price_table[s] for s in selections):
                continue
            implied = [implied_from_decimal(price_table[s][bk]) for s in selections]
            fair = devig_proportional(implied)
            fair_per_book.append(fair); used_books.append(bk)
        if not fair_per_book:
            return []

        consensus = np.mean(np.array(fair_per_book), axis=0).tolist()

        for i, s in enumerate(selections):
            fd_dec = price_table[s].get(exclude_book)
            if not fd_dec:
                continue
            p = consensus[i]
            rows.append({
                "Sport": SPORT_KEY, "Game": game_label, "Start (ET)": start_et.strftime("%Y-%m-%d %H:%M"),
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
        partner = next((o for o in fd_mkt["outcomes"] if o.get("point")==point and o["name"]!=side), None)
        if not partner:
            continue
        other_side = partner["name"]

        fair_pairs, used_books = [], []
        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key","").lower()
            if bk_key == exclude_book:
                continue
            m = get_market(bk, market_key)
            if not m:
                continue
            name_to_price = {oo["name"]: float(oo["price"]) for oo in m.get("outcomes", []) if oo.get("point")==point}
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
        p = p_consensus_side
        label = f"{side} {point}"

        rows.append({
            "Sport": SPORT_KEY, "Game": game_label, "Start (ET)": start_et.strftime("%Y-%m-%d %H:%M"),
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
# Driver
# ---------------------------
def run():
    os.makedirs("MLB", exist_ok=True)

    all_rows = []

    # Check quota at start (optional)
    _, rem0, used0 = fetch_json("https://api.the-odds-api.com/v4/sports", {"apiKey": API_KEY})
    print(f"Start quota — used: {used0}, remaining: {rem0}")

    # Per market, fetch MLB odds
    for mk in MARKETS:
        # Stop if quota low
        _, rem, used = fetch_json("https://api.the-odds-api.com/v4/sports", {"apiKey": API_KEY})
        try:
            if int(rem or 0) < RATE_LIMIT_MIN_REMAINING:
                print("Stopping early to preserve credits.")
                break
        except:
            pass

        url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
        params = dict(apiKey=API_KEY, regions=REGIONS_CONSENSUS, markets=mk, oddsFormat=ODDS_FORMAT)
        try:
            data, rem, used = fetch_json(url, params)
        except requests.HTTPError as e:
            print(f"[skip] {SPORT_KEY}:{mk}: {e}")
            continue

        print(f"Requests used: {used}, remaining: {rem} | {SPORT_KEY}:{mk}")

        events = data[:MAX_EVENTS] if MAX_EVENTS else data
        for ev in events:
            rows = consensus_rows_for_event(ev, mk, exclude_book=TARGET_BOOK)
            all_rows.extend(rows)

        time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No future MLB edges found (FD absent, or no consensus). Try different timing.")
        return df

    # Rank & sort
    df["EV Rank (game)"] = df.groupby(["Sport","Game"])["EV per $1"].rank(ascending=False, method="dense")
    df = df.sort_values(["EV per $1","Game"], ascending=[False, True]).reset_index(drop=True)

    # Outputs
    df.to_csv("MLB/all_candidates_mlb.csv", index=False)

    singles = df[df["EV per $1"] >= MIN_EV_PRIMARY].copy()
    if singles.empty:
        singles = df[df["EV per $1"] >= MIN_EV_FALLBACK].copy()
    singles.to_csv("MLB/ready_singles_mlb.csv", index=False)

    best_overall = singles.sort_values("EV per $1", ascending=False).head(1).copy()
    best_overall.to_csv("MLB/best_overall_mlb.csv", index=False)

    print("\nWrote: MLB/all_candidates_mlb.csv, MLB/ready_singles_mlb.csv, MLB/best_overall_mlb.csv")
    return df

if __name__ == "__main__":
    run()
