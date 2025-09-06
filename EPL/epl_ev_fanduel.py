# epl_ev_fanduel.py
import os, time, requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------
# Config
# ---------------------------
load_dotenv()
API_KEY = os.getenv("THE_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Put THE_ODDS_API_KEY in your .env")

SPORT   = "soccer_epl"          # Premier League
REGIONS = "uk,eu,us"            # pull a broad consensus (exclude FD later)
MARKETS = ["h2h", "spreads", "totals"]
TARGET_BOOK = "fanduel"

# ---------------------------
# Helpers
# ---------------------------
def implied_from_decimal(d: float) -> float:
    return 1.0 / float(d)

def devig_proportional(probs):
    """Proportional de-vig for N outcomes: rescale to sum to 1."""
    s = float(sum(probs))
    if s <= 0:
        return probs
    return [p / s for p in probs]

def ev_per_dollar(p_true: float, dec_odds: float) -> float:
    return p_true * (dec_odds - 1.0) - (1.0 - p_true)

def kelly_fraction(p_true: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    num = p_true * b - (1.0 - p_true)
    return max(0.0, num / b) if b > 0 else 0.0

def fetch_odds(market_key):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = dict(apiKey=API_KEY, regions=REGIONS, markets=market_key, oddsFormat="decimal")
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def get_market(bookmaker, key):
    for m in bookmaker.get("markets", []):
        if m.get("key") == key:
            return m
    return None

# ---------------------------
# Core: EPL consensus & EV vs FanDuel
# ---------------------------
def consensus_rows_for_event(event, market_key, exclude_book=TARGET_BOOK):
    game_label = f"{event['away_team']}@{event['home_team']}"
    commence = event["commence_time"]

    # Find FanDuel block/market (we price only what FD actually offers)
    fd = next((b for b in event.get("bookmakers", []) if b.get("key","").lower() == exclude_book), None)
    if not fd:
        return []
    fd_mkt = get_market(fd, market_key)
    if not fd_mkt or not fd_mkt.get("outcomes"):
        return []

    rows = []

    if market_key == "h2h":
        # ---- 3-way moneyline (Home / Draw / Away) for soccer ----
        # Selections (labels) come from FanDuel to fix the order
        selections = [o["name"] for o in fd_mkt["outcomes"]]
        # Build price table only for these selections
        price_table = {sel: {} for sel in selections}

        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key","").lower()
            m = get_market(bk, "h2h")
            if not m:
                continue
            name_to_price = {o["name"]: o["price"] for o in m.get("outcomes", [])}
            # require all selections (3 outcomes)
            if any(s not in name_to_price for s in selections):
                continue
            for s in selections:
                price_table[s][bk_key] = name_to_price[s]

        # Build per-book vig-free vectors, excluding FanDuel
        books_for_consensus = sorted({bk for d in price_table.values() for bk in d if bk != exclude_book})
        fair_per_book, used_books = [], []
        for bk in books_for_consensus:
            # gather bk's prices in selections order
            if any(bk not in price_table[s] for s in selections):
                continue
            implied = [implied_from_decimal(price_table[s][bk]) for s in selections]
            fair = devig_proportional(implied)  # N-way normalize
            fair_per_book.append(fair); used_books.append(bk)
        if not fair_per_book:
            return []

        consensus = np.mean(np.array(fair_per_book), axis=0).tolist()

        # Rows vs FanDuel per outcome
        for i, s in enumerate(selections):
            fd_dec = price_table[s].get(exclude_book)
            if not fd_dec:
                continue
            p = consensus[i]
            rows.append({
                "Game": game_label, "Start (UTC)": commence, "Market": market_key, "Selection": s,
                "Consensus P": round(p,4), "Win Prob (Consensus)": round(p * 100, 1), "Fair Odds": round(1.0/p,3),
                "FanDuel (dec)": round(float(fd_dec),3),
                "EV per $1": round(ev_per_dollar(p, float(fd_dec)),4),
                "Kelly": round(kelly_fraction(p, float(fd_dec)),4),
                "Consensus Books Used": ", ".join(used_books)
            })
        return rows

    else:
        # ---- Exact point-match for spreads/totals (2-way) ----
        # Build FanDuel's set of (side, point)
        fd_outcomes = []
        for o in fd_mkt.get("outcomes", []):
            side = o["name"]          # e.g., "Over" / "Under" OR team name for handicap
            point = o.get("point")    # goal line / handicap value
            if point is None:
                continue
            fd_outcomes.append((side, point, float(o["price"])))

        for side, point, fd_dec in fd_outcomes:
            # Find the partner side at the same point (e.g., Over ↔ Under 2.5)
            partner = next((o for o in fd_mkt["outcomes"] if o.get("point")==point and o["name"]!=side), None)
            if not partner:
                continue
            other_side = partner["name"]

            # Collect matching books with BOTH sides at the SAME point
            book_pairs, used_books = [], []
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
                    # proportional de-vig for this 2-way pair
                    ssum = imp_side + imp_other
                    if ssum <= 0:
                        continue
                    fair_side = imp_side / ssum
                    fair_other = imp_other / ssum
                    book_pairs.append([fair_side, fair_other])
                    used_books.append(bk_key)

            if not book_pairs:
                continue

            fair_arr = np.array(book_pairs)
            p_consensus_side = float(fair_arr[:,0].mean())

            label = f"{side} {point}"
            p = p_consensus_side
            rows.append({
                "Game": game_label, "Start (UTC)": commence, "Market": market_key, "Selection": label,
                "Consensus P": round(p,4), "Fair Odds": round(1.0/p,3),
                "FanDuel (dec)": round(fd_dec,3), "EV per $1": round(ev_per_dollar(p, fd_dec),4),
                "Kelly": round(kelly_fraction(p, fd_dec),4),
                "Consensus Books Used": ", ".join(sorted(set(used_books)))
            })
        return rows

# ---------------------------
# Driver
# ---------------------------
def run():
    all_rows = []
    for mk in MARKETS:
        data = fetch_odds(mk)
        for ev in data:
            all_rows += consensus_rows_for_event(ev, mk, exclude_book=TARGET_BOOK)
        time.sleep(0.35)  # politeness/rate-limit

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows produced. Try later or check market coverage/regions.")
        return df

    # Rank by EV within match
    df["EV Rank (game)"] = df.groupby("Game")["EV per $1"].rank(ascending=False, method="dense")
    df = df.sort_values(["Game","EV per $1"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv("EPL/epl_ev_fanduel.csv", index=False)
    print(df.head(40).to_string(index=False))
    print("\nWrote epl_ev_fanduel.csv")

    # ---- Best Bet of the Week (same logic you liked) ----
    MIN_EV_BEST = 0.02
    eligible = df[(df["EV per $1"] >= MIN_EV_BEST) & (df["Kelly"] > 0)].copy()
    if eligible.empty:
        eligible = df[(df["EV per $1"] >= 0.01) & (df["Kelly"] > 0)].copy()
    if eligible.empty:
        print("No Best Bet (threshold not met).")
        return df

    best = eligible.sort_values("EV per $1", ascending=False).head(1).copy()
    best_cols = ["Game","Market","Selection","Start (UTC)","Consensus P","Fair Odds","FanDuel (dec)","EV per $1","Kelly"]
    best[best_cols].to_csv("epl_best_bet.csv", index=False)
    row = best.iloc[0]
    print("\n⭐ EPL Best Bet ⭐")
    print(f"{row['Market']}: {row['Selection']} ({row['Game']})")
    print(f"Start: {row['Start (UTC)']}")
    print(f"Consensus P: {row['Consensus P']:.3f} | Fair Odds: {row['Fair Odds']:.2f}")
    print(f"FanDuel: {row['FanDuel (dec)']:.2f}")
    print(f"EV per $1: {row['EV per $1']:.3f} | Kelly: {row['Kelly']:.3f}")
    print("Wrote epl_best_bet.csv")
    return df

if __name__ == "__main__":
    run()
