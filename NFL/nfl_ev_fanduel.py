import os, time, math, requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load API_KEY from .env
load_dotenv()
API_KEY = os.getenv("THE_ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError("No API key found. Set the key in your environment variable file")

SPORT   = "americanfootball_nfl"
REGIONS = "us"                         # FanDuel + other US books
MARKETS = ["h2h", "spreads", "totals"] # moneyline, spread, total (all 2-way)
TARGET_BOOK = "fanduel"                # the book we are betting at

def american_to_decimal(a):
    return 1 + (a/100.0 if a > 0 else 100.0/abs(a))

def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))
    
def implied_from_decimal(d):
    return 1.0 / d

def devig_proportional(probs):
    s = sum(probs)
    return [p/s for p in probs] if s > 0 else probs

def ev_per_dollar(p_true, dec_odds):
    return p_true * (dec_odds - 1.0) - (1.0 - p_true)

def kelly_fraction(p_true, dec_odds):
    b = dec_odds - 1.0
    num = p_true * b - (1.0 - p_true)
    return max(0.0, num / b)

def fetch_odds(market_key):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = dict(apiKey=API_KEY, regions=REGIONS, markets=market_key, oddsFormat="decimal")
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining"), resp.headers.get("x-requests-used")

def consensus_rows_for_event(event, market_key, exclude_book="fanduel"):
    """
    event: one game JSON with many bookmakers
    market_key: 'h2h' | 'spreads' | 'totals'
    Returns rows (one per selection) with consensus P and EV/Kelly vs FanDuel.
    """

    game_label = f"{event['away_team']}@{event['home_team']}"
    commence = event["commence_time"]

    # Helper: get a bookmaker's market dict (key -> list of outcomes)
    def get_market(bookmaker, key):
        for m in bookmaker.get("markets", []):
            if m.get("key") == key:
                return m
        return None

    # Find FanDuel first; if missing, nothing to compare
    fd = next((b for b in event.get("bookmakers", []) if b.get("key","").lower() == exclude_book), None)
    if not fd:
        return []

    fd_mkt = get_market(fd, market_key)
    if not fd_mkt or not fd_mkt.get("outcomes"):
        return []

    rows = []

    if market_key == "h2h":
        # Build price table across books for the two h2h outcomes
        selections = [o["name"] for o in fd_mkt["outcomes"]]
        # Collect per-book decimal odds for exactly these selections
        price_table = {sel: {} for sel in selections}
        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key","").lower()
            m = get_market(bk, "h2h")
            if not m: 
                continue
            # map name -> price
            name_to_price = {o["name"]: o["price"] for o in m.get("outcomes", [])}
            # require both selections
            if any(s not in name_to_price for s in selections): 
                continue
            for s in selections:
                price_table[s][bk_key] = name_to_price[s]

        # de-vig per non-FD book, average to consensus
        books_for_consensus = sorted({bk for d in price_table.values() for bk in d if bk != exclude_book})
        fair_per_book, used_books = [], []
        for bk in books_for_consensus:
            implied = [1.0/price_table[s][bk] for s in selections] if all(bk in price_table[s] for s in selections) else None
            if not implied: 
                continue
            fair = devig_proportional(implied)
            fair_per_book.append(fair); used_books.append(bk)
        if not fair_per_book:
            return []
        consensus = np.mean(np.array(fair_per_book), axis=0).tolist()

        # rows vs FanDuel
        for i, s in enumerate(selections):
            fd_dec = price_table[s].get(exclude_book)
            if not fd_dec: 
                continue
            p = consensus[i]
            rows.append({
                "Game": game_label, "Start (UTC)": commence, "Market": market_key, "Selection": s,
                "Consensus P": round(p,4), "Fair Odds": round(1.0/p,3),
                "FanDuel (dec)": round(fd_dec,3), "EV per $1": round(ev_per_dollar(p, fd_dec),4),
                "Kelly": round(kelly_fraction(p, fd_dec),4),
                "Consensus Books Used": ", ".join(used_books)
            })
        return rows
    
    else:
        # --- Exact-match path for spreads/totals ---
        # Build FanDuel's set of (side, point) we want to price
        fd_outcomes = []
        for o in fd_mkt.get("outcomes", []):
            side = o["name"]
            point = o.get("point")
            if point is None:  # spreads/totals should have points; if not, skip
                continue
            fd_outcomes.append((side, point, o["price"]))

        # For each FanDuel (side,point), collect matching books
        for side, point, fd_dec in fd_outcomes:
            # Build per-book prices for the TWO sides at this SAME point
            # First, determine the "other side" label for the same point
            # For spreads: if side is Team A -3.5, other is Team B +3.5
            # For totals: if side is Over 47.5, other is Under 47.5
            # We'll discover both by scanning FanDuel outcomes at that point:
            partner = next((o for o in fd_mkt["outcomes"] 
                            if o.get("point") == point and o["name"] != side), None)
            if not partner:
                continue
            other_side = partner["name"]

            # Collect matching books that post BOTH sides at the SAME point
            book_pairs = []   # list of [implied_side, implied_other]
            used_books = []
            for bk in event.get("bookmakers", []):
                bk_key = bk.get("key","").lower()
                if bk_key == exclude_book:
                    continue
                m = get_market(bk, market_key)
                if not m: 
                    continue
                # Find both outcomes with the exact same point
                name_to_outcomes = {}
                for oo in m.get("outcomes", []):
                    if oo.get("point") == point:
                        name_to_outcomes[oo["name"]] = oo["price"]
                if side in name_to_outcomes and other_side in name_to_outcomes:
                    d_side = name_to_outcomes[side]
                    d_other = name_to_outcomes[other_side]
                    book_pairs.append([1.0/d_side, 1.0/d_other])
                    used_books.append(bk_key)

            if not book_pairs:
                continue

            fair = []
            # de-vig per book pair then average the first component (this side)
            fair_pairs = []
            for imp_side, imp_other in book_pairs:
                s = imp_side + imp_other
                if s <= 0: 
                    continue
                fair_pairs.append([imp_side/s, imp_other/s])
            if not fair_pairs:
                continue
            fair_arr = np.array(fair_pairs)
            p_consensus_side = float(fair_arr[:,0].mean())

            # Build display label "Side point"
            label = f"{side} {point}"
            p = p_consensus_side
            rows.append({
                "Game": game_label, "Start (UTC)": commence, "Market": market_key, "Selection": label,
                "Consensus P": round(p,4), "Win Prob (Consensus)": round(p * 100, 1), "Fair Odds": round(1.0/p,3),
                "FanDuel (dec)": round(fd_dec,3), "EV per $1": round(ev_per_dollar(p, fd_dec),4),
                "Kelly": round(kelly_fraction(p, fd_dec),4),
                "Consensus Books Used": ", ".join(sorted(set(used_books)))
            })
        return rows    

def run():
    all_rows = []
    for mk in MARKETS:
        data, rem, used = fetch_odds(mk)
        print(f"Number of requests used: {used}, Number of requests remaining: {rem}")
        for ev in data:
            all_rows += consensus_rows_for_event(ev, mk, exclude_book=TARGET_BOOK)
        time.sleep(0.35) # Don't overwhelm the API
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows produced. Check API key/plan or try again later.")
        return df
    # Rank by EV within each game
    df["EV Rank (game)"] = df.groupby("Game")["EV per $1"].rank(ascending=False, method="dense")
    df = df.sort_values(["Game", "EV per $1"], ascending=[True, False]).reset_index(drop=True)
    # (Optional) keep only meaningful edges:
    # df = df[(df["EV per $1"] >= 0.02) & (df["Kelly"] > 0)]

    MIN_EV_PRIMARY = 0.02   # require >= +2% EV
    MIN_EV_FALLBACK = 0.01  # if too few picks, allow >= +1% EV
    MIN_KELLY = 0.0         # must be > 0 (leave as 0.0)
    MAX_SINGLES = 20        # cap output size for readability

    # 1) filter to edges
    edges = df[(df["EV per $1"] >= MIN_EV_PRIMARY) & (df["Kelly"] > MIN_KELLY)].copy()
    if edges.empty:
        edges = df[(df["EV per $1"] >= MIN_EV_FALLBACK) & (df["Kelly"] > MIN_KELLY)].copy()

    # 2) best pick per game (highest EV per game)
    edges["PickRankInGame"] = edges.groupby("Game")["EV per $1"].rank(ascending=False, method="first")
    best_per_game = edges[edges["PickRankInGame"] == 1].copy()

    # 3) tidy columns for singles
    singles_cols = ["Game","Market","Selection","FanDuel (dec)","Consensus P","Fair Odds","EV per $1","Kelly","Start (UTC)"]
    for c in singles_cols:
        if c not in best_per_game.columns:
            best_per_game[c] = np.nan

    # add American odds for readability
    best_per_game["FanDuel (amer)"] = best_per_game["FanDuel (dec)"].apply(decimal_to_american)
    best_per_game = best_per_game.sort_values(["EV per $1","Game"], ascending=[False, True]).head(MAX_SINGLES)

    # 4) write singles slip
    singles_out = best_per_game[["Game","Market","Selection","Start (UTC)","FanDuel (dec)","FanDuel (amer)","Consensus P","Fair Odds","EV per $1","Kelly"]]
    singles_out.to_csv("NFL/ready_singles.csv", index=False)

    # 5) build parlays from the top N best_per_game legs (independence assumption!)
    def parlay_eval(rows: pd.DataFrame):
        if len(rows) < 2:
            return np.nan, np.nan, np.nan
        dec_prod = float(np.prod(rows["FanDuel (dec)"].astype(float)))
        p_prod = float(np.prod(rows["Consensus P"].astype(float)))
        ev = p_prod * (dec_prod - 1.0) - (1.0 - p_prod)
        return round(dec_prod, 3), round(p_prod, 4), round(ev, 4)

    top = best_per_game.copy().reset_index(drop=True)
    p2 = top.head(2);  p3 = top.head(3);  p4 = top.head(4)

    def legs_to_str(dflegs):
        return " • ".join([f"{r.Market}: {r.Selection} ({r.Game})" for _, r in dflegs.iterrows()])

    parlays_rows = []
    for label, legs in [("Conservative 2-leg", p2), ("Balanced 3-leg", p3), ("Aggressive 4-leg", p4)]:
        dec_odds, p_win, ev = parlay_eval(legs)
        parlays_rows.append({
            "Parlay": label,
            "Legs": len(legs),
            "Suggestion": legs_to_str(legs) if len(legs) >= 2 else "Not enough qualifying legs",
            "Parlay (dec)": dec_odds,
            "Consensus P (indep)": p_win,
            "EV per $1": ev
        })

    parlays_out = pd.DataFrame(parlays_rows)
    parlays_out.to_csv("NFL/ready_parlays.csv", index=False)

    print("\nWrote ready_singles.csv and ready_parlays.csv")

    df.to_csv("NFL/nfl_ev_fanduel.csv", index=False)
    #print(df.head(40).to_string(index=False))
    print("\nWrote nfl_ev_fanduel.csv")
    return df

if __name__ == "__main__":
    run()