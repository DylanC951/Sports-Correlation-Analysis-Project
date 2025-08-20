# backtest_december.py
import os, time, math, argparse, datetime as dt
import requests, pandas as pd
from pandas import Timestamp
from collections import defaultdict, Counter
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog
from tqdm import tqdm

from analysis import ParlayAnalyzer
from data_processing import LiveLineFetcher, clean_gamelog

# ---- config
MARKETS = ",".join([
    "player_points","player_rebounds","player_assists",
    "player_points_rebounds_assists","player_threes",
    "player_blocks_steals","player_turnovers"
])
STAT_MAP = {
    "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
    "player_points_rebounds_assists":"PRA","player_threes":"FG3M",
    "player_blocks_steals":"STOCKS","player_turnovers":"TOV"
}

def parse_utc(ts) -> Timestamp:
    return pd.to_datetime(ts, utc=True)

def dates_in_december_2024():
    d0, d1 = dt.date(2024,12,1), dt.date(2024,12,31)
    d = d0
    while d <= d1:
        yield d
        d += dt.timedelta(days=1)

# --- NBA helpers & caching
_all = nba_players.get_players()
_name2id = {p["full_name"].lower(): p["id"] for p in _all}

def find_player_id(name: str):
    key = name.strip().lower()
    if key in _name2id: return _name2id[key]
    # fallback: loose match
    from difflib import get_close_matches
    m = get_close_matches(key, _name2id.keys(), n=1, cutoff=0.7)
    return _name2id[m[0]] if m else None

_gamelog_cache = {}
def get_gamelog(player_id: int):
    if player_id in _gamelog_cache: return _gamelog_cache[player_id]
    gl = playergamelog.PlayerGameLog(player_id=player_id, season="2024-25",
                                     season_type_all_star="Regular Season").get_data_frames()[0]
    df = clean_gamelog(gl)
    _gamelog_cache[player_id] = df
    return df

def get_stat_on_date(player_id: int, stat: str, date_: dt.date):
    df = get_gamelog(player_id)
    r = df[df["GAME_DATE"] == date_]
    if r.empty: return None  # DNP
    return float(r.iloc[0][stat])

# Parse DFS props (per event)
def parse_event_props(data, platform: str):
    """
    returns:
      players: {normalized_name: set({STAT})}
      lines:   {(normalized_name, STAT): line_float}
    """
    players = defaultdict(set)
    lines = {}
    if not data or "bookmakers" not in data: return players, lines
    # choose the bookmaker of interest for this platform if present
    wanted = "prizepicks" if platform=="prizepicks" else "underdog"
    bks = [b for b in data["bookmakers"] if b["key"] == wanted] or data["bookmakers"]
    for bk in bks:
        for m in bk.get("markets", []):
            mkey = m["key"]
            if mkey not in STAT_MAP: continue
            stat = STAT_MAP[mkey]
            for out in m.get("outcomes", []):
                # Both Over and Under carry the same point line; description is player name
                nm = LiveLineFetcher.normalize_name(out.get("description",""))
                if not nm: continue
                pt = out.get("point", None)
                if pt is None: continue
                players[nm].add(stat)
                # store one line per (player, stat)
                lines[(nm, stat)] = float(pt)
    return players, lines

# payout calculators
def prizepicks_6flex(mult_ok, mult_5, mult_4):
    return {6:25.0, 5:2.0, 4:0.4}
def underdog_6flex():
    return {6:25.0, 5:2.6, 4:0.25}

def grade_entry(picks, platform: str):
    """
    picks: list of dicts:
      {"name":..., "stat":..., "dir": "OVER"/"UNDER", "line": float, "actual": float, "push": bool}
    returns (hits, pushes, paid_multiple)
    """
    hits = sum(1 for p in picks if (not p["push"]) and ((p["actual"] > p["line"]) if p["dir"]=="OVER" else (p["actual"] < p["line"])))
    pushes = sum(1 for p in picks if p["push"])
    legs = len(picks) - pushes
    # downgrade flex on pushes
    if platform=="underdog":
        table = underdog_6flex()
    else:
        table = prizepicks_6flex(25.0,2.0,0.4)
    # with pushes, legs reduce:
    target = legs
    # payout only defined for legs>=4 for 6-pick flex logic
    if target < 4: return hits, pushes, 0.0
    if hits >= target: return hits, pushes, table.get(target, 0.0)
    if hits == target-1: return hits, pushes, table.get(5 if target==6 else target-1, 0.0)
    if hits == target-2 and target==6: return hits, pushes, table.get(4, 0.0)
    return hits, pushes, 0.0

def build_three_entries_for_day(corr_df, game_players, lines_by_game, date_, platform):
    """
    Strategy: within each event, rank pairs by |corr| then draft top pairs greedily,
    ensuring no duplicate (player,stat) in an entry. Build up to 3 entries of 3 pairs each.
    """
    entries = []
    used_pairs_global = set()
    # sort by abs corr descending
    corr_df = corr_df.sort_values("correlation", ascending=False)
    # group by event for same-game constraint
    for event_id, sub in corr_df.groupby("event_id"):
        for _ in range(3):  # attempt to pull multiple pairs from same event
            chosen = []
            seen_players = set()
            for _, row in sub.iterrows():
                p1, p2, stat = row["player1"], row["player2"], row["stat"]
                if (p1,p2,stat) in used_pairs_global: continue
                if p1 in seen_players or p2 in seen_players: continue
                # both lines must exist on this platform snapshot
                L = lines_by_game[event_id]
                if (p1,stat) not in L or (p2,stat) not in L: continue
                chosen.append((p1,p2,stat,row["correlation"]))
                seen_players.update([p1,p2])
                if len(chosen)==3: break
            if len(chosen)==3:
                used_pairs_global.update((p1,p2,stat) for p1,p2,stat,_ in chosen)
                # convert to six picks, dir by corr sign
                six = []
                for p1,p2,stat,c in chosen:
                    d = "OVER" if c>0 else "UNDER"
                    six.append({"name":p1,"stat":stat,"dir":d,"line":lines_by_game[event_id][(p1,stat)]})
                    six.append({"name":p2,"stat":stat,"dir":d,"line":lines_by_game[event_id][(p2,stat)]})
                entries.append({"event_id": event_id, "date": date_, "platform": platform, "legs": six})
                if len(entries) == 3: return entries
    return entries

def main(platform: str, stake: float, bankroll: float):
    fetch = LiveLineFetcher(os.getenv("ODDS_API_KEY"))
    analyzer = ParlayAnalyzer(min_games=10, min_corr=0.35)

    results = []
    bal = bankroll

    for date_ in tqdm(list(dates_in_december_2024()), desc="Days"):
        # 1) list events for the day (use noon UTC snapshot to get the day's slate)
        iso_day = dt.datetime.combine(date_, dt.time(12,0)).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = fetch.historical_events(iso_day)
        # filter to games actually on that calendar day:
        todays = []
        for ev in events:
            ct = parse_utc(ev["commence_time"]).date()
            if ct == date_:
                todays.append(ev)

        if not todays: continue

        # 2) pull props for each event ~30 minutes before tip
        game_players = {}            # event_id -> { name -> set(stats) }
        lines_by_game = {}           # event_id -> { (name,stat) -> line }
        for ev in todays:
            ev_id = ev["id"]
            tip = parse_utc(ev["commence_time"])
            snap = (tip - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            data = fetch.historical_event_props(ev_id, snap, MARKETS)
            players_offered, lines = parse_event_props(data, platform)
            if players_offered:
                game_players[ev_id] = players_offered
                lines_by_game[ev_id] = lines

        if not game_players: continue

        # 3) load NBA logs for all players who actually had lines that day
        all_names = sorted({nm for roster in game_players.values() for nm in roster})
        player_data = {}
        for nm in all_names:
            pid = find_player_id(nm)
            if not pid: continue
            try:
                player_data[nm] = get_gamelog(pid)
            except Exception:
                pass

        # 4) correlations up to (not including) this date
        corr_df = analyzer.analyze_same_game_pairs(game_players, player_data, date_)

        if corr_df.empty: continue

        # 5) build 3 six-leg entries
        entries = build_three_entries_for_day(corr_df, game_players, lines_by_game, date_, platform)

        # 6) grade entries using actual stats for that date
        for ent in entries:
            for leg in ent["legs"]:
                pid = find_player_id(leg["name"])
                actual = get_stat_on_date(pid, leg["stat"], date_) if pid else None
                if actual is None:
                    leg.update({"actual": leg["line"], "push": True})
                else:
                    leg.update({"actual": actual, "push": abs(actual - leg["line"]) < 1e-9})
            hits, pushes, mult = grade_entry(ent["legs"], platform)
            pnl = stake*(mult - 1.0)  # net profit including stake (mult=0 -> -stake)
            bal += pnl
            results.append({
                "date": ent["date"], "platform": platform,
                "hits": hits, "pushes": pushes, "mult": mult,
                "stake": stake, "pnl": pnl, "bankroll": bal
            })

    df = pd.DataFrame(results).sort_values("date")
    df.to_csv(f"dec_2024_{platform}_backtest.csv", index=False)
    print(df.tail(10))
    print("\nFinal bankroll:", bal, "  P&L:", bal - bankroll)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["prizepicks","underdog"], default="prizepicks")
    ap.add_argument("--stake", type=float, default=100.0)
    ap.add_argument("--bankroll", type=float, default=10_000.0)
    args = ap.parse_args()
    main(args.platform, args.stake, args.bankroll)
