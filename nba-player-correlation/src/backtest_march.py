# backtest_january.py
import os, argparse, datetime as dt
import requests, pandas as pd
from pandas import Timestamp
from collections import defaultdict
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

def dates_in_january_2025():
    d0, d1 = dt.date(2025,3,1), dt.date(2025,3,31)
    d = d0
    while d <= d1:
        yield d
        d += dt.timedelta(days=1)

# --- NBA helpers & caching
_all = nba_players.get_players()
_name2id = {p["full_name"].lower(): p["id"] for p in _all}

def find_player_id(name: str):
    key = name.strip().lower()
    if key in _name2id:
        return _name2id[key]
    # fallback: loose match
    from difflib import get_close_matches
    m = get_close_matches(key, _name2id.keys(), n=1, cutoff=0.7)
    return _name2id[m[0]] if m else None

_gamelog_cache = {}
def get_gamelog(player_id: int):
    if player_id in _gamelog_cache:
        return _gamelog_cache[player_id]
    gl = playergamelog.PlayerGameLog(
        player_id=player_id,
        season="2024-25",
        season_type_all_star="Regular Season"
    ).get_data_frames()[0]
    df = clean_gamelog(gl)
    _gamelog_cache[player_id] = df
    return df

def get_stat_on_date(player_id: int, stat: str, date_: dt.date):
    df = get_gamelog(player_id)
    r = df[df["GAME_DATE"] == date_]
    if r.empty:
        return None  # DNP / no game that day
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
    if not data or "bookmakers" not in data:
        return players, lines
    # choose the bookmaker of interest for this platform if present
    wanted = "prizepicks" if platform == "prizepicks" else "underdog"
    bks = [b for b in data["bookmakers"] if b["key"] == wanted] or data["bookmakers"]
    for bk in bks:
        for m in bk.get("markets", []):
            mkey = m["key"]
            if mkey not in STAT_MAP:
                continue
            stat = STAT_MAP[mkey]
            for out in m.get("outcomes", []):
                # Both Over and Under carry the same point line; description is player name
                nm = LiveLineFetcher.normalize_name(out.get("description", ""))
                if not nm:
                    continue
                pt = out.get("point", None)
                if pt is None:
                    continue
                players[nm].add(stat)
                # store one line per (player, stat)
                lines[(nm, stat)] = float(pt)
    return players, lines

# payout calculators
def prizepicks_6flex():
    return {6: 37.5, 5: 0.0, 4: 0.0}

def underdog_6flex():
    return {6: 37.5, 5: 0.0, 4: 0.0}

def grade_entry(picks, platform: str):
    """
    picks: list of dicts:
      {"name":..., "stat":..., "dir": "OVER"/"UNDER", "line": float, "actual": float, "push": bool}
    returns (hits, pushes, paid_multiple)
    """
    hits = sum(
        1 for p in picks
        if (not p["push"]) and ((p["actual"] > p["line"]) if p["dir"] == "OVER" else (p["actual"] < p["line"]))
    )
    pushes = sum(1 for p in picks if p["push"])
    legs = len(picks) - pushes
    # downgrade flex on pushes
    table = underdog_6flex() if platform == "underdog" else prizepicks_6flex()
    target = legs
    # payout only defined for legs >= 4 in a 6-pick flex context
    if target < 4:
        return hits, pushes, 0.0
    if hits >= target:
        return hits, pushes, table.get(target, 0.0)
    if hits == target - 1:
        return hits, pushes, table.get(5 if target == 6 else target - 1, 0.0)
    if hits == target - 2 and target == 6:
        return hits, pushes, table.get(4, 0.0)
    return hits, pushes, 0.0

def leg_result(leg):
    if leg.get("push"):
        return "PUSH"
    hit = (leg["actual"] > leg["line"]) if leg["dir"] == "OVER" else (leg["actual"] < leg["line"])
    return "HIT" if hit else "MISS"

# -------------------- NEW: robust odds fetch with fallbacks --------------------
def fetch_event_props_with_fallback(fetcher: LiveLineFetcher, ev_id: str, tip: Timestamp):
    """
    Try several nearby snapshots around tipoff to reduce 404s from the historical odds endpoint.
    Returns (data, used_snapshot_iso) or (None, None).
    """
    candidates = [
        tip - pd.Timedelta(minutes=30),
        tip - pd.Timedelta(minutes=20),
        tip - pd.Timedelta(minutes=10),
        tip - pd.Timedelta(minutes=5),
        tip,  # at tip
        tip - pd.Timedelta(minutes=60),
    ]
    for ts in candidates:
        snap = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            data = fetcher.historical_event_props(ev_id, snap, MARKETS)
            # data comes already unwrapped (bookmakers at top-level)
            if data and isinstance(data, dict) and data.get("bookmakers"):
                return data, snap
        except requests.HTTPError as e:
            # 404/400 -> just try next snapshot; re-raise other codes
            code = getattr(e.response, "status_code", None)
            if code in (404, 400):
                continue
            raise
        except Exception:
            # network or JSON hiccup, try next
            continue
    return None, None

def _pra_conflict_for_player(prev_stats: set, new_stat: str) -> bool:
    """
    Enforce per-player PRA overlap across DIFFERENT slips (same event/day):
      - If player used with PTS/REB/AST in one slip, they can't be used with PRA in another slip.
      - If player used with PRA in one slip, they can't be used with PTS/REB/AST in another slip.
    """
    core = {"PTS", "REB", "AST"}
    if new_stat == "PRA" and (prev_stats & core):
        return True
    if new_stat in core and "PRA" in prev_stats:
        return True
    return False


def build_three_entries_for_day_any_market(corr_df, game_players, lines_by_game, date_, platform):
    """
    Build up to 3 six-leg entries (3 pairs per entry) for the day.

    Constraints across different entries of the SAME event (game):
      1) Do NOT re-use the exact (p1,stat1)-(p2,stat2) market combo for the same unordered pair {p1,p2}.
      2) Per-player PRA overlap rule (see _pra_conflict_for_player).

    Within a single entry we still avoid re-using a player (seen_players), so
    the new PRA rule only affects later entries for the same event.
    """
    entries = []
    used_pairs_global = set()  # exact quadruples (p1,p2,stat1,stat2) to avoid literal repeats

    # (1) Pair+market de-dup across entries:
    #     event_id -> set(frozenset({(p1, s1), (p2, s2)}))
    used_market_combos = defaultdict(set)

    # (2) Per-player stats used in prior entries (to enforce PRA overlap):
    #     event_id -> player -> set(stats_used)
    used_stats_by_player = defaultdict(lambda: defaultdict(set))

    # strongest first
    corr_df = (
        corr_df.assign(abs_corr=corr_df["correlation"].abs())
               .sort_values(["abs_corr", "p_value", "n_games"], ascending=[False, True, False])
               .drop(columns="abs_corr")
    )

    for event_id, sub in corr_df.groupby("event_id"):
        if len(entries) >= 3:
            break

        while len(entries) < 3:
            chosen = []
            seen_players = set()

            for _, row in sub.iterrows():
                p1, p2 = row["player1"], row["player2"]
                stat1, stat2 = row["stat1"], row["stat2"]

                # avoid exact row re-use and player duplication within the same entry
                if (p1, p2, stat1, stat2) in used_pairs_global:
                    continue
                if p1 in seen_players or p2 in seen_players:
                    continue

                # both legs must have lines on this snapshot
                L = lines_by_game.get(event_id, {})
                if (p1, stat1) not in L or (p2, stat2) not in L:
                    continue

                # (1) exact pair+market combo already used in another entry?
                combo_key = frozenset(((p1, stat1), (p2, stat2)))  # order-insensitive
                if combo_key in used_market_combos[event_id]:
                    continue

                # (2) per-player PRA overlap check vs stats from PRIOR entries only
                if _pra_conflict_for_player(used_stats_by_player[event_id][p1], stat1):
                    continue
                if _pra_conflict_for_player(used_stats_by_player[event_id][p2], stat2):
                    continue

                chosen.append((p1, p2, stat1, stat2, row["correlation"]))
                seen_players.update([p1, p2])
                if len(chosen) == 3:
                    break

            if len(chosen) < 3:
                # can't assemble another 3-pair entry from this game's pool
                break

            # bake an entry of 6 legs; direction by corr sign
            six_legs = []
            for idx, (p1, p2, s1, s2, c) in enumerate(chosen, 1):
                direction = "OVER" if c > 0 else "UNDER"
                L = lines_by_game[event_id]
                six_legs.append({
                    "name": p1, "stat": s1, "dir": direction,
                    "line": L[(p1, s1)],
                    "pair_idx": idx,
                })
                six_legs.append({
                    "name": p2, "stat": s2, "dir": direction,
                    "line": L[(p2, s2)],
                    "pair_idx": idx,
                })

                # mark exact usage and pair+market combo for cross-entry constraints
                used_pairs_global.add((p1, p2, s1, s2))
                used_market_combos[event_id].add(frozenset(((p1, s1), (p2, s2))))

            # after we finalize *this* entry, update per-player used stats
            # so the PRA overlap rule only affects subsequent entries (as intended)
            for (p1, p2, s1, s2, _) in chosen:
                used_stats_by_player[event_id][p1].add(s1)
                used_stats_by_player[event_id][p2].add(s2)

            entries.append({
                "event_id": event_id,
                "date": date_,
                "platform": platform,
                "legs": six_legs
            })

            if len(entries) >= 3:
                break

    return entries


# -------------------- NEW: evaluate pair outcomes (success/fail/push) --------------------
def evaluate_pair_outcomes(legs):
    """
    legs carry 'pair_id' (1..3). For each pair:
      - SUCCESS if both non-push and both HIT or both MISS
      - FAIL    if both non-push and one HIT / one MISS
      - PUSH    if either leg is a push
    Returns (success_count, fail_count, push_count).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for lg in legs:
        pid = lg.get("pair_id")
        if pid is not None:
            groups[pid].append(lg)

    succ = fail = push = 0
    for pid, items in groups.items():
        if len(items) != 2:
            continue  # safety
        r1, r2 = leg_result(items[0]), leg_result(items[1])
        if "PUSH" in (r1, r2):
            push += 1
        elif (r1 == "HIT" and r2 == "HIT") or (r1 == "MISS" and r2 == "MISS"):
            succ += 1
        else:
            fail += 1
    return succ, fail, push

def main(platform: str, stake: float, bankroll: float):
    # You may also pass the key into LiveLineFetcher(...) directly if preferred.
    fetch = LiveLineFetcher(os.getenv("ODDS_API_KEY"))
    analyzer = ParlayAnalyzer(min_games=15, min_corr=0.35)

    results = []       # per-entry summary rows
    legs_details = []  # per-leg detailed rows
    bal = bankroll
    entry_counter = 0

    # global totals for pair outcomes
    total_pair_success = 0
    total_pair_fail = 0
    total_pair_push = 0

    for date_ in tqdm(list(dates_in_january_2025()), desc="Days"):
        # 1) list events for the day (use noon UTC snapshot to get the day's slate)
        iso_day = dt.datetime.combine(date_, dt.time(12,0)).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = fetch.historical_events(iso_day)
        # filter to games actually on that calendar day:
        todays = []
        for ev in events:
            ct = parse_utc(ev["commence_time"]).date()
            if ct == date_:
                todays.append(ev)

        if not todays:
            continue

        # 2) pull props for each event using robust fallback snapshots
        game_players = {}            # event_id -> { name -> set(stats) }
        lines_by_game = {}           # event_id -> { (name,stat) -> line }
        for ev in todays:
            ev_id = ev["id"]
            tip = parse_utc(ev["commence_time"])

            data, used_snap = fetch_event_props_with_fallback(fetch, ev_id, tip)
            if not data:
                # No odds snapshot near tip; skip this game
                print(f"[WARN] No event odds for {ev_id} around tip {tip} — skipping game.")
                continue

            players_offered, lines = parse_event_props(data, platform)
            if players_offered:
                game_players[ev_id] = players_offered
                lines_by_game[ev_id] = lines

        if not game_players:
            continue

        # 3) load NBA logs for all players who actually had lines that day
        all_names = sorted({nm for roster in game_players.values() for nm in roster})
        player_data = {}
        for nm in all_names:
            pid = find_player_id(nm)
            if not pid:
                continue
            try:
                player_data[nm] = get_gamelog(pid)
            except Exception:
                pass

        # 4) CROSS-MARKET correlations up to (not including) this date
        corr_df = analyzer.analyze_cross_market_pairs(game_players, player_data, date_)

        if corr_df.empty:
            continue

        # 5) build 3 six-leg entries (cross-market aware, with pair_id)
        entries = build_three_entries_for_day_any_market(corr_df, game_players, lines_by_game, date_, platform)

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
            pnl = stake * (mult - 1.0)  # net profit including stake (mult=0 -> -stake)
            bal += pnl

            # Pair outcome counts
            ps, pf, pp = evaluate_pair_outcomes(ent["legs"])
            total_pair_success += ps
            total_pair_fail += pf
            total_pair_push += pp

            # PRINT every parlay & leg results
            entry_counter += 1
            print(f"\n[ENTRY #{entry_counter}] {date_} | platform={platform} | event={ent['event_id']}")
            for i, leg in enumerate(ent["legs"], 1):
                res = leg_result(leg)
                print(
                    f"  {i}. [pair {leg.get('pair_id')}] {leg['name']}  {leg['stat']}  {leg['dir']}  "
                    f"line={leg['line']}  actual={leg['actual']}  -> {res}"
                )
            outcome = "PROFIT" if pnl > 0 else ("BREAK-EVEN" if abs(pnl) < 1e-9 else "LOSS")
            print(
                f"  -> hits={hits}, pushes={pushes}, payout={mult}x, stake=${stake:.2f}, "
                f"pnl=${pnl:.2f} [{outcome}] | bankroll=${bal:.2f}"
            )
            print(f"  -> Pair outcomes: success={ps}, fail={pf}, push={pp}")

            # Save summary & legs to data structures
            results.append({
                "date": ent["date"], "platform": platform, "event_id": ent["event_id"],
                "entry_id": entry_counter, "hits": hits, "pushes": pushes, "mult": mult,
                "stake": stake, "pnl": pnl, "bankroll": bal,
                "pair_success": ps, "pair_fail": pf, "pair_push": pp
            })
            for i, leg in enumerate(ent["legs"], 1):
                legs_details.append({
                    "date": ent["date"], "platform": platform, "event_id": ent["event_id"],
                    "entry_id": entry_counter, "leg_idx": i, "pair_id": leg.get("pair_id"),
                    "player": leg["name"], "stat": leg["stat"], "dir": leg["dir"],
                    "line": leg["line"], "actual": leg["actual"], "result": leg_result(leg)
                })

    # --- write outputs
    df = pd.DataFrame(results).sort_values(["date","entry_id"])
    df.to_csv(f"jan_2025_{platform}_backtest.csv", index=False)

    df_legs = pd.DataFrame(legs_details).sort_values(["date","entry_id","leg_idx"])
    df_legs.to_csv(f"jan_2025_{platform}_backtest_legs.csv", index=False)

    print(df.tail(10))
    print("\nFinal bankroll:", bal, "  P&L:", bal - bankroll)
    print(f"Total pair outcomes across month -> success={total_pair_success}, fail={total_pair_fail}, push={total_pair_push}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["prizepicks","underdog"], default="prizepicks")
    ap.add_argument("--stake", type=float, default=100.0)
    ap.add_argument("--bankroll", type=float, default=10_000.0)
    args = ap.parse_args()
    main(args.platform, args.stake, args.bankroll)
