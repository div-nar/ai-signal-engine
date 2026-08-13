"""Export the latest persisted target to the dashboard's bundled targets.json.

The Vercel dashboard (ops/web/) reads account and positions live from Alpaca,
but target-vs-actual comes from a static targets.json baked into the deploy.
This rewrites that file from the engine's `targets` table; the values only
reach the site on the next `cd ops/web && vercel deploy --prod`.

    python export_targets.py                    # -> ops/web/targets.json
    python export_targets.py --out /tmp/t.json
"""
import argparse
import json
import os
import sys

from db import DEFAULT_DB, get_latest_target

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ops", "web", "targets.json")


class NoTargetError(Exception):
    """Raised when the targets table holds no row to export."""


def build_payload(target: dict) -> dict:
    """Project a targets-table row onto the keys api/data.py consumes, including
    the last trade-gate decision so the dashboard can show what the engine did."""
    thesis = (target.get("thesis_update") or "").strip()
    return {
        "id": target["id"],
        "computed_at": target["computed_at"],
        "weights": target["target_weights"],
        "regime": target["market_regime"],
        "urgency": target.get("rebalance_urgency", ""),
        "trade_gate": target.get("trade_gate", ""),
        "thesis": thesis[:400],
    }


def export(db_path: str = str(DEFAULT_DB), out_path: str = DEFAULT_OUT) -> dict:
    """Write the latest target to out_path and return the payload.

    Raises NoTargetError when the table is empty, leaving any existing file
    untouched rather than publishing empty targets over a good snapshot.
    """
    target = get_latest_target(db_path)
    if target is None:
        raise NoTargetError(
            f"no target in {db_path} — run `./run.sh --mode trade` first; "
            f"leaving {out_path} untouched"
        )

    payload = build_payload(target)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="DB path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output path")
    args = parser.parse_args()

    try:
        result = export(db_path=args.db, out_path=args.out)
    except NoTargetError as e:
        sys.exit(str(e))

    print(f"wrote {args.out}: target id={result['id']} "
          f"computed_at={result['computed_at']} "
          f"regime={result['regime']} names={len(result['weights'])}")
