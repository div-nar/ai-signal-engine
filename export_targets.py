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

from db import get_latest_target

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ops", "web", "targets.json")


def build_payload(target: dict) -> dict:
    """Project a targets-table row onto the four keys api/data.py consumes."""
    return {
        "id": target["id"],
        "computed_at": target["computed_at"],
        "weights": target["target_weights"],
        "regime": target["market_regime"],
    }


def export(db_path: str | None = None, out_path: str = DEFAULT_OUT) -> dict:
    """Write the latest target to out_path. Exits non-zero if none exists,
    leaving any existing file untouched rather than publishing empty targets."""
    target = get_latest_target(db_path) if db_path else get_latest_target()
    if target is None:
        sys.exit("no target in DB — run `./run.sh --mode trade` first; "
                 f"leaving {out_path} untouched")

    payload = build_payload(target)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="DB path (default: engine default)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output path")
    args = parser.parse_args()

    result = export(db_path=args.db, out_path=args.out)
    print(f"wrote {args.out}: target id={result['id']} "
          f"computed_at={result['computed_at']} "
          f"regime={result['regime']} names={len(result['weights'])}")
