# Layer-cake launchd cadence

Three agents replace the old single `com.divnar.ai-signal-engine` job. The old job must be
unloaded first so the legacy daily full-rebalance never double-fires with these:

    launchctl unload ~/Library/LaunchAgents/com.divnar.ai-signal-engine.plist

`run.sh` must forward its arguments to main.py — confirm it ends with:

    python main.py "$@"

Install:

    cp ops/launchd/com.divnar.layercake.*.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.passive.plist
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.sell.plist
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.buy.plist

Cadence (host is on IST; times are EDT-correct, ~1h earlier in EST — still pre/at-open):
- passive: Tue–Fri 18:00 IST — ingest + compute/persist target, NO trades (Monday excluded so the buy leg uses Friday's locked target)
- sell:    Fri 18:30 IST (~09:00 ET) — compute+persist target, execute sells
- buy:     Mon 19:00 IST (~09:30 ET open) — execute buys from the latest target

Only ONE job ever trades per leg; there is no second rebalance path.
