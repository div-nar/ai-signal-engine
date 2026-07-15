# Layer-cake launchd cadence

`run.sh` must forward its arguments to main.py — confirm it ends with:

    python main.py "$@"

## Daily cadence (current)

One agent, one decision path: `--mode trade` ingests research, runs the
(agentic) thesis pass, computes + persists the target, then executes the sell
and buy legs back-to-back in the same session. The LLM's `rebalance_urgency`
is the throttle — on an unchanged thesis it answers `"hold"` and the day is
zero-churn.

Install (unload the weekly agents first — only ONE path may ever trade):

    launchctl unload ~/Library/LaunchAgents/com.divnar.layercake.passive.plist
    launchctl unload ~/Library/LaunchAgents/com.divnar.layercake.sell.plist
    launchctl unload ~/Library/LaunchAgents/com.divnar.layercake.buy.plist
    cp ops/launchd/com.divnar.layercake.trade.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.trade.plist

Cadence (host is on IST):
- trade: Mon–Fri 19:00 IST (~09:30 ET open) — full same-day rebalance, gated
  by the LLM's urgency. Log: /tmp/layercake-trade.log

On-demand runs any time: `./run.sh --mode trade` (add `--force` to bypass the
trade gate).

## Weekly cadence (legacy alternative)

The original three-agent weekly split — passive research daily, sells Friday,
buys Monday. Load these INSTEAD of the trade agent, never alongside it:

    cp ops/launchd/com.divnar.layercake.{passive,sell,buy}.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.passive.plist
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.sell.plist
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.buy.plist

- passive: Tue–Fri 18:00 IST — ingest + compute/persist target, NO trades
- sell:    Fri 18:30 IST (~09:00 ET) — compute+persist target, execute sells
- buy:     Mon 19:00 IST (~09:30 ET open) — execute buys from the latest target

## Retired

The pre-layer-cake daily full-rebalance job must stay unloaded:

    launchctl unload ~/Library/LaunchAgents/com.divnar.ai-signal-engine.plist
