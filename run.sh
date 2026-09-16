#!/bin/bash
cd /Users/div-nar/sideproj/ai-signal-engine
# launchd runs with a bare PATH (/usr/bin:/bin:...); Homebrew (opencode) lives in
# /opt/homebrew/bin, and the claude CLI lives in ~/.local/bin. Without both, the
# scheduled thesis step dies with FileNotFoundError: 'opencode' or 'claude'.
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export $(cat .env | xargs)
# -i: prevent idle system sleep for the life of this process. The Mac going to
# sleep mid-run (confirmed via `pmset -g log`) is the likely trigger behind
# two separate multi-day hangs (embedding download stalling mid-transfer).
caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 main.py "$@"
