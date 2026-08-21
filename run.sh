#!/bin/bash
cd /Users/div-nar/sideproj/ai-signal-engine
# launchd runs with a bare PATH (/usr/bin:/bin:...); Homebrew (opencode) lives in /opt/homebrew/bin.
# Without this the scheduled thesis step dies with FileNotFoundError: 'opencode'.
export PATH="/opt/homebrew/bin:$PATH"
export $(cat .env | xargs)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 main.py "$@"
