#!/bin/bash
cd /Users/div-nar/sideproj/ai-signal-engine
export $(cat .env | xargs)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 main.py
