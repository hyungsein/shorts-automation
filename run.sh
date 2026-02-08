#!/bin/bash
cd "$(dirname "$0")"
/opt/homebrew/Caskroom/miniconda/base/envs/shorts_automation/bin/python -m src.main "$@"
