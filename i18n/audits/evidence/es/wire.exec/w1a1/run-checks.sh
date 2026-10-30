#!/usr/bin/env bash
# Reproduces every mechanical check behind this audit bundle. READ-ONLY.
set -x
cd /Users/damian/lo-i18n-es
git rev-parse HEAD
git status --porcelain -- local_operator/i18n/catalogues/es/wire.exec.json
ls local_operator/i18n/catalogues/en/wire.exec.context.json
sha256sum local_operator/i18n/catalogues/en/wire.exec.json \
          local_operator/i18n/catalogues/es/wire.exec.json
python3 i18n/audits/evidence/es/wire.exec/w1a1/parity.py.txt wire.exec
python3 i18n/audits/evidence/es/wire.exec/w1a1/tokens.py.txt wire.exec
python3 i18n/audits/evidence/es/wire.exec/w1a1/epilog.py.txt
echo '--- render site of waiting_on_you (stderr receipt, not a tip slot) ---'
grep -rn "waiting_on_you" --include=*.py local_operator/
sed -n '833,841p' local_operator/exec_mode.py
