#!/usr/bin/env bash
# Reproduces every mechanical check behind this audit bundle. READ-ONLY:
# no git writes, no catalogue edits, no test suite.
set -x
cd /Users/damian/lo-i18n-es
git rev-parse HEAD
git status --porcelain -- local_operator/i18n/catalogues/es/wire.settings.hub_defaults.json
sha256sum local_operator/i18n/catalogues/en/wire.settings.hub_defaults.json \
          local_operator/i18n/catalogues/en/wire.settings.hub_defaults.context.json \
          local_operator/i18n/catalogues/es/wire.settings.hub_defaults.json
python3 i18n/audits/evidence/es/wire.settings.hub_defaults/w1a1/parity.py.txt wire.settings.hub_defaults
python3 i18n/audits/evidence/es/wire.settings.hub_defaults/w1a1/labels.py.txt wire.settings.hub_defaults
grep -n "_VALUE_COLUMN = \|_ROW_INDENT = \|_MARKER_WIDTH = \|_LABEL_BUDGET = " local_operator/tui/widgets/settings_view.py
sed -n '4014,4021p' local_operator/tui/widgets/settings_view.py
grep -n "budgeted to <=72 cells" local_operator/settings_io.py
sed -n '5178,5207p' local_operator/tui/widgets/settings_view.py
sed -n '4924,4935p' local_operator/tui/widgets/settings_view.py
echo '--- cross-locale "link" renderings ---'
grep -c 'vínculo' local_operator/i18n/catalogues/es/*.json | grep -v ':0'
grep -n 'wire.slash.links' local_operator/i18n/catalogues/en/wire.slash.json local_operator/i18n/catalogues/es/wire.slash.json
grep -n 'wire.slash.hub' local_operator/i18n/catalogues/en/wire.slash.json local_operator/i18n/catalogues/es/wire.slash.json
