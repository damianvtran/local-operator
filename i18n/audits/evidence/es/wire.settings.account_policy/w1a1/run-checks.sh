#!/usr/bin/env bash
# Reproduces every mechanical check behind this audit bundle. READ-ONLY.
set -x
cd /Users/damian/lo-i18n-es
git rev-parse HEAD
git status --porcelain -- local_operator/i18n/catalogues/es/wire.settings.account_policy.json
sha256sum local_operator/i18n/catalogues/en/wire.settings.account_policy.json \
          local_operator/i18n/catalogues/en/wire.settings.account_policy.context.json \
          local_operator/i18n/catalogues/es/wire.settings.account_policy.json
python3 i18n/audits/evidence/es/wire.settings.account_policy/w1a1/parity.py.txt wire.settings.account_policy
python3 i18n/audits/evidence/es/wire.settings.account_policy/w1a1/labels.py.txt wire.settings.account_policy
python3 i18n/audits/evidence/es/wire.settings.account_policy/w1a1/legacy_refs.py.txt
echo '--- "policy" rendered as a settings label elsewhere in es ---'
grep -n 'providers_openrouter_sort.label' local_operator/i18n/catalogues/en/wire.settings.json local_operator/i18n/catalogues/es/wire.settings.json
echo '--- "headroom" rendered elsewhere in es ---'
grep -n 'retry_usagereservepercent' local_operator/i18n/catalogues/en/wire.settings.json local_operator/i18n/catalogues/es/wire.settings.json
echo '--- "request" drift across the es locale (petición vs solicitud) ---'
grep -c 'petición' local_operator/i18n/catalogues/es/*.json | grep -v ':0'
grep -c 'solicitud' local_operator/i18n/catalogues/es/*.json | grep -v ':0'
echo '--- "computer" drift across the es locale (máquina vs equipo) ---'
grep -c 'máquina' local_operator/i18n/catalogues/es/*.json | grep -v ':0'
grep -n 'este equipo' local_operator/i18n/catalogues/es/*.json
