from pathlib import Path
import sys,json,hashlib,re,datetime
root=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
sys.path.insert(0,str(root))
from scripts.i18n import check,ledger
from local_operator.i18n import runtime
scratch=Path('/Users/damian/.local-operator/sessions/32077ea6da37/scratchpad')
paths={'en':root/'local_operator/i18n/catalogues/en/wire.teams.effective.json','context':root/'local_operator/i18n/catalogues/en/wire.teams.effective.context.json','fr':Path('/Users/damian/.local-operator/sessions/34a8703ccbff/scratchpad/wire.teams.effective.fr.json'),'translator_receipt':Path('/Users/damian/.local-operator/sessions/34a8703ccbff/scratchpad/translator-receipt.fr.json'),'style':root/'i18n/style-guides/fr.md','glossary':root/'i18n/glossary.md'}
hashes={name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths.items()}
expected={'en':'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','context':'01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','fr':'9c7ffd5f52a5ad10267d739074e19c56c90f532b6c79b15314e2256655038226','translator_receipt':'23c5bab253b77d5f858781da77afcfa84a12db532bd1c2d3c6fb9ae1f1933dff'}
for name,want in expected.items(): assert hashes[name]['sha256']==want,name

def unique(pairs):
 d={}
 for k,v in pairs:
  assert k not in d,('duplicate',k)
  d[k]=v
 return d
source=json.loads(paths['en'].read_text(),object_pairs_hook=unique)
target=json.loads(paths['fr'].read_text(),object_pairs_hook=unique)
context=json.loads(paths['context'].read_text(),object_pairs_hook=unique)
assert set(source)==set(target)==set(context)
assert len(source)==9
plural=json.loads((root/'local_operator/i18n/data/plural_rules.json').read_text())
problems=[]
rows=[]
for key,english in source.items():
 translated=target[key]
 want=runtime.message_arguments(english); got=runtime.message_arguments(translated)
 problem=check._arguments_problem(want,got,translated_source=translated,locale='fr',plural=plural)
 assert problem is None,(key,problem)
 check._check_plural_categories(problems,'fr','wire.teams.effective',key,translated,plural,'fr')
 assert not runtime.plural_selectors(english) and not runtime.plural_selectors(translated)
 # Independent literal checks supplement repository ICU routines.
 assert re.findall(r'\{[^{}]+\}',english)==re.findall(r'\{[^{}]+\}',translated),key
 for literal in ['subagents.models.','provider/model']:
  assert english.count(literal)==translated.count(literal),(key,literal)
 if key.endswith('tier_invalid_mapping'):
  for literal in ['model','effort']:
   assert re.search(r'\b'+literal+r'\b',translated),literal
 assert not re.search('[\u202a-\u202e\u2066-\u2069]',translated)
 assert not re.search(r'[,}]\s*(plural|select)\b',translated)
 assert not re.findall(r'`[^`]*`|</?[^>]+>',english)
 assert not re.findall(r'`[^`]*`|</?[^>]+>',translated)
 assert check.catalogues.context_entry_problem(context[key]) is None
 rows.append({'key':key,'source':english,'current':translated,'repository_argument_signature':got,'characters':len(translated),'mechanical':'PASS'})
assert not problems,problems
print('PASS: pinned SHA-256 inputs; strict JSON/duplicate members; 9/9 keys and context; repository ICU parser/argument/plural validator routines; independent ordered placeholders/technical literals/markup/bidi checks.')
print('CLDR: zero plural/select messages; French one/many/other need no arms in plain interpolation messages.')
print('Scope: direct repository validator routines on external scratch candidate, NOT full scripts/i18n/check.py CLI, generated check, build, test suite or rendered UI.')
(scratch/'mechanical-results.fr.json').write_text(json.dumps({'input_hashes':hashes,'keys':rows,'problems':problems,'french_categories':plural['locales']['fr']['categories']},ensure_ascii=False,indent=2)+'\n')
