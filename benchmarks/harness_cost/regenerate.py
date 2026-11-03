#!/usr/bin/env python3
"""Reproduce exploratory v1 tables and figures without network or model calls.

Why: the original publication mixed measured wire costs with undocumented
interactive counterfactuals. This executable preserves the raw rate-card costs,
reconstructs those counterfactuals explicitly, and checks them against frozen
references. Constraints: snapshot-only, rep == 0, no rescoring of private
artifacts, no causal/ranking claims, USD (not cents) in all machine outputs.
Only matplotlib and its pinned dependencies are needed; all accounting and
percentiles use the standard library. Never silently deduplicate or impute.
"""
from __future__ import annotations

import argparse
import collections as C
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
H = ['local-operator', 'omp', 'opencode', 'claude-code', 'codex']
FOUR = {'eb303e01', 'aceb0368', '81c425f5', 'bc2b57f3'}
RATES = {
    'sonnet': dict(input=2., cache_write_5m=2.5, cache_write_1h=4., cache_read=.1, output=10.),
    'haiku-lo': dict(input=.1, cache_write_5m=.125, cache_write_1h=.2, cache_read=.01, output=.5),
    'haiku-hi': dict(input=.5, cache_write_5m=.625, cache_write_1h=1., cache_read=.05, output=2.5),
}
EPS = 1e-10
LABEL = 'EXPLORATORY v1 — materially confounded; see CORRECTIONS.md and reviews/'


def check(condition, message):
    """Do not use assert: validation must also run under python -O."""
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(a, b, where):
    check(math.isclose(a, b, rel_tol=1e-11, abs_tol=EPS), f'{where}: {a} != {b}')


def price_row(model, u):
    """Frozen Nov 2 rate card; usage.cost is a reconciliation, not an invoice.

    Rates reproduce the metering log, not a claim about current API pricing.
    Haiku >100k and one-hour writes are not exercised by the v1 population.
    """
    for key in ['input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens']:
        check(isinstance(u.get(key), int) and u[key] >= 0, f'Bad/missing usage: {key}')
    cc = u['cache_creation_input_tokens']
    cd = u.get('cache_creation') or {}
    w1 = cd.get('ephemeral_1h_input_tokens', 0)
    w5 = cd.get('ephemeral_5m_input_tokens', cc - w1)
    check(w1 >= 0 and w5 >= 0 and w1 + w5 == cc, 'Cache TTL subtotals do not tie')
    p = dict(input=u['input_tokens'], cache_write_5m=w5, cache_write_1h=w1,
             cache_read=u['cache_read_input_tokens'], output=u['output_tokens'])
    check(model in {'anthropic/claude-sonnet-5.5', 'anthropic/claude-haiku-5.5'}, f'Unknown model {model}')
    tier = ('haiku-hi' if p['input'] + cc + p['cache_read'] > 100_000 else 'haiku-lo') if 'haiku' in model else 'sonnet'
    dollars = {k: v * RATES[tier][k] / 1e6 for k, v in p.items()}
    return sum(dollars.values()), p, dollars, RATES[tier]


def parity(actual, expected, where='reference'):
    """Compare every historical key while allowing explicitly added fields."""
    if isinstance(expected, dict):
        for k, v in expected.items():
            check(k in actual, f'{where}: missing {k}')
            parity(actual[k], v, f'{where}/{k}')
    elif isinstance(expected, list):
        check(len(actual) == len(expected), f'{where}: population mismatch')
        for i, (a, b) in enumerate(zip(actual, expected)):
            parity(a, b, f'{where}/{i}')
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        close(actual, expected, where)
    else:
        check(actual == expected, f'{where}: values differ')


def quantile(xs, p):
    """Historical quartiles used statistics.quantiles' exclusive convention."""
    check(p in (.25, .75), 'Only historical quartiles supported')
    return st.quantiles(xs,n=4,method='exclusive')[0 if p==.25 else 2]


def export(out, name, data, rows):
    """JSON retains nesting; CSV stores nested values as JSON, never repr()."""
    (out / f'{name}.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    rows=[{'evidence_status':LABEL,'corrections':'CORRECTIONS.md',**r} for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / f'{name}.csv').open('w', newline='') as f:
        # Explicit LF keeps generated CSV portable and avoids CR bytes being
        # reported as trailing whitespace by Git's publication checks.
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def summarize(rows, label):
    result = []
    for h in H:
        rs = [r for r in rows if r['harness'] == h]
        if not rs:
            continue
        cost = [r['cost_usd'] for r in rs]
        passed = sum(r['score'] >= 1 for r in rs)
        result.append(dict(evidence_status=LABEL, corrections='CORRECTIONS.md', slice=label, harness=h, n=len(rs), passed=passed,
            mean_cost=st.mean(cost), median_cost=st.median(cost), total_cost=sum(cost),
            cost_per_pass=sum(cost) / passed if passed else None,
            mean_calls=st.mean(r['calls'] for r in rs),
            mean_input_k=st.mean(sum(r[k] for k in ['input','cache_write_5m','cache_write_1h','cache_read']) for r in rs) / 1000,
            mean_output_k=st.mean(r['output'] for r in rs) / 1000,
            mean_wall_s=st.mean(r['wall_s'] for r in rs)))
    return result


def derive(snapshot):
    tasks = read_json(snapshot / 'tasks.json')
    check(len(tasks) == len({t['id'] for t in tasks}) == 24, 'Task duplicates/count')
    check(C.Counter(t['_tier'] for t in tasks) == {'short':16, 'long':8}, 'Task tiers')
    tier = {t['id']:t['_tier'] for t in tasks}
    runs = jsonl(snapshot / 'runs.jsonl')
    check(len(runs) == len({r['run_id'] for r in runs}) == 186, 'Run duplicates/count')
    check(len({(r['arm'],r['harness'],r['task_id'],r['rep']) for r in runs}) == 186, 'Duplicate experiment cells')
    check(C.Counter(r['rep'] for r in runs) == {0:180, 9:5, 8:1}, 'Unexpected repetitions')
    check(all(r['harness'] in H and r['arm'] in {'clean','full'} for r in runs),'Unknown harness/arm')
    check(C.Counter((r['rep'],r['harness']) for r in runs if r['rep']!=0) == C.Counter({(9,h):1 for h in H}) + C.Counter({(8,'claude-code'):1}), 'Smoke population')
    runmap = {r['run_id']:r for r in runs}
    req = jsonl(snapshot / 'requests.jsonl')
    check(len(req) == 1575, 'Request population changed')
    check(len({(q['run_id'],q['ts'],q['path']) for q in req}) == 1575, 'Duplicate requests')
    check(all(q['run_id'] in runmap for q in req), 'Orphan requests')
    calls = C.defaultdict(list)
    for q in req:
        if q['path'].rstrip('/').endswith('v1/messages') and q.get('usage'):
            check(q['status'] == 200, 'Priced unsuccessful response')
            cost, tokens, dollars, rates = price_row(q['served_model'], q['usage'])
            check(isinstance(q['usage'].get('cost'), (float, int)), 'Missing OpenRouter usage.cost')
            close(cost, q['usage']['cost'], 'OpenRouter call reconciliation')
            calls[q['run_id']].append((q, cost, tokens, dollars, rates))
    check(sum(map(len,calls.values())) == 1533, 'Priced request count')
    check(sum(q['path'] == 'v1/models' for q in req) == 41, 'Discovery requests')
    failed = [q for q in req if q['path'] == 'v1/messages' and not q.get('usage')]
    check(len(failed) == 1 and failed[0]['run_id'] == 'full__claude-code__eb03d19a__r0' and failed[0]['status'] == 200, 'Unmetered request classification')
    rows, reconciliations = [], []
    allctx, allcost = {}, {}
    details = {}
    for run in runs:
        rid = run['run_id']
        check(run['task_id'] in tier and run['score'] in (0, 1), f'Invalid task/score {rid}')
        check(isinstance(run.get('wall_s'),(int,float)) and math.isfinite(run['wall_s']) and run['wall_s']>=0, f'Invalid/missing wall time {rid}')
        cs = calls[rid]
        check(bool(cs), f'No priced calls {rid}')
        # Historical derivations use log (completion) order. Concurrent title
        # calls can have earlier start timestamps but appear later in this file;
        # sorting would change first-call metadata and break historical parity.
        tokens, parts = C.Counter(), C.Counter()
        for q, cost, ts, ds, rates in cs:
            tokens.update(ts)
            parts.update(ds)
        cost = sum(x[1] for x in cs)
        orcost = sum(x[0]['usage']['cost'] for x in cs)
        close(cost, sum(parts.values()), f'Components {rid}')
        close(cost, orcost, f'OpenRouter run {rid}')
        # Historical report calls every tools-bearing request 'main', including
        # Codex one-tool compaction calls. Context plots instead used >1 tools.
        legacy_main = [x for x in cs if x[0].get('n_tools',0)]
        main = [x for x in cs if x[0].get('n_tools',0) > 1]
        contexts = [sum(x[2][k] for k in ['input','cache_write_5m','cache_write_1h','cache_read']) for x in legacy_main]
        plotcontexts = [sum(x[2][k] for k in ['input','cache_write_5m','cache_write_1h','cache_read']) for x in main]
        cold = cost + (legacy_main[0][2]['cache_read'] if legacy_main else 0) * 2.4 / 1e6
        established = C.Counter()
        extra = 0.
        for q, c, ts, ds, rates in sorted(cs,key=lambda x:x[0]['ts']):
            m = q['served_model']
            extra += max(0, ts['cache_read'] - established[m]) * (rates['cache_write_5m'] - rates['cache_read']) / 1e6
            established[m] = max(established[m], ts['cache_read'] + ts['cache_write_5m'] + ts['cache_write_1h'])
        r = {k:run[k] for k in ['run_id','arm','harness','task_id','score','wall_s','exit_code','timed_out']}
        r.update(rep=run['rep'], tier=tier[run['task_id']], cost_usd=cost, model_cost_usd=cost,
                 radient_classifier_cost_usd=0., classifier_status='absent by v1 configuration; zero marginal cost, not a usage measurement',
                 total_cost_usd=cost, openrouter_usage_cost_usd=orcost,
                 calls=len(cs), main_calls=len(legacy_main), plot_main_calls=len(main),
                 first_ctx=contexts[0] if contexts else 0, max_ctx=max(contexts, default=0),
                 models=dict(C.Counter(x[0]['served_model'] for x in cs)), **tokens,
                 cold_cost=cold, cold_cross_run_estimate_usd=cost+extra,
                 infrastructure_failure=(rid == 'full__claude-code__eb03d19a__r0'))
        if run['arm'] == 'full' and run['harness'] == 'codex':
            # Exact signed historical sensitivity. Do NOT clamp to zero: later
            # reads can exceed the minimum prefix, slightly increasing cost.
            prefix = min(x[2]['cache_read'] + x[2]['cache_write_5m'] + x[2]['cache_write_1h'] for x in main)
            delta = sum((prefix - x[2]['cache_read']) * 2.4 / 1e6 for x in main[1:])
            r.update(codex_cache_corrected_cost=cost-delta,
                     codex_sensitivity_prefix_tokens=prefix,
                     codex_sensitivity_status='historical post-hoc sensitivity, NOT measured')
        reconciliations.append(dict(run_id=rid, rep=run['rep'], calls=len(cs),
            model_cost_usd=cost, component_cost_usd=sum(parts.values()),
            openrouter_usage_cost_usd=orcost, difference_usd=cost-orcost))
        details[rid] = dict(parts=parts, side=sum(x[1] for x in cs if x[0].get('n_tools',0) <= 1),
                            input_tokens=sum(tokens[k] for k in ['input','cache_write_5m','cache_write_1h','cache_read']))
        if run['rep'] == 0:
            rows.append(r)
            if run['arm'] == 'full':
                if plotcontexts:
                    allctx[rid] = plotcontexts
                allcost[rid] = [x[1] for x in cs]
    check(len(rows) == 180, 'Analysis population')
    for h in H:
        clean = [r for r in rows if r['arm'] == 'clean' and r['harness'] == h]
        full = [r for r in rows if r['arm'] == 'full' and r['harness'] == h]
        check({r['task_id'] for r in clean} == {t['id'] for t in tasks if t['_tier']=='short'}, f'Clean matrix {h}')
        expected = {t['id'] for t in tasks if h != 'codex' or t['id'][:8] in FOUR}
        check({r['task_id'] for r in full} == expected, f'Full matrix {h}')
    full = [r for r in rows if r['arm'] == 'full']
    full4 = [r for r in full if r['harness'] != 'codex']
    tables = summarize([r for r in rows if r['arm']=='clean'], 'clean / 16 short tasks')
    tables += summarize([r for r in full4 if r['tier']=='short'], 'full loadout / 16 short tasks')
    tables += summarize([r for r in full4 if r['tier']=='long'], 'full loadout / 8 long tasks')
    tables += summarize(full4, 'full loadout / all 24 tasks')
    tables += summarize([r for r in full if r['task_id'][:8] in FOUR], 'full loadout / 4 long tasks incl. Codex')
    first, growth = {}, {}
    for arm in ['clean','full']:
        for h in H:
            xs = [r['first_ctx'] for r in rows if r['arm']==arm and r['harness']==h and r['first_ctx']]
            if xs:
                first[f'{arm}/{h}'] = st.median(xs)
    for h in H:
        seqs = [[sum(x[2][k] for k in ['input','cache_write_5m','cache_write_1h','cache_read']) for x in calls[r['run_id']] if x[0].get('n_tools',0)] for r in full if r['harness']==h and r['tier']=='long']
        # Match report.py: each k averages only runs that reached that call.
        growth[h] = [st.median(s[k] for s in seqs if len(s)>k) for k in range(min(20,max(map(len,seqs))))]
    figure, components = {}, {}
    base = {r['task_id']:r for r in full if r['harness']=='local-operator'}
    for h in H:
        rs = [r for r in full if r['harness']==h]
        long = [r for r in rs if r['tier']=='long']
        like = [r for r in rs if r['task_id'][:8] in FOUR]
        passed = sum(r['score']>=1 for r in rs)
        v = dict(n=len(rs), first=st.median(r['first_ctx'] for r in rs if r['first_ctx']), passed=passed)
        if h != 'codex':
            costs = [r['cost_usd'] for r in rs]
            v.update(all=st.mean(costs), short=st.mean(r['cost_usd'] for r in rs if r['tier']=='short'),
                long=st.mean(r['cost_usd'] for r in long), long4=st.mean(r['cost_usd'] for r in like),
                cold=st.mean(r['cold_cost'] for r in rs), cpp=sum(costs)/passed,
                ratio=st.median(r['cost_usd']/base[r['task_id']]['cost_usd'] for r in rs),
                wins=sum(r['cost_usd']>base[r['task_id']]['cost_usd'] for r in rs), npairs=len(rs),
                p25=quantile(costs,.25), p75=quantile(costs,.75))
        else:
            sensitivity=st.mean(r['codex_cache_corrected_cost'] for r in rs)
            v.update(long4=sensitivity, long4_measured=st.mean(r['cost_usd'] for r in rs),
                     cpp=sum(r['codex_cache_corrected_cost'] for r in rs)/passed,
                     ratio4=sensitivity/st.mean(r['cost_usd'] for r in full if r['harness']=='local-operator' and r['task_id'][:8] in FOUR),
                     historical_fields='long4, cpp, ratio4 are post-hoc sensitivity ONLY; do not headline')
        v.update(observed_mean_model_cost_usd=st.mean(r['cost_usd'] for r in rs),
                 cold_cross_run_estimate_usd=st.mean(r['cold_cross_run_estimate_usd'] for r in rs),
                 label=LABEL)
        figure[h]=v
        n=len(long)
        parts={k:sum(details[r['run_id']]['parts'][k] for r in long)/n for k in RATES['sonnet']}
        total=st.mean(r['cost_usd'] for r in long)
        close(sum(parts.values()),total,f'Long component means {h}')
        components[h]=dict(write=parts['cache_write_5m']+parts['cache_write_1h'], read=parts['cache_read'],
            uncached=parts['input'], output=parts['output'], total=total,
            side=st.mean(details[r['run_id']]['side'] for r in long), n=n,
            tokens=dict(mean_all_long=st.mean(details[r['run_id']]['input_tokens'] for r in long), n=n,
                        mean_4long=st.mean(details[r['run_id']]['input_tokens'] for r in like)),
            cache_write_5m=parts['cache_write_5m'],cache_write_1h=parts['cache_write_1h'],
            label=LABEL, side_note='subset of components, not additive',
            radient_classifier_cost_usd=0.,total_including_classifier_usd=total)
    results=dict(tables=tables, first_call_median=first, growth=growth, rows=rows,
                 label=LABEL, currency='USD', classifier='Absent in v1; total equals model cost, not an estimate of classifier-enabled runs')
    sequences=dict(ctx=allctx,cost=allcost,_metadata=dict(label=LABEL,corrections='CORRECTIONS.md',ctx_unit='tokens',cost_unit='USD'))
    reconciliation=dict(label=LABEL, corrections='CORRECTIONS.md', runs=reconciliations, priced_requests=1533, all_requests=1575,
        excluded_discovery_requests=41, unmetered_message_requests=1,
        model_cost_usd=sum(r['model_cost_usd'] for r in reconciliations),
        openrouter_usage_cost_usd=sum(r['openrouter_usage_cost_usd'] for r in reconciliations),
        rep0_cost_usd=sum(r['cost_usd'] for r in rows),
        smoke_cost_usd=sum(r['model_cost_usd'] for r in reconciliations if r['rep'] != 0),
        note='OpenRouter usage.cost is derived from the same usage block, not an independent invoice; earlier aborted logs are not included')
    return results,figure,components,sequences,reconciliation



def cost_table_rows(results):
    """One row per explicitly matched population; never mix Codex's 4 with 24.

    Why: the readable graphic and accessible text need one authoritative set of
    values, not independently rounded/ad-hoc chart labels. Costs use observed
    model dollars and named counterfactuals; historical Codex sensitivity and
    comparative rankings are deliberately excluded. Counts are descriptive.
    """
    groups=[('clean','Clean — 16 short tasks',H),
            ('full','Full — 24 tasks (excluding Codex)',H[:4]),
            ('codex-full','Codex full — 4 selected long tasks',H[4:])]
    rows=[]
    for group,label,harnesses in groups:
        arm='clean' if group=='clean' else 'full'
        for harness in harnesses:
            runs=[r for r in results['rows'] if r['arm']==arm and r['harness']==harness]
            n=len(runs)
            check(n==(16 if arm=='clean' else 4 if harness=='codex' else 24),'Table population')
            rows.append(dict(population=group,population_label=label,harness=harness,n=n,
                passed=sum(r['score']>=1 for r in runs),
                observed_usd_per_task=st.mean(r['cost_usd'] for r in runs),
                historical_cold_usd_per_task=st.mean(r['cold_cost'] for r in runs),
                cross_run_cold_usd_per_task=st.mean(r['cold_cross_run_estimate_usd'] for r in runs)))
    return rows


DISPLAY_NAMES={'local-operator':'Local Operator','omp':'omp','opencode':'OpenCode',
               'claude-code':'Claude Code','codex':'Codex'}


def plot_labels(harnesses,runs):
    """Codex labels use the actual panel population, never a global n=4."""
    return [DISPLAY_NAMES[h]+(f" (n={sum(r['harness']==h for r in runs)})" if h=='codex' else '') for h in harnesses]


def color_contrast(a,b):
    """WCAG relative-luminance ratio; boundaries carry non-text distinctions."""
    def luminance(color):
        rgb=[int(color[i:i+2],16)/255 for i in (1,3,5)]
        linear=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        return sum(v*w for v,w in zip(linear,[.2126,.7152,.0722]))
    x,y=sorted([luminance(a),luminance(b)])
    return (y+.05)/(x+.05)


def cost_table_markdown(rows):
    """Accessible, selectable equivalent of the table graphic; fixed USD units."""
    lines=['# EXPLORATORY v1 — cost per task','',
        'Materially confounded; one run per cell. No current ranking or quality-parity claim.',
        'All costs are mean **USD per task** (six decimal places). Passed / n are logged counts.',
        '**Cold columns are counterfactual estimates, not measured cold executions.**',
        'Classifier absent in v1; total equals observed model cost. See [corrections](../CORRECTIONS.md).','']
    last=None
    for row in rows:
        if row['population']!=last:
            lines+=['## '+row['population_label'],'',
                '| Harness | Passed / n | Observed USD/task | Historical cold estimate USD/task | Cross-run cold estimate USD/task |',
                '|---|---:|---:|---:|---:|']
            last=row['population']
        values=[DISPLAY_NAMES[row['harness']],f"{row['passed']} / {row['n']}"]
        values += [f"{row[k]:.6f}" for k in ['observed_usd_per_task','historical_cold_usd_per_task','cross_run_cold_usd_per_task']]
        lines.append('| '+' | '.join(values)+' |')
        if row is rows[-1] or rows[rows.index(row)+1]['population']!=last:
            lines.append('')
    lines+=['Codex full is a separate n=4 population with a metadata/compaction confound.',
            'No historical cache-adjusted Codex cost or ratio is shown. Pass counts do not establish quality parity.','']
    return '\n'.join(lines)


def check_table_markdown(markdown,rows):
    """Parse the published numeric cells and compare them against derived JSON.

    This is intentionally not just a renderer-string equality check: incorrect
    population/count/cost cells fail even if formatting changes innocently.
    Six-place USD formatting is checked at its half-unit rounding tolerance.
    """
    check([line[3:] for line in markdown.splitlines() if line.startswith('## ')]==list(dict.fromkeys(row['population_label'] for row in rows)),'Markdown population headings')
    parsed=[line for line in markdown.splitlines() if line.startswith('| ') and not line.startswith('| Harness')]
    check(len(parsed)==len(rows),'Markdown table row population')
    for line,row in zip(parsed,rows):
        cells=[c.strip() for c in line.strip('|').split('|')]
        check(len(cells)==5,'Markdown table columns')
        check(cells[0]==DISPLAY_NAMES[row['harness']],'Markdown harness label')
        check(cells[1]==f"{row['passed']} / {row['n']}",'Markdown pass/sample count')
        for cell,key in zip(cells[2:],['observed_usd_per_task','historical_cold_usd_per_task','cross_run_cold_usd_per_task']):
            check(abs(float(cell)-row[key])<=.000000500001,'Markdown/table JSON numeric mismatch')


def figures(out, results, components):
    """Readable paired graphs/table with redundant, non-color-only encoding.

    Why: a wide three-panel chart made numeric glyphs unreadable at README
    widths. Stack panels vertically, publish numbers in a companion graphic
    and Markdown, and distinguish estimates/components with hatching and dark
    boundaries. Constraints: USD, explicit populations, observed vs estimated,
    no comparative rank or historical Codex sensitivity. Geometry gates measure
    clipping, table text overlap and font sizes at an 861 CSS-pixel embed width.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,
                         'text.parse_math':False,'hatch.linewidth':1.2})
    geometry=[]

    def save(fig,name,table=False,encoding=None):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        box=fig.bbox
        width_inches=fig.get_tightbbox(renderer).width+.4
        bounds=[]
        boxes=[]
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                b=text.get_window_extent(renderer)
                check(b.x0>=-1 and b.y0>=-1 and b.x1<=box.x1+1 and b.y1<=box.y1+1,
                      f'Clipped text {text.get_text()}')
                css_px=text.get_fontsize()/72*861/width_inches
                check(css_px>=12,f'Text below 12 CSS px at 861 px width: {text.get_text()}')
                bounds.append(dict(text=text.get_text(),bbox_pixels=list(b.extents),css_font_px_at_861=css_px))
                boxes.append(b)
        for legend in fig.legends:
            for ax in fig.axes:
                for tick in ax.get_xticklabels():
                    check(not tick.get_window_extent(renderer).overlaps(legend.get_window_extent(renderer)),
                          'Legend overlaps tick label')
        if table:
            for i,b in enumerate(boxes):
                check(not any(b.overlaps(other) for other in boxes[i+1:]),'Table text overlaps')
        geometry.append(dict(figure=name,canvas_pixels=list(box.extents),measurement_dpi=fig.dpi,
            export_dpi=150,all_text_within_canvas=True,legend_tick_overlap=False,
            table_text_overlap=False if table else None,embed_width_css_px=861,
            min_css_font_px_at_861=min(b['css_font_px_at_861'] for b in bounds),encoding=encoding,text_bounds=bounds))
        fig.savefig(out/name,dpi=150,facecolor='white',bbox_inches='tight',pad_inches=.2)
        plt.close(fig)

    # The companion table is the authoritative readable numeric view. Avoid
    # 30 tiny rotated value labels: graphs communicate shape, tables numbers.
    fig,axs=plt.subplots(3,1,figsize=(10.8,11.8))
    fig.subplots_adjust(top=.85,bottom=.16,hspace=.66,left=.12,right=.97)
    fig.suptitle('EXPLORATORY v1: model cost per task',fontsize=20,y=.975)
    fig.text(.5,.925,'Observed versus counterfactual cold estimates · mean USD/task',ha='center',fontsize=14)
    fig.text(.5,.89,'Confounded; one run per cell. Numbers: cost-per-task-table.md / .png',ha='center',fontsize=12)
    panels=[('clean','Clean: 16 short tasks per harness',H),
            ('full','Full: 24 tasks per harness (excluding Codex)',H[:4]),
            ('full','Codex full: 4 selected long tasks; separate scale',H[4:])]
    encodings=[('cost_usd','Observed','#164e63',None,-.25),
               ('cold_cost','Historical cold estimate','#64748b','///',0),
               ('cold_cross_run_estimate_usd','Cross-run cold estimate','#b45309','...',.25)]
    for ax,(arm,title,hs) in zip(axs,panels):
        rs=[r for r in results['rows'] if r['arm']==arm]
        for field,label,color,hatch,offset in encodings:
            vals=[st.mean(r[field] for r in rs if r['harness']==h) for h in hs]
            ax.bar([i+offset for i in range(len(hs))],vals,width=.23,label=label,
                   color=color,hatch=hatch,edgecolor='#111827',linewidth=1.3,zorder=3)
        # Population is contextual, not embedded in a globally reused Codex name.
        ax.set_xticks(range(len(hs)),plot_labels(hs,rs),fontsize=13)
        ax.tick_params(axis='y',labelsize=13)
        ax.set_title(title,fontsize=14,pad=12)
        ax.set_ylabel('USD / task',fontsize=13)
        ax.set_ylim(0,ax.get_ylim()[1]*1.13)
        ax.grid(axis='y',alpha=.2,zorder=0)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=12)
    fig.text(.5,.035,'Cold = estimates, not measured cold runs. Classifier absent. See CORRECTIONS.md.',ha='center',fontsize=12)
    grouped_encoding=[dict(label=label,fill=color,hatch=hatch or 'solid',outline='#111827',fill_background_contrast=color_contrast(color,'#ffffff')) for _,label,color,hatch,_ in encodings]
    check(all(e['fill_background_contrast']>=3 for e in grouped_encoding),'Grouped bar/background contrast')
    save(fig,'fig1-cost-per-task.png',encoding=grouped_encoding)

    fig,axs=plt.subplots(2,1,figsize=(10.8,9.6),gridspec_kw={'height_ratios':[1.15,1]})
    fig.subplots_adjust(top=.83,bottom=.25,hspace=.6,left=.12,right=.97)
    fig.suptitle('EXPLORATORY v1: all observed cost components',fontsize=18,y=.975)
    fig.text(.5,.925,'Full arm, long tasks only · mean observed USD/task',ha='center',fontsize=14)
    fig.text(.5,.875,'Includes uncached input and side calls. See CORRECTIONS.md.',ha='center',fontsize=12)
    # Adjacent pale fills are separated by dark outlines and distinct patterns;
    # readers need not infer boundaries or identities from hue differences.
    specs=[('uncached','Uncached input','#e5e7eb','xx'),
           ('cache_write_5m','5m cache writes','#bfdbfe','///'),
           ('cache_write_1h','1h cache writes','#ffedd5','\\\\'),
           ('read','Cache reads','#ddd6fe','...'),
           ('output','Output incl. thinking','#bbf7d0','oo')]
    for ax,hs in [(axs[0],H[:4]),(axs[1],H[4:])]:
        bottom=[0.]*len(hs)
        for key,label,color,hatch in specs:
            vals=[components[h][key] for h in hs]
            ax.bar(range(len(hs)),vals,bottom=bottom,label=label,color=color,hatch=hatch,
                   edgecolor='#111827',linewidth=1.3,width=.55,zorder=3)
            bottom=[a+b for a,b in zip(bottom,vals)]
        for i,h in enumerate(hs):
            close(bottom[i],components[h]['total'],'Figure stack')
            ax.text(i,bottom[i]*1.035,f"USD {bottom[i]:.4f}; n={components[h]['n']}",ha='center',fontsize=13)
        ax.set_xticks(range(len(hs)),[DISPLAY_NAMES[h] for h in hs],fontsize=13)
        ax.tick_params(axis='y',labelsize=13)
        ax.set_ylim(0,max(bottom)*1.25)
        ax.set_ylabel('USD / long task',fontsize=13)
        ax.set_title('8 long tasks per harness' if len(hs)>1 else 'Codex: 4 selected long tasks; separate scale',fontsize=14,pad=12)
        ax.grid(axis='y',alpha=.2,zorder=0)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.085),ncol=3,frameon=False,fontsize=12)
    fig.text(.5,.04,'Side calls are subsets, not added twice. Classifier absent. No adjusted Codex costs.',ha='center',fontsize=12)
    stacked_encoding=[dict(label=label,fill=color,hatch=hatch,outline='#111827',boundary_contrast=color_contrast(color,'#111827')) for _,label,color,hatch in specs]
    check(all(e['boundary_contrast']>=3 for e in stacked_encoding),'Stacked fill/boundary contrast')
    save(fig,'fig2-why-costs-differ.png',encoding=stacked_encoding)

    rows=cost_table_rows(results)
    fig=plt.figure(figsize=(10.8,9.6))
    fig.suptitle('EXPLORATORY v1 — cost per task',fontsize=20,y=.975)
    fig.text(.05,.925,'Observed costs and counterfactual estimates · mean USD per task',fontsize=14)
    fig.text(.05,.885,'One run per cell; materially confounded. No current ranking or quality-parity claim.',fontsize=12)
    xs=[.05,.345,.515,.705,.89]
    groups=[('clean',.82,.785,.735),('full',.49,.455,.405),('codex-full',.205,.17,.12)]
    for group,title_y,header_y,first_y in groups:
        subset=[r for r in rows if r['population']==group]
        fig.text(.05,title_y,subset[0]['population_label'],fontsize=15,weight='bold')
        for i,(x,label) in enumerate(zip(xs,['Harness','Passed / n','Observed\nUSD/task','Historical cold\nUSD/task','Cross-run cold\nUSD/task'])):
            fig.text(x,header_y,label,fontsize=13,ha='left' if i==0 else 'center',va='center',weight='bold')
        for i,row in enumerate(subset):
            y=first_y-i*.045
            cells=[DISPLAY_NAMES[row['harness']],f"{row['passed']} / {row['n']}"]
            cells += [f"{row[k]:.6f}" for k in ['observed_usd_per_task','historical_cold_usd_per_task','cross_run_cold_usd_per_task']]
            for j,(x,cell) in enumerate(zip(xs,cells)):
                fig.text(x,y,cell,fontsize=14,ha='left' if j==0 else 'center',va='center')
            fig.add_artist(Line2D([.05,.97],[y-.020,y-.020],transform=fig.transFigure,color='#9ca3af',linewidth=.6))
    fig.text(.05,.065,'Cold columns are counterfactuals, not measured cold runs. Classifier absent in v1.',fontsize=12)
    fig.text(.05,.03,'No cache-adjusted Codex values. Accessible text: cost-per-task-table.md · CORRECTIONS.md',fontsize=12)
    save(fig,'cost-per-task-table.png',table=True)
    (out/'figure_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'v1-snapshot')
    parser.add_argument('--out',type=Path,default=ROOT/'generated')
    args=parser.parse_args()
    result,figure,components,sequences,reconciliation=derive(args.snapshot)
    # Frozen references are validation only; none is an input to derive().
    # The old context sequences used >1 tools; report growth used >0 tools.
    for name,value in [('results',result),('figure_data',figure),('cost_components_long',components),('call_sequences',sequences)]:
        parity(value,read_json(args.snapshot/f'{name}.json'),name)
    args.out.mkdir(parents=True,exist_ok=True)
    export(args.out,'results',result,result['rows'])
    export(args.out,'summary_tables',result['tables'],result['tables'])
    export(args.out,'figure_data',figure,[dict(harness=h,**v) for h,v in figure.items()])
    export(args.out,'cost_components_long',components,[dict(harness=h,**v) for h,v in components.items()])
    export(args.out,'call_sequences',sequences,[dict(run_id=rid,sequence=kind,call_index=i+1,value=v,unit='tokens' if kind=='ctx' else 'USD') for kind in ['ctx','cost'] for rid,values in sequences[kind].items() for i,v in enumerate(values)])
    export(args.out,'reconciliation',reconciliation,reconciliation['runs'])
    table_rows=cost_table_rows(result)
    export(args.out,'cost-per-task-table',dict(label=LABEL,currency='USD',rows=table_rows),table_rows)
    table_path=args.out/'cost-per-task-table.md'
    table_path.write_text(cost_table_markdown(table_rows))
    check_table_markdown(table_path.read_text(),read_json(args.out/'cost-per-task-table.json')['rows'])
    figures(args.out,result,components)
    print(json.dumps(dict(status='PASS', historical_numeric_parity=True, **{k:v for k,v in reconciliation.items() if k!='runs'}),indent=2))


if __name__=='__main__':
    main()
