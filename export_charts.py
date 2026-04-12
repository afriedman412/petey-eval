"""
export_charts.py — Generate all benchmark chart PNGs from scored CSVs.

Usage:
    python export_charts.py
    python export_charts.py --out-dir charts_new
"""
import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Theme & constants ---

LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='#111111',
    plot_bgcolor='#1c1c1c',
    font=dict(family='DM Sans, sans-serif', color='#f2ece4'),
    margin=dict(t=60, b=60, l=60, r=30),
)

C = {
    'gpt-4.1-mini': '#d8572a',
    'gpt-4.1': '#e06b3e',
    'gpt-5-mini': '#ff8c66',
    'gpt-5': '#ff6b35',
    'gpt-5.4-mini': '#ffb899',
    'gpt-5.4': '#ff9e7a',
    'claude-sonnet-4-6': '#7cb9e8',
    'claude-haiku-4-5': '#5b9bd5',
    'deepseek-chat': '#86efac',
    'gemini-2.5-flash': '#f7b538',
    'llama-v3p3-70b-instruct': '#c084fc',
}

PC = {
    'pymupdf': '#86efac',
    'marker': '#f7b538',
    'unstructured': '#7cb9e8',
}

DISPLAY = {
    'gpt-4.1-mini': 'GPT-4.1 Mini',
    'gpt-4.1': 'GPT-4.1',
    'gpt-5-mini': 'GPT-5 Mini',
    'gpt-5': 'GPT-5',
    'gpt-5.4-mini': 'GPT-5.4 Mini',
    'gpt-5.4': 'GPT-5.4',
    'claude-sonnet-4-6': 'Sonnet',
    'claude-haiku-4-5': 'Haiku',
    'deepseek-chat': 'DeepSeek',
    'gemini-2.5-flash': 'Gemini Flash',
    'llama-v3p3-70b-instruct': 'Llama 70B',
    'pymupdf': 'PyMuPDF',
    'marker': 'Datalab',
    'unstructured': 'Unstructured',
}

MODEL_ORDER = [
    'gpt-4.1-mini', 'gpt-4.1', 'gpt-5-mini', 'gpt-5',
    'gpt-5.4-mini', 'gpt-5.4',
    'claude-sonnet-4-6', 'claude-haiku-4-5',
    'gemini-2.5-flash', 'deepseek-chat', 'llama-v3p3-70b-instruct',
]

PARSER_ORDER = ['pymupdf', 'marker', 'unstructured']

LLM_COST = {
    'gpt-4.1-mini': 0.01, 'gpt-4.1': 0.03,
    'gpt-5-mini': 0.015, 'gpt-5': 0.04,
    'gpt-5.4-mini': 0.015, 'gpt-5.4': 0.04,
    'claude-sonnet-4-6': 0.05, 'claude-haiku-4-5': 0.01,
    'gemini-2.5-flash': 0.01, 'deepseek-chat': 0.005,
    'llama-v3p3-70b-instruct': 0.01,
}

PARSER_COST = {'pymupdf': 0, 'marker': 0.005, 'unstructured': 0.003}


def dname(x):
    return DISPLAY.get(x, x)


# --- Chart functions ---


def chart_model_comparison(med, par, out_dir):
    med_mkr = (med[med['parser'] == 'marker']
               .set_index('model').reindex(MODEL_ORDER).dropna())
    par_mkr = (par[par['parser'] == 'marker']
               .set_index('model').reindex(MODEL_ORDER).dropna())

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f'Medical ({len(med_mkr)} models, 102 docs, 1pg)',
        f'PAR ({len(par_mkr)} models, 114 docs, ~3pg)',
    ])
    fig.add_trace(go.Bar(
        x=[dname(m) for m in med_mkr.index],
        y=med_mkr['overall'],
        marker_color=[C.get(m, '#999') for m in med_mkr.index],
        text=[f'{v:.1%}' for v in med_mkr['overall']],
        textposition='outside',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[dname(m) for m in par_mkr.index],
        y=par_mkr['overall'],
        marker_color=[C.get(m, '#999') for m in par_mkr.index],
        text=[f'{v:.1%}' for v in par_mkr['overall']],
        textposition='outside',
    ), row=1, col=2)
    fig.update_layout(
        **LAYOUT, title='Model Comparison — Datalab',
        showlegend=False, height=450, width=1000,
    )
    fig.update_yaxes(range=[0.7, 1.02], tickformat='.0%', title='Accuracy')
    fig.write_image(out_dir / '01_model_comparison.png', scale=2)
    print('  01_model_comparison.png')


def chart_parser_comparison(med, par, out_dir):
    med_3p = med.groupby('model').filter(
        lambda g: set(PARSER_ORDER).issubset(set(g['parser'])))
    par_3p = par.groupby('model').filter(
        lambda g: set(PARSER_ORDER).issubset(set(g['parser'])))
    shared_med = sorted(med_3p['model'].unique())
    shared_par = sorted(par_3p['model'].unique())

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f'Medical ({len(shared_med)} shared models)',
        f'PAR ({len(shared_par)} shared models)',
    ])
    for parser in PARSER_ORDER:
        sub = (med_3p[med_3p['parser'] == parser]
               .set_index('model').reindex(shared_med))
        fig.add_trace(go.Bar(
            name=dname(parser),
            x=[dname(m) for m in sub.index], y=sub['overall'],
            marker_color=PC[parser],
            text=[f'{v:.1%}' for v in sub['overall']],
            textposition='outside', legendgroup=parser,
        ), row=1, col=1)
    for parser in PARSER_ORDER:
        sub = (par_3p[par_3p['parser'] == parser]
               .set_index('model').reindex(shared_par))
        fig.add_trace(go.Bar(
            name=dname(parser),
            x=[dname(m) for m in sub.index], y=sub['overall'],
            marker_color=PC[parser],
            text=[f'{v:.1%}' for v in sub['overall']],
            textposition='outside', legendgroup=parser,
            showlegend=False,
        ), row=1, col=2)
    fig.update_layout(
        **LAYOUT, title='Parser Comparison — Medical vs PAR',
        barmode='group', height=450, width=1000,
        legend=dict(orientation='h', y=1.12),
    )
    fig.update_yaxes(range=[0.7, 1.02], tickformat='.0%', title='Accuracy')
    fig.write_image(out_dir / '02_parser_comparison.png', scale=2)
    print('  02_parser_comparison.png')


def chart_schema_quality(par, par_det, out_dir):
    par_s = par[par['parser'] == 'marker'].set_index('model')
    par_d = par_det[par_det['parser'] == 'marker'].set_index('model')
    shared = [m for m in MODEL_ORDER
              if m in par_s.index and m in par_d.index
              and par_d.loc[m, 'overall'] > 0.5]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Detailed schema',
        x=[dname(m) for m in shared],
        y=[par_d.loc[m, 'overall'] for m in shared],
        marker_color='#86efac',
        text=[f'{par_d.loc[m, "overall"]:.1%}' for m in shared],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='Simple schema',
        x=[dname(m) for m in shared],
        y=[par_s.loc[m, 'overall'] for m in shared],
        marker_color='#d8572a',
        text=[f'{par_s.loc[m, "overall"]:.1%}' for m in shared],
        textposition='outside',
        marker_pattern_shape='/',
    ))
    fig.update_layout(
        **LAYOUT,
        title=f'PAR Data — Schema Quality Impact (Datalab, {len(shared)} models)',
        barmode='group', height=450, width=900,
        legend=dict(orientation='h', y=1.08),
    )
    fig.update_yaxes(range=[0.7, 1.02], tickformat='.0%', title='Accuracy')
    fig.write_image(out_dir / '03_schema_quality.png', scale=2)
    print('  03_schema_quality.png')


def chart_runtime(out_dir):
    runtime = pd.DataFrame([
        ('PyMuPDF', 'Medical', 30),
        ('Datalab', 'Medical', 73),
        ('Unstructured', 'Medical', 110),
        ('PyMuPDF', 'PAR', 500),
        ('Datalab', 'PAR', 170),
        ('Unstructured', 'PAR', 177),
    ], columns=['parser', 'dataset', 'seconds'])

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Medical (102 docs, 1pg)', 'PAR (114 docs, ~3pg)',
    ])
    for i, ds in enumerate(['Medical', 'PAR'], 1):
        sub = runtime[runtime['dataset'] == ds]
        fig.add_trace(go.Bar(
            x=sub['parser'], y=sub['seconds'],
            marker_color=['#86efac', '#f7b538', '#7cb9e8'],
            text=[f'{v}s' for v in sub['seconds']],
            textposition='outside',
        ), row=1, col=i)
    fig.update_layout(
        **LAYOUT, title='Runtime by Parser',
        showlegend=False, height=400, width=900,
    )
    fig.update_yaxes(title='Avg seconds / doc')
    fig.write_image(out_dir / '04_runtime.png', scale=2)
    print('  04_runtime.png')


def chart_ocr(out_dir):
    ocr = pd.DataFrame([
        ('Tesseract (local)', 510, 0.8729),
        ('Datalab (API)', 107, 0.9071),
    ], columns=['method', 'seconds', 'accuracy'])

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Runtime (seconds/doc)', 'Accuracy',
    ])
    fig.add_trace(go.Bar(
        x=ocr['method'], y=ocr['seconds'],
        marker_color=['#d8572a', '#86efac'],
        text=[f'{v}s' for v in ocr['seconds']],
        textposition='outside',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=ocr['method'], y=ocr['accuracy'],
        marker_color=['#d8572a', '#86efac'],
        text=[f'{v:.1%}' for v in ocr['accuracy']],
        textposition='outside',
    ), row=1, col=2)
    fig.update_layout(
        **LAYOUT, title='PAR OCR Comparison (pymupdf + gpt-4.1-mini)',
        showlegend=False, height=400, width=900,
    )
    fig.update_yaxes(row=1, col=2, range=[0.8, 0.95], tickformat='.0%')
    fig.write_image(out_dir / '05_ocr_comparison.png', scale=2)
    print('  05_ocr_comparison.png')


def chart_dataset_stats(out_dir):
    stats = pd.DataFrame([
        ('Medical', 1.0, 300),
        ('PAR', 3.0, 1400),
    ], columns=['dataset', 'pages_per_doc', 'tokens_per_doc'])

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Pages per Doc', 'Input Tokens per Doc',
    ])
    fig.add_trace(go.Bar(
        x=stats['dataset'], y=stats['pages_per_doc'],
        marker_color=['#d8572a', '#f7b538'],
        text=[f'{v:.1f}' for v in stats['pages_per_doc']],
        textposition='outside',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=stats['dataset'], y=stats['tokens_per_doc'],
        marker_color=['#d8572a', '#f7b538'],
        text=[f'{v:,.0f}' for v in stats['tokens_per_doc']],
        textposition='outside',
    ), row=1, col=2)
    fig.update_layout(
        **LAYOUT, title='Dataset Comparison',
        showlegend=False, height=400, width=800,
    )
    fig.write_image(out_dir / '06_dataset_stats.png', scale=2)
    print('  06_dataset_stats.png')


def chart_cost(par, out_dir):
    cost_data = []
    for _, row in par.iterrows():
        p, m = row['parser'], row['model']
        cost = (LLM_COST.get(m, 0.01) + PARSER_COST.get(p, 0)) * 1000
        cost_data.append({
            'parser': dname(p), 'model': dname(m),
            'parser_raw': p,
            'cost_per_1k': cost, 'accuracy': row['overall'],
        })
    cost_df = pd.DataFrame(cost_data)

    # Bar chart
    fig = go.Figure()
    for parser in ['PyMuPDF', 'Datalab', 'Unstructured']:
        sub = cost_df[cost_df['parser'] == parser].sort_values('cost_per_1k')
        raw = [k for k, v in DISPLAY.items() if v == parser][0]
        fig.add_trace(go.Bar(
            name=parser, x=sub['model'], y=sub['cost_per_1k'],
            marker_color=PC.get(raw, '#999'),
            text=[f'${v:.2f}' for v in sub['cost_per_1k']],
            textposition='outside',
        ))
    fig.update_layout(
        **LAYOUT, title='Estimated Cost per 1,000 Pages',
        barmode='group', height=450, width=1000,
        yaxis_title='Cost ($)',
        legend=dict(orientation='h', y=1.08),
    )
    fig.write_image(out_dir / '07_cost_comparison.png', scale=2)
    print('  07_cost_comparison.png')

    # Scatter
    fig = go.Figure()
    for parser in ['PyMuPDF', 'Datalab', 'Unstructured']:
        sub = cost_df[cost_df['parser'] == parser]
        raw = [k for k, v in DISPLAY.items() if v == parser][0]
        fig.add_trace(go.Scatter(
            name=parser, x=sub['cost_per_1k'], y=sub['accuracy'],
            mode='markers+text',
            marker=dict(size=12, color=PC.get(raw, '#999')),
            text=sub['model'], textposition='top center',
            textfont=dict(size=9),
        ))
    fig.update_layout(
        **LAYOUT, title='PAR Data — Cost vs Accuracy',
        xaxis_title='Cost per 1,000 pages ($)',
        yaxis_title='Accuracy',
        height=500, width=900,
        legend=dict(orientation='h', y=1.08),
    )
    fig.update_yaxes(range=[0.7, 1.0], tickformat='.0%')
    fig.write_image(out_dir / '08_cost_vs_accuracy.png', scale=2)
    print('  08_cost_vs_accuracy.png')


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description='Export benchmark charts as PNGs')
    parser.add_argument(
        '--out-dir', default='charts',
        help='Output directory for PNGs (default: charts/)')
    parser.add_argument(
        '--scores-dir', default='scores',
        help='Directory containing scored CSVs (default: scores/)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    scores_dir = Path(args.scores_dir)

    med = pd.read_csv(scores_dir / 'medical.csv')
    par = pd.read_csv(scores_dir / 'par_simple.csv')
    par_det = pd.read_csv(scores_dir / 'par_detailed.csv')

    print(f'Medical: {len(med)} configs')
    print(f'PAR Simple: {len(par)} configs')
    print(f'PAR Detailed: {len(par_det)} configs')
    print(f'Output: {out_dir}/\n')

    chart_model_comparison(med, par, out_dir)
    chart_parser_comparison(med, par, out_dir)
    chart_schema_quality(par, par_det, out_dir)
    chart_runtime(out_dir)
    chart_ocr(out_dir)
    chart_dataset_stats(out_dir)
    chart_cost(par, out_dir)

    print(f'\nDone — 8 charts exported to {out_dir}/')


if __name__ == '__main__':
    main()
