"""
export_charts.py — Generate benchmark chart PNGs from scored results.

Reads med_results.csv and par_results.csv (output of score_results.py).

Usage:
    python export_charts.py
    python export_charts.py --out-dir charts_new
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Theme & constants ---

LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='#111111',
    plot_bgcolor='#1c1c1c',
    font=dict(family='DM Sans, sans-serif', color='#f2ece4'),
)

MARGIN = dict(t=30, b=80, l=60, r=30)
MARGIN_TALL = dict(t=30, b=100, l=60, r=30)

C = {
    'gpt-4.1-mini':            '#e63946',
    'gpt-4.1':                 '#f77f00',
    'gpt-5-mini':              '#fcbf49',
    'gpt-5':                   '#2a9d8f',
    'gpt-5.4':                 '#80ced6',
    'claude-sonnet-4-6':       '#4895ef',
    'claude-haiku-4-5':        '#a855f7',
    'gemini/gemini-2.5-flash': '#06d6a0',
    'deepseek/deepseek-chat':  '#ef476f',
}

PC = {'pymupdf': '#86efac', 'datalab': '#f7b538', 'unstructured': '#7cb9e8'}

DISPLAY = {
    'gpt-4.1-mini': 'GPT-4.1<br>Mini', 'gpt-4.1': 'GPT-4.1',
    'gpt-5-mini': 'GPT-5<br>Mini', 'gpt-5': 'GPT-5',
    'gpt-5.4': 'GPT-5.4',
    'claude-sonnet-4-6': 'Claude<br>Sonnet', 'claude-haiku-4-5': 'Claude<br>Haiku',
    'deepseek/deepseek-chat': 'DeepSeek', 'gemini/gemini-2.5-flash': 'Gemini<br>Flash',
    'pymupdf': 'PyMuPDF', 'datalab': 'Datalab', 'unstructured': 'Unstructured',
}
DISPLAY_FLAT = {k: v.replace('<br>', ' ') for k, v in DISPLAY.items()}

MODEL_ORDER = [
    'gpt-4.1-mini', 'gpt-4.1', 'gpt-5-mini', 'gpt-5', 'gpt-5.4',
    'claude-sonnet-4-6', 'claude-haiku-4-5',
    'gemini/gemini-2.5-flash', 'deepseek/deepseek-chat',
]
PARSER_ORDER = ['pymupdf', 'datalab', 'unstructured']
PARSER_SHAPES = {'pymupdf': 'circle', 'datalab': 'diamond', 'unstructured': 'square'}

LLM_PRICING = {
    'gpt-4.1-mini':            (0.40, 1.60),
    'gpt-4.1':                 (2.00, 8.00),
    'gpt-5-mini':              (1.00, 4.00),
    'gpt-5':                   (2.50, 10.00),
    'gpt-5.4':                 (10.00, 40.00),
    'claude-sonnet-4-6':       (3.00, 15.00),
    'claude-haiku-4-5':        (0.80, 4.00),
    'gemini/gemini-2.5-flash': (0.15, 0.60),
    'deepseek/deepseek-chat':  (0.27, 1.10),
}
TOKENS_PER_PAGE = 500
OUTPUT_RATIO = 0.2
PARSER_COST_PER_PAGE = {'pymupdf': 0, 'datalab': 0.004, 'unstructured': 0.003}


def dname(x): return DISPLAY.get(x, x)
def dflat(x): return DISPLAY_FLAT.get(x, x)


def get_fields(df):
    return [c for c in df.columns if not c.startswith('_') and c != 'source_file']


def agg(df, fields=None):
    df = df.copy()
    if fields is None:
        fields = get_fields(df)
    df['_acc'] = df[fields].mean(axis=1)
    return (df.groupby(['_parser', '_model'])['_acc']
            .mean().reset_index().rename(columns={'_acc': 'accuracy'}))


def cost_per_1k_pages(model):
    input_rate, output_rate = LLM_PRICING.get(model, (1.0, 4.0))
    input_tokens = TOKENS_PER_PAGE * 1000
    output_tokens = input_tokens * OUTPUT_RATIO
    return (input_rate * input_tokens + output_rate * output_tokens) / 1_000_000


def pretty_field(f):
    parts = f.replace('_', ' ').title().split()
    out = []
    for p in parts:
        if p.lower() == 'ra':
            out.append('RA')
        elif p.lower() == 'adm':
            out.append('Adm.')
        else:
            out.append(p)
    return ' '.join(out)


# --- Charts ---


def chart_dataset_stats(out_dir):
    stats = pd.DataFrame([
        ('Medical', 102, 1.0, 300), ('PAR', 114, 3.0, 1400),
    ], columns=['dataset', 'docs', 'pages_per_doc', 'tokens_per_doc'])
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Pages per Doc', 'Input Tokens per Doc'])
    fig.add_trace(go.Bar(
        x=stats['dataset'], y=stats['pages_per_doc'], marker_color=['#d8572a', '#f7b538'],
        text=[f'{v:.1f}' for v in stats['pages_per_doc']], textposition='outside',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=stats['dataset'], y=stats['tokens_per_doc'], marker_color=['#d8572a', '#f7b538'],
        text=[f'{v:,.0f}' for v in stats['tokens_per_doc']], textposition='outside',
    ), row=1, col=2)
    fig.update_layout(**LAYOUT, showlegend=False, height=400, width=800, margin=MARGIN)
    fig.write_image(out_dir / '01_dataset_stats.png', scale=2)
    print('  01_dataset_stats.png')


def chart_model_comparison(med, par_s, out_dir):
    med_agg = agg(med)
    par_agg = agg(par_s)
    med_dl = med_agg[med_agg['_parser'] == 'datalab'].set_index('_model')
    par_dl = par_agg[par_agg['_parser'] == 'datalab'].set_index('_model')
    models = [m for m in MODEL_ORDER if m in med_dl.index and m in par_dl.index]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Medical (102 docs, 1pg)', 'PAR Simple (114 docs, ~3pg)',
    ])
    fig.add_trace(go.Bar(
        x=[dname(m) for m in models], y=[med_dl.loc[m, 'accuracy'] for m in models],
        marker_color=[C.get(m, '#999') for m in models],
        text=[f'{med_dl.loc[m, "accuracy"]:.1%}' for m in models], textposition='outside',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[dname(m) for m in models], y=[par_dl.loc[m, 'accuracy'] for m in models],
        marker_color=[C.get(m, '#999') for m in models],
        text=[f'{par_dl.loc[m, "accuracy"]:.1%}' for m in models], textposition='outside',
    ), row=1, col=2)
    fig.update_layout(**LAYOUT, showlegend=False, height=500, width=1000, margin=MARGIN_TALL,
                      title='Model Performance')
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(range=[0.7, 1.02], tickformat='.0%', title='Accuracy')
    fig.write_image(out_dir / '02_model_comparison.png', scale=2)
    print('  02_model_comparison.png')


def chart_parser_medical(med, out_dir):
    med_agg = agg(med)
    med_models = [m for m in MODEL_ORDER
                  if all(m in med_agg[med_agg['_parser'] == p]['_model'].values
                         for p in PARSER_ORDER)]
    fig = go.Figure()
    for parser in PARSER_ORDER:
        sub = med_agg[med_agg['_parser'] == parser].set_index('_model')
        fig.add_trace(go.Bar(
            name=dflat(parser),
            x=[dname(m) for m in med_models],
            y=[sub.loc[m, 'accuracy'] if m in sub.index else 0 for m in med_models],
            marker_color=PC[parser],
        ))
    fig.update_layout(**LAYOUT, barmode='group', height=450, width=900, margin=MARGIN_TALL,
                      legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
                      title='Model x Parser Performance (Medical)')
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(range=[0.9, 1.0], tickformat='.0%', title='Accuracy')
    fig.write_image(out_dir / '03a_parser_medical.png', scale=2)
    print('  03a_parser_medical.png')


def chart_parser_par(par_s, out_dir):
    par_agg = agg(par_s)
    par_models = [m for m in MODEL_ORDER
                  if all(m in par_agg[par_agg['_parser'] == p]['_model'].values
                         for p in PARSER_ORDER)]
    fig = go.Figure()
    for parser in PARSER_ORDER:
        sub = par_agg[par_agg['_parser'] == parser].set_index('_model')
        fig.add_trace(go.Bar(
            name=dflat(parser),
            x=[dname(m) for m in par_models],
            y=[sub.loc[m, 'accuracy'] if m in sub.index else 0 for m in par_models],
            marker_color=PC[parser],
        ))
    fig.update_layout(**LAYOUT, barmode='group', height=450, width=900, margin=MARGIN_TALL,
                      legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
                      title='Model x Parser Performance (PAR)')
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(range=[0.7, 1.02], tickformat='.0%', title='Accuracy')
    fig.write_image(out_dir / '03b_parser_par.png', scale=2)
    print('  03b_parser_par.png')


def chart_schema_quality(par_s, par_d, out_dir):
    s_agg = agg(par_s)
    d_agg = agg(par_d)
    s_dl = s_agg[s_agg['_parser'] == 'datalab'].set_index('_model')
    d_dl = d_agg[d_agg['_parser'] == 'datalab'].set_index('_model')
    models = [m for m in MODEL_ORDER if m in s_dl.index and m in d_dl.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Detailed schema', x=[dname(m) for m in models],
        y=[d_dl.loc[m, 'accuracy'] for m in models], marker_color='#86efac',
        text=[f'{d_dl.loc[m, "accuracy"]:.1%}' for m in models], textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='Simple schema', x=[dname(m) for m in models],
        y=[s_dl.loc[m, 'accuracy'] for m in models], marker_color='#d8572a',
        text=[f'{s_dl.loc[m, "accuracy"]:.1%}' for m in models], textposition='outside',
        marker_pattern_shape='/',
    ))
    fig.update_layout(**LAYOUT, barmode='group', height=500, width=900, margin=MARGIN_TALL,
                      legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
                      title='Simple Schema v. Detailed Schema')
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(range=[0.7, 1.02], tickformat='.0%', title='Accuracy')
    fig.write_image(out_dir / '04_schema_quality.png', scale=2)
    print('  04_schema_quality.png')


def chart_field_accuracy(par_s, out_dir):
    par_fields = get_fields(par_s)
    pretty_labels = [pretty_field(f) for f in par_fields]
    # Stagger labels: odd-indexed fields get a newline prefix to offset vertically
    staggered = []
    for i, label in enumerate(pretty_labels):
        staggered.append(f'<br>{label}' if i % 2 == 1 else label)
    dl = par_s[par_s['_parser'] == 'datalab']
    models = [m for m in MODEL_ORDER if m in dl['_model'].unique()]

    np.random.seed(42)
    fig = go.Figure()
    for model in models:
        sub = dl[dl['_model'] == model]
        means = [sub[f].mean() for f in par_fields]
        color = C.get(model, '#999')
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill = f'rgba({r},{g},{b},0.2)'
        jitter = np.random.uniform(-0.15, 0.15, len(par_fields))
        fig.add_trace(go.Scatter(
            x=np.arange(len(par_fields)) + jitter, y=means,
            mode='markers',
            marker=dict(size=14, color=fill, line=dict(width=2, color=color)),
            name=dflat(model),
        ))
    fig.update_layout(**LAYOUT, height=550, width=1200,
                      margin=dict(t=30, b=100, l=60, r=30),
                      legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center'),
                      xaxis=dict(tickmode='array', tickvals=list(range(len(par_fields))),
                                 ticktext=staggered, tickangle=0, tickfont=dict(size=11)))
    fig.update_yaxes(range=[0.4, 1.05], tickformat='.0%',
                     title='Accuracy')
    fig.update_layout(title='Model Accuracy by Field')
    fig.write_image(out_dir / '05_field_accuracy.png', scale=2)
    print('  05_field_accuracy.png')


def chart_ocr(par_s, out_dir):
    par_agg_all = agg(par_s)
    pymupdf_acc = par_agg_all[
        (par_agg_all['_parser'] == 'pymupdf') & (par_agg_all['_model'] == 'gpt-4.1-mini')
    ]['accuracy'].values[0]
    datalab_acc = par_agg_all[
        (par_agg_all['_parser'] == 'datalab') & (par_agg_all['_model'] == 'gpt-4.1-mini')
    ]['accuracy'].values[0]

    ocr = pd.DataFrame([
        ('Tesseract\n(PyMuPDF)', pymupdf_acc, 5.0),
        ('Datalab', datalab_acc, 0.6),
    ], columns=['method', 'accuracy', 'sec_per_doc'])

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Accuracy', 'Seconds per Doc'])
    fig.add_trace(go.Bar(
        x=ocr['method'], y=ocr['accuracy'], marker_color=['#d8572a', '#86efac'],
        text=[f'{v:.1%}' for v in ocr['accuracy']], textposition='outside', width=0.5,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=ocr['method'], y=ocr['sec_per_doc'], marker_color=['#d8572a', '#86efac'],
        text=[f'{v:.1f}s' for v in ocr['sec_per_doc']], textposition='outside', width=0.5,
    ), row=1, col=2)
    fig.update_layout(**LAYOUT, showlegend=False, height=400, width=700, margin=MARGIN)
    fig.update_yaxes(row=1, col=1, range=[0.7, 1.0], tickformat='.0%')
    fig.write_image(out_dir / '06_ocr_comparison.png', scale=2)
    print('  06_ocr_comparison.png')


def chart_cost(par_s, out_dir):
    par_agg_all = agg(par_s)
    cost_data = []
    for _, row in par_agg_all.iterrows():
        p, m = row['_parser'], row['_model']
        llm = cost_per_1k_pages(m)
        parser_cost = PARSER_COST_PER_PAGE.get(p, 0) * 1000
        cost_data.append({
            'parser': dflat(p), 'model': dflat(m),
            'parser_raw': p, 'model_raw': m,
            'cost_per_1k_llm': llm,
            'cost_per_1k_total': llm + parser_cost,
            'accuracy': row['accuracy'],
        })
    cost_df = pd.DataFrame(cost_data)

    # LLM-only bar chart
    cost_models = cost_df.drop_duplicates('model_raw').sort_values('cost_per_1k_llm')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[dname(m) for m in cost_models['model_raw']],
        y=cost_models['cost_per_1k_llm'],
        marker_color=[C.get(m, '#999') for m in cost_models['model_raw']],
        text=[f'${v:.2f}' for v in cost_models['cost_per_1k_llm']],
        textposition='outside',
    ))
    fig.update_layout(**LAYOUT, showlegend=False, height=450, width=1000,
                      margin=MARGIN_TALL, yaxis_title='LLM Cost ($)',
                      title='Cost per 1,000 Pages')
    fig.update_xaxes(tickfont=dict(size=10))
    fig.write_image(out_dir / '07_cost.png', scale=2)
    print('  07_cost.png')

    # Cost vs accuracy scatter (total cost incl. parser)
    fig = go.Figure()
    for _, row in cost_df.iterrows():
        m_raw = row['model_raw']
        p_raw = row['parser_raw']
        color = C.get(m_raw, '#999')
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=[row['cost_per_1k_total']], y=[row['accuracy']],
            mode='markers',
            marker=dict(size=14,
                        color=f'rgba({r},{g},{b},0.5)',
                        line=dict(width=2, color=color),
                        symbol=PARSER_SHAPES.get(p_raw, 'circle')),
            showlegend=False,
        ))
    for m in MODEL_ORDER:
        if m in cost_df['model_raw'].values:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers', name=dflat(m),
                marker=dict(size=10, color=C.get(m, '#999')),
                legendgroup='models', legendgrouptitle_text='Model (color)',
            ))
    for p in PARSER_ORDER:
        p_cost = PARSER_COST_PER_PAGE.get(p, 0) * 1000
        label = (f'{dflat(p)} (+${p_cost:.0f}/1K pg)'
                 if p_cost else f'{dflat(p)} (free)')
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', name=label,
            marker=dict(size=10, color='#aaa', symbol=PARSER_SHAPES[p]),
            legendgroup='parsers', legendgrouptitle_text='Parser (shape)',
        ))
    fig.update_layout(**LAYOUT,
                      title='Cost v. Accuracy for Model x Parser Combinations',
                      xaxis_title='Cost per 1,000 pages ($)',
                      yaxis_title='Accuracy',
                      height=550, width=1100,
                      margin=dict(t=50, b=60, l=60, r=30),
                      legend=dict(x=1.02, y=0.5, yanchor='middle',
                                  groupclick='toggleitem'))
    fig.update_yaxes(range=[0.7, 1.0], tickformat='.0%')
    fig.write_image(out_dir / '08_cost_vs_accuracy.png', scale=2)
    print('  08_cost_vs_accuracy.png')


# --- Main ---


def main():
    ap = argparse.ArgumentParser(description='Export benchmark charts as PNGs')
    ap.add_argument('--out-dir', default='charts', help='Output directory')
    ap.add_argument('--med', default='results/med_results.csv', help='Medical results CSV')
    ap.add_argument('--par', default='results/par_results.csv', help='PAR results CSV')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    med = pd.read_csv(args.med, index_col=0)
    par = pd.read_csv(args.par, index_col=0)
    par_s = par[par['_dataset'] == 'par_simple']
    par_d = par[par['_dataset'] == 'par_detailed']

    print(f'Medical: {len(med)} rows')
    print(f'PAR Simple: {len(par_s)} rows')
    print(f'PAR Detailed: {len(par_d)} rows')
    print(f'Output: {out_dir}/\n')

    chart_dataset_stats(out_dir)
    chart_model_comparison(med, par_s, out_dir)
    chart_parser_medical(med, out_dir)
    chart_parser_par(par_s, out_dir)
    chart_schema_quality(par_s, par_d, out_dir)
    chart_field_accuracy(par_s, out_dir)
    chart_ocr(par_s, out_dir)
    chart_cost(par_s, out_dir)

    print(f'\nDone — 8 charts exported to {out_dir}/')


if __name__ == '__main__':
    main()
