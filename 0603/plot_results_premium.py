import os
import re
import csv

def parse_result(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    with open(file_path, 'r') as f:
        content = f.read()
    ttft_match = re.search(r'Mean TTFT \(ms\):\s+([\d\.]+)', content)
    tpot_match = re.search(r'Mean TPOT \(ms\):\s+([\d\.]+)', content)
    throughput_match = re.search(r'Output token throughput \(tok/s\):\s+([\d\.]+)', content)
    if ttft_match and throughput_match:
        tpot = float(tpot_match.group(1)) if tpot_match else None
        return float(ttft_match.group(1)), tpot, float(throughput_match.group(1))
    return None, None, None

def discover_concurrencies(*folders):
    values = set()
    pattern = re.compile(r'_c(\d+)\.txt$')
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            match = pattern.search(name)
            if match:
                values.add(int(match.group(1)))
    return sorted(values)

series = {
    '0529_kv_fp8': {
        'folder': './0529/kv_fp8_no_ngram',
        'ttft': [],
        'throughput': [],
        'c': [],
        'label': '0529 KV FP8/MXINT8 (vLLM 0.10.1)',
        'color': '#444d56',
        'marker': 'o',
        'ls': '-',
    },
    '0602_kv_fp8': {
        'folder': './0602/kv_fp8_no_ngram',
        'ttft': [],
        'throughput': [],
        'c': [],
        'label': '0602 KV FP8/MXINT8 (vLLM v0.20)',
        'color': '#ef4444',
        'marker': '^',
        'ls': '--',
    },
    '0603_kv_fp8': {
        'folder': './0603/kv_fp8_no_ngram',
        'ttft': [],
        'throughput': [],
        'c': [],
        'label': '0603 KV FP8/MXINT8 (vLLM v0.20)',
        'color': '#00a3ff',
        'marker': 'D',
        'ls': '-',
    },
}

high_concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256]
concurrencies = sorted(set(high_concurrencies) | set(discover_concurrencies(
    *(data['folder'] for data in series.values()))))

# Load data from 0529, 0602, and 0603 KV FP8 cache folders.
for data in series.values():
    for c in concurrencies:
        file_path = (
            f"{data['folder']}/"
            f"result_v1_no_ngram_no_jeff_len2048_c{c}.txt"
        )
        t, tpot, h = parse_result(file_path)
        if t is None or h is None:
            continue
        data['ttft'].append(t)
        data.setdefault('tpot', []).append(tpot)
        data['throughput'].append(h)
        data['c'].append(c)

csv_output_path = './0603/comparison_0529_0602_0603_kv_fp8_no_ngram_raw_data.csv'
with open(csv_output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['concurrency']
    for key in series:
        header.extend([
            f'{key}_max_num_seq96_mean_ttft_ms',
            f'{key}_max_num_seq96_mean_tpot_ms',
            f'{key}_max_num_seq96_output_token_throughput_tok_s',
        ])
    writer.writerow(header)

    for c in concurrencies:
        row = [c]
        for data in series.values():
            if c in data['c']:
                i = data['c'].index(c)
                row.extend([
                    f"{data['ttft'][i]:.2f}",
                    '' if data['tpot'][i] is None else f"{data['tpot'][i]:.2f}",
                    f"{data['throughput'][i]:.2f}",
                ])
            else:
                row.extend(['', '', ''])
        writer.writerow(row)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(f"Raw data saved to {csv_output_path}")
    print("matplotlib is not installed; skipped plot generation.")
    raise SystemExit(0)

# Create Plot
fig, ax = plt.subplots(figsize=(14, 9))

for key, data in series.items():
    if not data['ttft']: continue
    ax.plot(data['ttft'], data['throughput'], data['ls'], color=data['color'], linewidth=2.5, label=data['label'], alpha=0.8)
    ax.scatter(data['ttft'], data['throughput'], marker=data['marker'], color=data['color'], s=100, zorder=5)
    
    for i, c in enumerate(data['c']):
        ax.annotate(f'c={c}', (data['ttft'][i], data['throughput'][i]),
                    textcoords="offset points", xytext=(5, 10), ha='left',
                    fontsize=10, color=data['color'], fontweight='bold')

# Styling
ax.set_xlabel('Time to first token (ms)', fontsize=14, fontweight='bold', labelpad=20)
ax.set_ylabel('Throughput (TPS)', fontsize=14, fontweight='bold', labelpad=20)

# X-axis: Smaller value (Faster) on the Right, Larger value (Slower) on the Left
ax.invert_xaxis()

# Grid and Spines
ax.yaxis.grid(True, linestyle='-', alpha=0.3)
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add "Slow -> Faster" indicators (Matching 0514 style)
# Arrow points to the RIGHT (where smaller TTFT values are)
ax.annotate('', xy=(0.95, -0.12), xycoords='axes fraction', xytext=(0.05, -0.12),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(0.05, -0.12, 'Slow', transform=ax.transAxes, ha='right', va='center', fontweight='bold', color='grey', fontsize=12)
ax.text(0.95, -0.12, 'Faster', transform=ax.transAxes, ha='left', va='center', fontweight='bold', color='grey', fontsize=12)

# Legend and Title
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 1.12), frameon=True, fontsize=11, ncol=2)
plt.title("KV Cache Benchmark (V1 Engine)\n0529 vs 0602 vs 0603 FP8/MXINT8, No N-Gram\n(max-model-len=2048, max-num-seq=96)", loc='right', fontsize=18, fontweight='bold', pad=70)

# Footer
plt.figtext(0.5, 0.01, 'Qwen2.5-7B-Instruct @Qualcomm aic100 ultra *1, Benchmark by RAG dataset', 
            ha='center', fontsize=10, color='grey')

plt.tight_layout()
output_path = './0603/comparison_0529_0602_0603_kv_fp8_no_ngram_premium.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Raw data saved to {csv_output_path}")
print(f"Comparison plot saved to {output_path}")
