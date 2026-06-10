import os
import re
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_result(file_path):
    if not os.path.exists(file_path):
        return None, None
    with open(file_path, 'r') as f:
        content = f.read()
    ttft_match = re.search(r'Mean TTFT \(ms\):\s+([\d\.]+)', content)
    throughput_match = re.search(r'Output token throughput \(tok/s\):\s+([\d\.]+)', content)
    if ttft_match and throughput_match:
        return float(ttft_match.group(1)), float(throughput_match.group(1))
    return None, None

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

concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256]
series = {
    '0529_kv_fp16': {
        'folder': '0529/kv_fp16_no_ngram',
        'filename': 'result_v1_no_ngram_no_jeff_len2048_c{c}.txt',
        'ttft': [], 'throughput': [], 'c': [],
        'label': '0529 KV Cache FP16 (No N-Gram, max_num_seq=96)',
        'color': '#444d56', 'marker': 'o', 'ls': '-',
    },
    '0529_kv_fp8': {
        'folder': '0529/kv_fp8_no_ngram',
        'filename': 'result_v1_no_ngram_no_jeff_len2048_c{c}.txt',
        'ttft': [], 'throughput': [], 'c': [],
        'label': '0529 KV Cache FP8/MXINT8 (No N-Gram, max_num_seq=96)',
        'color': '#00a3ff', 'marker': 'D', 'ls': '-',
    },
    '0601_kv_fp16_max_num_seq128': {
        'folder': '0601/kv_fp16_no_ngram_max_num_seq128',
        'filename': 'result_v1_no_ngram_max_num_seq128_no_jeff_len2048_c{c}.txt',
        'ttft': [], 'throughput': [], 'c': [],
        'label': '0601 KV Cache FP16 (No N-Gram, max_num_seq=128)',
        'color': '#8a5cf6', 'marker': '^', 'ls': '--',
    },
    '0601_kv_fp8': {
        'folder': '0601/kv_fp8_no_ngram',
        'filename': 'result_v1_no_ngram_no_jeff_len2048_c{c}.txt',
        'ttft': [], 'throughput': [], 'c': [],
        'label': '0601 KV Cache FP8/MXINT8 (No N-Gram, max_num_seq=256)',
        'color': '#f97316', 'marker': 's', 'ls': '--',
    },
}

# Load data from 0529 and 0601 KV cache folders
for c in concurrencies:
    for data in series.values():
        file_path = os.path.join(ROOT_DIR, data['folder'],
                                 data['filename'].format(c=c))
        t, h = parse_result(file_path)
        if t:
            data['ttft'].append(t)
            data['throughput'].append(h)
            data['c'].append(c)

# Create Plot
fig, ax = plt.subplots(figsize=(14, 11))

for key, data in series.items():
    if not data['ttft']: continue
    ax.plot(data['ttft'], data['throughput'], data['ls'], color=data['color'], linewidth=2.5, label=data['label'], alpha=0.8)
    ax.scatter(data['ttft'], data['throughput'], marker=data['marker'], color=data['color'], s=100, zorder=5)
    
    for i, c in enumerate(data['c']):
        ax.annotate(f'c={c}', (data['ttft'][i], data['throughput'][i]),
                    textcoords="offset points", xytext=(5, 10), ha='left',
                    fontsize=10, color=data['color'], fontweight='bold')

# Styling
ax.set_xlabel('Time to first token (ms)', fontsize=14, fontweight='bold', labelpad=34)
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
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 1.18), frameon=True, fontsize=10, ncol=2)
plt.title("KV Cache Benchmark (V1 Engine)\nFP16 vs FP8/MXINT8, No N-Gram",
          loc='right', fontsize=18, fontweight='bold', y=1.23)

# Footer
plt.figtext(0.5, 0.01, 'Qwen2.5-7B-Instruct @Qualcomm aic100 ultra *1, Benchmark by RAG dataset', 
            ha='center', fontsize=10, color='grey')

plt.tight_layout(rect=(0, 0.04, 1, 0.84))
output_path = os.path.join(ROOT_DIR, '0601',
                           'comparison_kv_fp16_vs_fp8_no_ngram_premium.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to {output_path}")
