import os
import re
import matplotlib.pyplot as plt
import numpy as np

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

concurrencies = [1, 8, 16, 32, 64, 128, 256]
series = {
    'kv_fp16': {'ttft': [], 'throughput': [], 'c': [], 'label': 'KV Cache FP16 (No N-Gram)', 'color': '#444d56', 'marker': 'o', 'ls': '-'},
    'kv_fp8': {'ttft': [], 'throughput': [], 'c': [], 'label': 'KV Cache FP8/MXINT8 (No N-Gram)', 'color': '#00a3ff', 'marker': 'D', 'ls': '-'},
}

# Load data from 0529 KV cache folders
for c in concurrencies:
    t, h = parse_result(f'./0529/kv_fp16_no_ngram/result_v1_no_ngram_no_jeff_len2048_c{c}.txt')
    if t:
        series['kv_fp16']['ttft'].append(t); series['kv_fp16']['throughput'].append(h); series['kv_fp16']['c'].append(c)
    
    t, h = parse_result(f'./0529/kv_fp8_no_ngram/result_v1_no_ngram_no_jeff_len2048_c{c}.txt')
    if t:
        series['kv_fp8']['ttft'].append(t); series['kv_fp8']['throughput'].append(h); series['kv_fp8']['c'].append(c)

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
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 1.15), frameon=True, fontsize=11, ncol=2)
plt.title("KV Cache Benchmark (V1 Engine)\nFP16 vs FP8/MXINT8, No N-Gram", loc='right', fontsize=18, fontweight='bold', pad=40)

# Footer
plt.figtext(0.5, 0.01, 'Qwen2.5-7B-Instruct @Qualcomm aic100 ultra *1, Benchmark by RAG dataset', 
            ha='center', fontsize=10, color='grey')

plt.tight_layout()
output_path = './0529/comparison_kv_fp16_vs_fp8_no_ngram_premium.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to {output_path}")
