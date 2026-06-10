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

concurrencies = [1, 2, 4, 8, 16, 32]
series = {
    'baseline_before': {'ttft': [], 'throughput': [], 'c': [], 'label': 'Baseline (Before Opt)', 'color': '#444d56', 'marker': 'x', 'ls': '--'},
    'ngram_before': {'ttft': [], 'throughput': [], 'c': [], 'label': 'N-Gram (Before Opt)', 'color': '#ff7f0e', 'marker': 'o', 'ls': '-'},
    'baseline_after': {'ttft': [], 'throughput': [], 'c': [], 'label': 'Baseline (After Opt)', 'color': '#888d94', 'marker': '+', 'ls': ':'},
    'ngram_after': {'ttft': [], 'throughput': [], 'c': [], 'label': 'N-Gram (After Jeff\'s Opt)', 'color': '#00d4ff', 'marker': 'D', 'ls': '-'},
}

# Load data from 0515 folders
for c in concurrencies:
    # Before Opt
    t, h = parse_result(f'./0515/before_jeff_opt/result_v1_no_ngram_no_jeff_len2048_c{c}.txt')
    if t:
        series['baseline_before']['ttft'].append(t); series['baseline_before']['throughput'].append(h); series['baseline_before']['c'].append(c)
    
    t, h = parse_result(f'./0515/before_jeff_opt/result_v1_ngram_no_jeff_len2048_c{c}.txt')
    if t:
        series['ngram_before']['ttft'].append(t); series['ngram_before']['throughput'].append(h); series['ngram_before']['c'].append(c)
        
    # After Opt
    t, h = parse_result(f'./0515/after_jeff_opt/result_v1_no_ngram_jeff_len2048_c{c}.txt')
    if t:
        series['baseline_after']['ttft'].append(t); series['baseline_after']['throughput'].append(h); series['baseline_after']['c'].append(c)
    
    t, h = parse_result(f'./0515/after_jeff_opt/result_v1_ngram_jeff_len2048_c{c}.txt')
    if t:
        series['ngram_after']['ttft'].append(t); series['ngram_after']['throughput'].append(h); series['ngram_after']['c'].append(c)

# Create Plot
fig, ax = plt.subplots(figsize=(14, 9))

for key, data in series.items():
    if not data['ttft']: continue
    ax.plot(data['ttft'], data['throughput'], data['ls'], color=data['color'], linewidth=2.5, label=data['label'], alpha=0.8)
    ax.scatter(data['ttft'], data['throughput'], marker=data['marker'], color=data['color'], s=100, zorder=5)
    
    # Annotate concurrency for the N-Gram (After) to show scale
    if key == 'ngram_after':
        for i, c in enumerate(data['c']):
            ax.annotate(f'c={c}', (data['ttft'][i], data['throughput'][i]), 
                        textcoords="offset points", xytext=(5,10), ha='left', fontsize=10, color=data['color'], fontweight='bold')

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
plt.title("N-Gram Optimization Benchmark (V1 Engine)\nBefore vs After Jeff's Opts", loc='right', fontsize=18, fontweight='bold', pad=40)

# Footer
plt.figtext(0.5, 0.01, 'Qwen2.5-7B-Instruct @Qualcomm aic100 ultra *1, Benchmark by RAG dataset', 
            ha='center', fontsize=10, color='grey')

plt.tight_layout()
output_path = './0515/comparison_before_after_opt_premium.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to {output_path}")
