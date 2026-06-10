import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_result(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    ttft_match = re.search(r'Mean TTFT \(ms\):\s+([\d\.]+)', content)
    throughput_match = re.search(r'Output token throughput \(tok/s\):\s+([\d\.]+)', content)
    if ttft_match and throughput_match:
        return float(ttft_match.group(1)), float(throughput_match.group(1))
    return None, None

concurrencies = [1, 2, 4, 8, 16, 32]
data_ngram = {'ttft': [], 'throughput': [], 'c': []}
data_no_ngram = {'ttft': [], 'throughput': [], 'c': []}

for c in concurrencies:
    f_ngram = f'./result_ngram_len2048_max_num_seq32_c{c}.txt'
    if os.path.exists(f_ngram):
        ttft, thr = parse_result(f_ngram)
        if ttft is not None:
            data_ngram['ttft'].append(ttft)
            data_ngram['throughput'].append(thr)
            data_ngram['c'].append(c)
    
    f_no_ngram = f'./result_no_ngram_len2048_max_num_seq32_c{c}.txt'
    if os.path.exists(f_no_ngram):
        ttft, thr = parse_result(f_no_ngram)
        if ttft is not None:
            data_no_ngram['ttft'].append(ttft)
            data_no_ngram['throughput'].append(thr)
            data_no_ngram['c'].append(c)

# Create Plot
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors
color_no_ngram = '#444d56'
color_ngram = '#ff7f0e'

# Plot No Ngram (Dashed + x)
ax.plot(data_no_ngram['ttft'], data_no_ngram['throughput'], '--', color=color_no_ngram, linewidth=2, label='Qwen2.5-7B-Instruct @vLLM-Qualcomm aic100 ultra')
ax.scatter(data_no_ngram['ttft'], data_no_ngram['throughput'], marker='x', color=color_no_ngram, s=100, zorder=5)

# Plot Ngram (Solid + o)
ax.plot(data_ngram['ttft'], data_ngram['throughput'], '-', color=color_ngram, linewidth=2, label='Qwen2.5-7B-Instruct @vLLM-Qualcomm aic100 ultra + Speculative Decode')
ax.scatter(data_ngram['ttft'], data_ngram['throughput'], marker='o', color=color_ngram, s=100, zorder=5)

# Annotate concurrency
for i, c in enumerate(data_no_ngram['c']):
    ax.annotate(f'c={c}', (data_no_ngram['ttft'][i], data_no_ngram['throughput'][i]), 
                textcoords="offset points", xytext=(5,10), ha='left', fontsize=10, color=color_no_ngram, fontweight='bold')

for i, c in enumerate(data_ngram['c']):
    ax.annotate(f'c={c}', (data_ngram['ttft'][i], data_ngram['throughput'][i]), 
                textcoords="offset points", xytext=(5,-15), ha='left', fontsize=10, color=color_ngram, fontweight='bold')

# Styling
ax.set_xlabel('Time to first token (ms)', fontsize=14, fontweight='bold', labelpad=20)
ax.set_ylabel('Throughput (TPS)', fontsize=14, fontweight='bold', labelpad=20)

# Reverse X-axis
ax.invert_xaxis()

# Grid - only horizontal
ax.yaxis.grid(True, linestyle='-', alpha=0.3)
ax.xaxis.grid(False)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add "Slow -> Faster" and "Low -> High" indicators
# Y-axis High/Low
ax.annotate('', xy=(-0.05, 0.95), xycoords='axes fraction', xytext=(-0.05, 0.05),
            arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.5))
ax.text(-0.08, 0.95, 'High', transform=ax.transAxes, ha='center', va='center', fontweight='bold', color='grey', fontsize=12)
ax.text(-0.08, 0.05, 'Low', transform=ax.transAxes, ha='center', va='center', fontweight='bold', color='grey', fontsize=12)

# X-axis Slow/Faster
ax.annotate('', xy=(0.05, -0.12), xycoords='axes fraction', xytext=(0.95, -0.12),
            arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
ax.text(0.05, -0.12, 'Slow', transform=ax.transAxes, ha='right', va='center', fontweight='bold', color='grey', fontsize=12)
ax.text(0.95, -0.12, 'Faster', transform=ax.transAxes, ha='left', va='center', fontweight='bold', color='grey', fontsize=12)

# Legend and Top Title
ax.legend(loc='upper left', bbox_to_anchor=(0.05, 1.15), frameon=False, fontsize=11)
plt.title('model-len = 2048\nmax-num-seq = 32', loc='right', fontsize=16, fontweight='bold', pad=40)

# Footer
plt.figtext(0.5, 0.01, 'Qwen2.5-7B-Instruct @Qualcomm aic100 ultra *1, Benchmark by RAG dataset', 
            ha='center', fontsize=10, color='grey')

plt.tight_layout()
output_svg = './ttft_vs_throughput_premium_style_max_seq32.svg'
plt.savefig(output_svg, bbox_inches='tight')
plt.savefig(output_svg.replace('.svg', '.png'), bbox_inches='tight')
print(f"Plot saved to {output_svg}")
