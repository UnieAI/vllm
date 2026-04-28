import matplotlib.pyplot as plt
import os

# Data ngram (previously extracted)
concurrency = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
throughput_ngram = [7.08, 11.77, 17.06, 22.45, 25.66, 25.86, 25.60, 25.49, 25.64, 25.72]
ttft_ngram = [2157.94, 4291.44, 8495.75, 15179.97, 30442.06, 64530.85, 103651.08, 142142.67, 220132.60, 298612.87]

# Data without ngram (newly extracted from result_c1~c128)
throughput_no_ngram = [11.41, 19.17, 28.37, 37.53, 44.62, 44.53, 44.33, 44.42, 44.69, 44.59]
ttft_no_ngram = [2171.02, 4329.20, 6511.33, 17378.98, 34585.46, 55675.45, 79618.81, 102928.99, 148710.12, 194849.11]

# Create output directory
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Plot: Throughput vs TTFT comparison
plt.figure(figsize=(12, 8))

# Plot ngram data
plt.plot(ttft_ngram, throughput_ngram, marker='o', linestyle='-', color='b', linewidth=2, label='with ngram')
# Plot without ngram data
plt.plot(ttft_no_ngram, throughput_no_ngram, marker='s', linestyle='--', color='r', linewidth=2, label='without ngram')

# Annotate concurrency levels
for i, c in enumerate(concurrency):
    plt.annotate(f'c={c}', (ttft_ngram[i], throughput_ngram[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='blue')
    plt.annotate(f'c={c}', (ttft_no_ngram[i], throughput_no_ngram[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red')

plt.title('Throughput (TPS) vs Mean TTFT (ms) - Qwen2.5-7B on AI 100', fontsize=14, pad=20)
plt.xlabel('Mean TTFT (ms)', fontsize=12)
plt.ylabel('Throughput (Tokens/sec)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
plot_path = os.path.join(output_dir, 'throughput_vs_ttft_comparison.png')
plt.savefig(plot_path)

print(f"Plot generated successfully: {plot_path}")