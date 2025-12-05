import matplotlib.pyplot as plt
import numpy as np

# Data
items = [29150, 80778, 206394, 367946]
labels = ['29k (50k rows)', '81k (200k rows)', '206k (1M rows)', '368k (15M rows)']
x = np.arange(len(labels))
width = 0.35

# Build Time Data
build_custom = [2.63, 9.95, 40.96, 303.64]
build_lib = [15.40, 42.82, 126.23, 272.60]

# Query Latency Data
query_custom = [39, 41, 970, 2285]
query_lib = [29, 36, 73, 79]

# Memory Data
mem_custom = [502.8, 1309.8, 3542.0, 6574.5]
mem_lib = [356.8, 956.4, 2536.8, 4649.0]

# Recall Data
recall_custom = [100.0, 44.4, 55.3, 22.0]
recall_lib = [50.0, 33.3, 18.7, 0.0]

# Plot 1: Index Build Time
plt.figure(figsize=(8, 6))
plt.plot(labels, build_custom, marker='o', label='Custom (Ours)', linewidth=2)
plt.plot(labels, build_lib, marker='s', label='Datasketch (Lib)', linewidth=2)
plt.title('Index Build Time (Lower is Better)')
plt.ylabel('Time (seconds)')
plt.xlabel('Dataset Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('index_build_time.png')
plt.close()

# Plot 2: Query Latency
plt.figure(figsize=(8, 6))
plt.plot(labels, query_custom, marker='o', label='Custom (Ours)', color='red')
plt.plot(labels, query_lib, marker='s', label='Datasketch (Lib)', color='green')
plt.title('Query Latency (The Scalability Bottleneck)')
plt.ylabel('Time (microseconds)')
plt.xlabel('Dataset Size')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('query_latency.png')
plt.close()

# Plot 3: Memory Usage
plt.figure(figsize=(8, 6))
plt.bar(x - width/2, mem_custom, width, label='Custom (Ours)')
plt.bar(x + width/2, mem_lib, width, label='Datasketch (Lib)')
plt.title('Memory Usage (Lower is Better)')
plt.ylabel('Peak RAM (MB)')
plt.xlabel('Dataset Size')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.savefig('memory_usage.png')
plt.close()

# Plot 4: Recall@5
plt.figure(figsize=(8, 6))
plt.bar(x - width/2, recall_custom, width, label='Custom (Ours)', color='purple')
plt.bar(x + width/2, recall_lib, width, label='Datasketch (Lib)', color='orange')
plt.title('Recall@5 (Sensitivity)')
plt.ylabel('Recall (%)')
plt.xlabel('Dataset Size')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.savefig('recall_at_5.png')
plt.close()

print("Files generated: index_build_time.png, query_latency.png, memory_usage.png, recall_at_5.png")

methods = ['Custom PyTorch (CPU)', 'Implicit Lib (CPU)']
times = [1646.88, 2.23]  # Seconds per epoch
colors = ['blue', 'orange']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(methods, times, color=colors, width=0.5)

# Add title and labels
ax.set_title('BPR Training Time per Epoch (Lower is Better)', fontsize=14)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_yscale('log') # Log scale because difference is ~740x

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
            f'{height:.2f} s',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add Speedup Annotation
speedup = times[0] / times[1]
ax.text(0.5, 0.5, f'Speedup: {speedup:.1f}x', 
        transform=ax.transAxes, ha='center', fontsize=16, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save
output_file = 'bpr_benchmark_time.png'
plt.savefig(output_file)
print(f"Plot generated: {output_file}")