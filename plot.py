#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(filename='canny_performance.csv'):
    """Load performance data from CSV file."""
    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        print("Please run the C++ benchmark first: ./canny_performance_test")
        return None
    
    df = pd.read_csv(filename)
    df['size_label'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
    return df

def plot_execution_time(df, output_file='execution_time.png'):
    """Plot average execution time vs image size."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    
    # Plot bars with error bars
    bars = ax.bar(x, df['avg_time_ms'], yerr=df['std_dev_ms'], 
                   capsize=5, alpha=0.7, color='steelblue', 
                   edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['avg_time_ms'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Image Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('OpenCV Canny Edge Detection - Execution Time vs Image Size', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['size_label'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_throughput(df, output_file='throughput.png'):
    """Plot throughput (FPS) vs image size."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    
    bars = ax.bar(x, df['throughput_fps'], alpha=0.7, color='coral',
                   edgecolor='darkred', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['throughput_fps'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Image Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (FPS)', fontsize=12, fontweight='bold')
    ax.set_title('OpenCV Canny Edge Detection - Throughput vs Image Size', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['size_label'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_time_vs_pixels(df, output_file='time_vs_pixels.png'):
    """Plot execution time vs number of pixels (log-log scale)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plot with line
    ax.scatter(df['pixels'], df['avg_time_ms'], s=100, alpha=0.7, 
               color='green', edgecolor='darkgreen', linewidth=2, zorder=3)
    ax.plot(df['pixels'], df['avg_time_ms'], '--', alpha=0.5, 
            color='darkgreen', linewidth=1.5, zorder=2)
    
    # Add labels for each point
    for i, row in df.iterrows():
        ax.annotate(row['size_label'], 
                   (row['pixels'], row['avg_time_ms']),
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center',
                   fontsize=9,
                   fontweight='bold')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Pixels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('OpenCV Canny Edge Detection - Scaling Behavior (Log-Log)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_combined(df, output_file='combined_performance.png'):
    """Create a combined plot with multiple subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OpenCV Canny Edge Detection - Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    x = np.arange(len(df))
    
    # Subplot 1: Execution Time
    ax1 = axes[0, 0]
    ax1.bar(x, df['avg_time_ms'], yerr=df['std_dev_ms'], capsize=5, 
            alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Image Size', fontweight='bold')
    ax1.set_ylabel('Avg Time (ms)', fontweight='bold')
    ax1.set_title('Execution Time', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['size_label'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Throughput
    ax2 = axes[0, 1]
    ax2.bar(x, df['throughput_fps'], alpha=0.7, color='coral', edgecolor='darkred')
    ax2.set_xlabel('Image Size', fontweight='bold')
    ax2.set_ylabel('Throughput (FPS)', fontweight='bold')
    ax2.set_title('Throughput', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['size_label'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Time vs Pixels (log-log)
    ax3 = axes[1, 0]
    ax3.scatter(df['pixels'], df['avg_time_ms'], s=100, alpha=0.7,
                color='green', edgecolor='darkgreen', linewidth=2)
    ax3.plot(df['pixels'], df['avg_time_ms'], '--', alpha=0.5, color='darkgreen')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Number of Pixels', fontweight='bold')
    ax3.set_ylabel('Avg Time (ms)', fontweight='bold')
    ax3.set_title('Scaling Behavior (Log-Log)', fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Subplot 4: Min/Max/Avg comparison
    ax4 = axes[1, 1]
    width = 0.25
    x_pos = np.arange(len(df))
    ax4.bar(x_pos - width, df['min_time_ms'], width, label='Min', 
            alpha=0.8, color='lightgreen', edgecolor='darkgreen')
    ax4.bar(x_pos, df['avg_time_ms'], width, label='Avg', 
            alpha=0.8, color='steelblue', edgecolor='navy')
    ax4.bar(x_pos + width, df['max_time_ms'], width, label='Max', 
            alpha=0.8, color='salmon', edgecolor='darkred')
    ax4.set_xlabel('Image Size', fontweight='bold')
    ax4.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax4.set_title('Min/Avg/Max Comparison', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(df['size_label'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def print_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\nFastest configuration: {df.loc[df['avg_time_ms'].idxmin(), 'size_label']}")
    print(f"  Time: {df['avg_time_ms'].min():.3f} ms")
    print(f"  Throughput: {df.loc[df['avg_time_ms'].idxmin(), 'throughput_fps']:.2f} FPS")
    
    print(f"\nSlowest configuration: {df.loc[df['avg_time_ms'].idxmax(), 'size_label']}")
    print(f"  Time: {df['avg_time_ms'].max():.3f} ms")
    print(f"  Throughput: {df.loc[df['avg_time_ms'].idxmax(), 'throughput_fps']:.2f} FPS")
    
    speedup = df['avg_time_ms'].max() / df['avg_time_ms'].min()
    print(f"\nSlowdown factor (largest vs smallest): {speedup:.2f}x")
    
    # Calculate complexity approximation
    if len(df) >= 2:
        # Use first and last points for rough O(n) estimate
        pixels_ratio = df.iloc[-1]['pixels'] / df.iloc[0]['pixels']
        time_ratio = df.iloc[-1]['avg_time_ms'] / df.iloc[0]['avg_time_ms']
        complexity = np.log(time_ratio) / np.log(pixels_ratio)
        print(f"\nApproximate time complexity: O(n^{complexity:.2f})")
        print(f"  (where n is number of pixels)")
    
    print("\n" + "="*60 + "\n")

def main():
    print("OpenCV Canny Performance Plotter")
    print("="*50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} benchmark results\n")
    
    # Generate plots
    print("Generating plots...")
    plot_execution_time(df)
    plot_throughput(df)
    plot_time_vs_pixels(df)
    plot_combined(df)
    
    # Print statistics
    print_statistics(df)
    
    print("All plots generated successfully!")
    print("\nGenerated files:")
    print("  - execution_time.png")
    print("  - throughput.png")
    print("  - time_vs_pixels.png")
    print("  - combined_performance.png")

if __name__ == '__main__':
    main()