#!/usr/bin/env python3
"""
Simple GPU monitoring that outputs periodic snapshots suitable for logging
"""

import subprocess
import time
import sys

def get_gpu_stats():
    """Get GPU stats using nvidia-smi."""
    try:
        # Get utilization
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        lines = result.stdout.strip().split('\n')
        stats = []
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 7:
                stats.append({
                    'gpu_id': int(parts[0]),
                    'name': parts[1],
                    'utilization': int(parts[2]),
                    'memory_used': float(parts[3]) / 1024,  # Convert to GB
                    'memory_total': float(parts[4]) / 1024,
                    'temperature': int(parts[5]),
                    'power': float(parts[6])
                })
        return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def monitor_gpus(interval=5):
    """Monitor GPUs with periodic snapshots."""
    print("GPU Monitoring Started - Updates every {} seconds".format(interval))
    print("Time format: YYYY-MM-DD HH:MM:SS | GPU utilization% | Memory GB | Temp°C | Power W")
    print("-" * 100)
    
    try:
        while True:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            stats = get_gpu_stats()
            
            # Print summary line
            util_values = [s['utilization'] for s in stats]
            mem_values = [s['memory_used'] for s in stats]
            
            print(f"\n{timestamp}")
            for i, stat in enumerate(stats):
                mem_percent = (stat['memory_used'] / stat['memory_total']) * 100
                print(f"  GPU {i}: {stat['utilization']:3d}% util | {stat['memory_used']:5.1f}/{stat['memory_total']:5.1f} GB ({mem_percent:4.1f}%) | {stat['temperature']:3d}°C | {stat['power']:5.1f}W")
            
            # Print average
            if stats:
                avg_util = sum(util_values) / len(util_values)
                total_mem = sum(mem_values)
                print(f"  AVG: {avg_util:5.1f}% | Total Memory: {total_mem:.1f} GB")
            
            sys.stdout.flush()
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor GPU usage with periodic snapshots')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    args = parser.parse_args()
    
    monitor_gpus(args.interval)