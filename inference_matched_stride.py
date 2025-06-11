#!/usr/bin/env python3
"""
Run inference with the same stride used during training
This should fix the scale mismatch issue
"""

import subprocess
import sys
from pathlib import Path

# CRITICAL: Match the stride used in training data generation
TRAINING_STRIDE = 30  # The model expects 1.5 second intervals (30 frames at 20fps)
TEST_SEQUENCES = ["016", "017", "018", "019"]
CHECKPOINT = "fixed_scale_v1/epoch_epoch=024_val_loss_val/total_loss=14.812485.ckpt"

print("="*80)
print("RUNNING INFERENCE WITH MATCHED STRIDE")
print("="*80)
print(f"Training stride: {TRAINING_STRIDE} frames (1.5 seconds at 20fps)")
print(f"This matches the temporal resolution the model was trained on")
print("="*80)

results = {}

for seq_id in TEST_SEQUENCES:
    print(f"\nProcessing sequence {seq_id}...")
    
    # Run inference with proper stride
    cmd = [
        "python", "inference_full_sequence.py",
        "--sequence-id", seq_id,
        "--checkpoint", CHECKPOINT,
        "--processed-dir", "data/aria_processed",
        "--stride", str(TRAINING_STRIDE),  # This is the key!
        "--batch-size", "64"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract metrics from output
        for line in result.stdout.split('\n'):
            if "Average translation error" in line:
                ate = float(line.split(':')[1].strip().split()[0])
                results[seq_id] = {'ate': ate}
            elif "frame-to-frame" in line and seq_id in results:
                rpe = float(line.split(':')[1].strip().split()[0])
                results[seq_id]['rpe'] = rpe
        
        # Generate plots for first sequence
        if seq_id == TEST_SEQUENCES[0]:
            print(f"\nGenerating visualization for sequence {seq_id}...")
            plot_cmd = [
                "python", "plot_short_term_trajectory.py",
                "--npz-file", f"inference_results_realtime_seq_{seq_id}_stride_{TRAINING_STRIDE}.npz",
                "--output-dir", f"short_term_plots_matched_stride_{seq_id}",
                "--duration", "5"
            ]
            subprocess.run(plot_cmd)
    else:
        print(f"❌ Inference failed for sequence {seq_id}")
        print(result.stderr)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

if results:
    print("\nMetrics with matched stride:")
    print(f"{'Sequence':<10} {'ATE (m)':<10} {'RPE (cm)':<10}")
    print("-" * 30)
    
    ate_values = []
    rpe_values = []
    
    for seq_id, metrics in results.items():
        ate = metrics.get('ate', 0)
        rpe = metrics.get('rpe', 0)
        ate_values.append(ate)
        rpe_values.append(rpe)
        print(f"{seq_id:<10} {ate:<10.3f} {rpe:<10.2f}")
    
    print("-" * 30)
    print(f"{'Average':<10} {sum(ate_values)/len(ate_values):<10.3f} {sum(rpe_values)/len(rpe_values):<10.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("The model was trained on data with 30-frame intervals (1.5 seconds)")
print("Running inference with stride=1 expects 50ms predictions")
print("This 30x temporal mismatch was causing the scale issues")
print("\nFor real-time AR/VR applications, you need to either:")
print("  1. Retrain the model on consecutive frames (stride=1)")
print("  2. Use interpolation between sparse predictions")
print("  3. Accept lower temporal resolution (1.5s updates)")
print("="*80)