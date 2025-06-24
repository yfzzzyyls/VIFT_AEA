#!/usr/bin/env python3
"""
Visualize the curriculum learning schedule.
"""

import matplotlib.pyplot as plt
import numpy as np

# Configuration
min_seq_len = 10        # Start with 10 frames
max_seq_len = 50        # Cap at 50 frames
curriculum_step = 10    # Change every 10 epochs
curriculum_increment = 5 # Add 5 frames each step
num_epochs = 100

# Calculate curriculum schedule
epochs = []
seq_lengths = []

for epoch in range(1, num_epochs + 1):
    steps_completed = (epoch - 1) // curriculum_step
    current_max_len = min_seq_len + (steps_completed * curriculum_increment)
    current_max_len = min(current_max_len, max_seq_len)
    
    epochs.append(epoch)
    seq_lengths.append(current_max_len)

# Print the schedule
print("Curriculum Learning Schedule:")
print("=" * 50)
for step in range(0, 10):
    epoch_start = step * curriculum_step + 1
    epoch_end = min((step + 1) * curriculum_step, num_epochs)
    seq_len = min_seq_len + (step * curriculum_increment)
    seq_len = min(seq_len, max_seq_len)
    
    if epoch_start <= num_epochs:
        duration_sec = seq_len / 20  # 20 FPS for Aria
        print(f"Epochs {epoch_start:3d}-{epoch_end:3d}: {seq_len:2d} frames ({duration_sec:.2f} seconds)")
        
        if seq_len >= max_seq_len:
            break

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, seq_lengths, 'b-', linewidth=2)

# Mark step changes
step_epochs = []
step_lengths = []
for step in range(0, 10):
    epoch = step * curriculum_step + 1
    if epoch <= num_epochs:
        step_epochs.append(epoch)
        step_lengths.append(seq_lengths[epoch-1])

plt.scatter(step_epochs, step_lengths, 
           color='red', s=100, zorder=5, label='Step changes')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Maximum Sequence Length (frames)', fontsize=12)
plt.title('Step-wise Curriculum Learning Schedule', fontsize=14)
plt.grid(True, alpha=0.3)

# Add secondary y-axis for time duration
ax2 = plt.gca().twinx()
ax2.set_ylabel('Duration (seconds)', fontsize=12)
ax2.set_ylim(np.array(plt.ylim()) / 20)  # 20 FPS

plt.tight_layout()
plt.savefig('curriculum_schedule.png', dpi=150)
plt.show()

print(f"\nPlot saved as 'curriculum_schedule.png'")