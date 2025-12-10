import matplotlib.pyplot as plt
import numpy as np
from src.processor import z_score_normalize

def plot_matches(target_window, matches, future_length=5):
    plt.figure(figsize=(14, 7))
    
    # 1. Plot Current Pattern (Target) - Thick black line
    # We normalize the target window for visual comparison
    norm_target = z_score_normalize(target_window)
    x_current = range(len(norm_target))
    plt.plot(x_current, norm_target, label='Current Pattern (Last 30 Days)', color='black', linewidth=3, zorder=10)
    
    # 2. Plot Historical Matches + Future Projections
    for i, match in enumerate(matches):
        # Get the full data (30 days history + 5 days future)
        full_pattern = match['full_window'] 
        
        # CRITICAL: We must normalize the "Future" using the "History's" statistics.
        # This prevents the future data from changing the shape of the history.
        history_part = full_pattern[:-future_length]
        mean = np.mean(history_part)
        std = np.std(history_part)
        
        if std == 0: std = 1
        norm_full = (full_pattern - mean) / std
        
        # Define X-axes
        x_history = range(len(history_part))
        # Start future plot from the last history point to ensure lines connect
        x_future = range(len(history_part) - 1, len(full_pattern)) 
        
        # Color for this specific match
        color = plt.cm.tab10(i) 
        date_str = match['date'].strftime('%Y-%m-%d')
        
        # Plot History part (Dashed line)
        plt.plot(x_history, norm_full[:-future_length], 
                 linestyle='--', alpha=0.5, color=color, linewidth=1.5)
        
        # Plot Future part (Solid line with dots) - This is the projection
        plt.plot(x_future, norm_full[-future_length-1:], 
                 linestyle='-', marker='o', markersize=4, alpha=0.8, color=color, 
                 label=f"Match {i+1}: {date_str}")

    # Draw a vertical line separating History and Projection
    plt.axvline(x=len(norm_target)-1, color='gray', linestyle=':', alpha=0.5)
    plt.text(len(norm_target) + 0.5, plt.ylim()[1]*0.9, 'Projection Zone', color='gray', fontsize=10)

    plt.title(f"Pattern Matching & Future Projection (Next {future_length} Days)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlabel("Days")
    plt.ylabel("Normalized Price (Z-Score)")
    plt.show()