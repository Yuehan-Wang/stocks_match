import numpy as np
from src.processor import z_score_normalize

class PatternMatcher:
    def __init__(self, windows, dates):
        self.windows = windows
        self.dates = dates
        self.normalized_windows = np.array([z_score_normalize(w) for w in windows])

    def find_similar_patterns(self, target_window, top_k=5):
        norm_target = z_score_normalize(target_window)
        
        # Calculate Euclidean distances
        # Dist = sqrt(sum((a - b)^2))
        distances = np.linalg.norm(self.normalized_windows - norm_target, axis=1)
        
        # Get indices of the smallest distances
        # We exclude the last item because that is the target itself
        sorted_indices = np.argsort(distances)
        
        # Filter out overlapping windows to ensure distinct patterns
        # (Simple implementation: just take top k distinct)
        matches = []
        for idx in sorted_indices:
            # Skip if the match is the target itself (distance ~ 0)
            if distances[idx] < 1e-9:
                continue
            
            matches.append({
                'index': idx,
                'date': self.dates[idx],
                'distance': distances[idx],
                'window': self.windows[idx]
            })
            
            if len(matches) >= top_k:
                break
                
        return matches