import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        # Using IndexFlatL2 for exact Euclidean search
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_data(self, features, dates, ticker, start_indices):
        """
        Args:
            features: Already normalized and concatenated vectors (Price + Volume)
            dates: List of dates
            ticker: Stock symbol
            start_indices: Pointers to raw data
        """
        if len(features) == 0:
            return

        # Faiss requires float32
        vectors = features.astype('float32')
        
        # Add to index
        self.index.add(vectors)
        
        # Store metadata
        for i in range(len(features)):
            self.metadata.append({
                'ticker': ticker,
                'date': dates[i],
                'start_index': start_indices[i] 
            })

    def search(self, target_feature, k=10):
        # Reshape for Faiss (1, dimension)
        query_vector = target_feature.astype('float32').reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            meta = self.metadata[idx]
            results.append({
                'ticker': meta['ticker'],
                'date': meta['date'],
                'start_index': meta['start_index'],
                'distance': distances[0][i]
            })
            
        return results