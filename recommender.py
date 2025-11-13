"""
Core recommender engine with TF-IDF feature vectorization, scoring, and ranking.
Includes mock similarity matrix for demonstration purposes.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import pickle
import os

# Seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class EcoRecommender:
    """Recommender system for eco-friendly products."""
    
    def __init__(self, dataset_path: str = "products_dataset.csv"):
        """Initialize recommender with dataset."""
        self.dataset_path = dataset_path
        self.df = None
        self.vectorizer = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.load_dataset()
        self.build_features()
        self.build_similarity_matrix()
    
    def load_dataset(self):
        """Load product dataset from CSV."""
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
            print(f"Loaded {len(self.df)} products from {self.dataset_path}")
        else:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
    
    def build_features(self):
        """Build TF-IDF feature vectors from product descriptions."""
        # Combine text features for vectorization
        text_features = (
            self.df['description'].fillna('') + ' ' +
            self.df['name'].fillna('') + ' ' +
            self.df['category'].fillna('') + ' ' +
            self.df['eco_attributes'].fillna('')
        )
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform
        self.feature_matrix = self.vectorizer.fit_transform(text_features)
        print(f"Built TF-IDF feature matrix: {self.feature_matrix.shape}")
    
    def build_similarity_matrix(self):
        """Build cosine similarity matrix between all products."""
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f"Built similarity matrix: {self.similarity_matrix.shape}")
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """Convert query text to TF-IDF vector."""
        query_vector = self.vectorizer.transform([query])
        return query_vector
    
    def score_products(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Score products based on query similarity."""
        # Vectorize query
        query_vector = self.vectorize_query(query)
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.feature_matrix)[0]
        
        # Check if we have any meaningful matches
        max_similarity = np.max(similarities)
        
        # If no good matches, use keyword/category fallback
        if max_similarity < 0.01:
            similarities = self._fallback_keyword_matching(query)
        
        # Normalize eco scores to 0-1
        eco_scores = self.df['eco_score'].values / 100.0
        
        # Improved scoring: prioritize relevance first
        # Only add eco_score boost when similarity is above a threshold
        min_similarity_threshold = 0.01  # Minimum similarity to consider
        
        # For products with good similarity, add eco boost
        # For products with low similarity, prioritize similarity heavily
        combined_scores = np.where(
            similarities >= min_similarity_threshold,
            # Good matches: 85% similarity + 15% eco_score boost
            0.85 * similarities + 0.15 * eco_scores,
            # Poor matches: 95% similarity + 5% eco_score (to break ties)
            0.95 * similarities + 0.05 * eco_scores
        )
        
        # Create results dataframe
        results = self.df.copy()
        results['similarity_score'] = similarities
        results['combined_score'] = combined_scores
        results['rank'] = results['combined_score'].rank(ascending=False, method='dense')
        
        # Sort by similarity first, then by combined score
        # This ensures most relevant products come first
        results = results.sort_values(['similarity_score', 'combined_score'], ascending=[False, False])
        return results.head(top_k)
    
    def _fallback_keyword_matching(self, query: str) -> np.ndarray:
        """Fallback matching using keywords and categories when TF-IDF fails."""
        query_lower = query.lower()
        scores = np.zeros(len(self.df))
        
        # Category mapping based on keywords (order matters - more specific first)
        category_keywords = {
            'Electronics': ['air conditioner', 'ac', 'cooling', 'heating', 'fan', 'appliance', 
                          'electronic', 'device', 'phone', 'laptop', 'tablet', 'computer', 
                          'charger', 'speaker', 'tech', 'energy efficient', 'tv', 'television', 
                          'fridge', 'refrigerator'],
            'Home & Garden': ['cooling', 'heating', 'fan', 'appliance', 'home', 'garden', 
                            'led', 'solar', 'water', 'compost'],
            'Clothing': ['clothing', 'clothes', 'shirt', 'tee', 'cotton', 'bamboo', 'hemp', 
                        'linen', 'wool', 'fabric', 'apparel'],
            'Food & Beverages': ['food', 'beverage', 'drink', 'coffee', 'tea', 'snack', 
                                'juice', 'cereal', 'organic food'],
            'Beauty & Personal Care': ['beauty', 'personal care', 'shampoo', 'soap', 'lotion', 
                                      'toothpaste', 'deodorant', 'skincare'],
            'Sports & Outdoors': ['sports', 'outdoor', 'backpack', 'water bottle', 'yoga', 
                                 'running', 'camping', 'gear'],
            'Books': ['book', 'guide', 'handbook', 'reading', 'sustainability'],
            'Toys & Games': ['toy', 'game', 'puzzle', 'block', 'board game']
        }
        
        # Score based on category matches
        for idx, row in self.df.iterrows():
            category = row['category']
            keywords = category_keywords.get(category, [])
            
            # Check if query matches category keywords
            for keyword in keywords:
                if keyword in query_lower:
                    # Give higher weight to exact matches like "air conditioner"
                    if keyword == query_lower or keyword in query_lower:
                        if category == 'Electronics' and ('air conditioner' in query_lower or 'ac' in query_lower):
                            scores[idx] += 0.8  # Strong boost for Electronics + AC
                        else:
                            scores[idx] += 0.4
            
            # Boost if category name is in query
            if category.lower() in query_lower:
                scores[idx] += 0.5
            
            # Check product name and description for keyword matches
            name_lower = str(row['name']).lower()
            desc_lower = str(row['description']).lower()
            
            # Extract meaningful words from query (remove stop words)
            query_words = [w for w in query_lower.split() if len(w) > 3]
            for word in query_words:
                if word in name_lower:
                    scores[idx] += 0.2
                if word in desc_lower:
                    scores[idx] += 0.1
        
        # Normalize scores to 0-1 range
        if np.max(scores) > 0:
            scores = scores / np.max(scores) * 0.5  # Cap at 0.5 for fallback matches
        
        return scores
    
    def get_similar_products(self, product_id: str, top_k: int = 5) -> pd.DataFrame:
        """Get similar products based on product ID."""
        if product_id not in self.df['product_id'].values:
            raise ValueError(f"Product ID not found: {product_id}")
        
        # Find product index
        product_idx = self.df[self.df['product_id'] == product_id].index[0]
        
        # Get similarity scores for this product
        similarities = self.similarity_matrix[product_idx]
        
        # Create results
        results = self.df.copy()
        results['similarity'] = similarities
        results = results.sort_values('similarity', ascending=False)
        
        # Exclude the product itself and return top k
        results = results[results['product_id'] != product_id]
        return results.head(top_k)
    
    def rank_by_eco_score(self, top_k: int = 10) -> pd.DataFrame:
        """Rank products by eco score."""
        results = self.df.copy()
        results = results.sort_values('eco_score', ascending=False)
        return results.head(top_k)
    
    def filter_by_category(self, category: str) -> pd.DataFrame:
        """Filter products by category."""
        return self.df[self.df['category'] == category].copy()
    
    def filter_by_eco_attributes(self, attributes: List[str]) -> pd.DataFrame:
        """Filter products containing specified eco attributes."""
        mask = self.df['eco_attributes'].apply(
            lambda x: any(attr.lower() in str(x).lower() for attr in attributes)
        )
        return self.df[mask].copy()
    
    def get_product_by_id(self, product_id: str) -> Optional[pd.Series]:
        """Get product details by ID."""
        matches = self.df[self.df['product_id'] == product_id]
        if len(matches) > 0:
            return matches.iloc[0]
        return None
    
    def save_model(self, model_path: str = "recommender_model.pkl"):
        """Save recommender model (vectorizer and metadata)."""
        model_data = {
            'vectorizer': self.vectorizer,
            'feature_matrix': self.feature_matrix,
            'similarity_matrix': self.similarity_matrix
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "recommender_model.pkl"):
        """Load recommender model."""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.feature_matrix = model_data['feature_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file not found: {model_path}")


def create_fake_similarity_matrix(n_products: int = 500, seed: int = 42) -> np.ndarray:
    """Create a fake similarity matrix for demonstration (when real data unavailable)."""
    np.random.seed(seed)
    # Create a symmetric matrix with values between 0 and 1
    matrix = np.random.rand(n_products, n_products)
    # Make it symmetric
    matrix = (matrix + matrix.T) / 2
    # Set diagonal to 1 (products are identical to themselves)
    np.fill_diagonal(matrix, 1.0)
    # Ensure values are in [0, 1]
    matrix = np.clip(matrix, 0, 1)
    return matrix


if __name__ == "__main__":
    # Test the recommender
    print("Initializing EcoRecommender...")
    recommender = EcoRecommender("products_dataset.csv")
    
    # Test query
    print("\n" + "="*80)
    print("Testing query: 'organic cotton clothing'")
    print("="*80)
    results = recommender.score_products("organic cotton clothing", top_k=5)
    print(results[['product_id', 'name', 'category', 'eco_score', 'combined_score']].to_string())
    
    # Test similar products
    print("\n" + "="*80)
    print("Testing similar products for first product")
    print("="*80)
    first_product_id = recommender.df.iloc[0]['product_id']
    similar = recommender.get_similar_products(first_product_id, top_k=5)
    print(similar[['product_id', 'name', 'category', 'eco_score', 'similarity']].to_string())
    
    # Test eco ranking
    print("\n" + "="*80)
    print("Top 5 products by eco score")
    print("="*80)
    top_eco = recommender.rank_by_eco_score(top_k=5)
    print(top_eco[['product_id', 'name', 'category', 'eco_score']].to_string())
