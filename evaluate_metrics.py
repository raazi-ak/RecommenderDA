"""
Evaluation metrics calculator for EcoShopper Bot.
Calculates NDCG, MAP, Precision@K, Recall@K, and MRR.
"""

import pandas as pd
import numpy as np
from recommender import EcoRecommender
from typing import List, Dict, Set

# Test queries with ground truth relevance
TEST_QUERIES = {
    "organic cotton": {
        "relevant_products": ["Organic Hemp Tee", "Organic Wool Tee", "Organic Bamboo Tee", 
                             "Organic Linen Tee", "Organic Cotton Tee", "Sustainable Cotton"],
        "category": "Clothing"
    },
    "LED": {
        "relevant_products": ["Eco LED", "Green LED Kit", "Energy Efficient LED Set", 
                             "Recycled LED Set", "Carbon Neutral LED Set"],
        "category": "Home & Garden"
    },
    "yoga mat": {
        "relevant_products": ["Sustainable Yoga Mat", "Eco Yoga Mat", "Green Yoga Mat"],
        "category": "Sports & Outdoors"
    },
    "shampoo": {
        "relevant_products": ["Renewable Materials Shampoo Cream", "Eco Shampoo", 
                             "Natural Shampoo", "Organic Shampoo"],
        "category": "Beauty & Personal Care"
    },
    "TV": {
        "relevant_products": ["Energy Efficient Smart TV Pro", "Solar Powered LED TV", 
                             "Recycled Materials Smart TV", "Eco TV Pro", "Green TV Max"],
        "category": "Electronics"
    },
    "fridge": {
        "relevant_products": ["Energy Efficient Refrigerator", "Solar Compatible Eco Fridge", 
                             "Sustainable Refrigerator Pro", "Eco Fridge Pro"],
        "category": "Electronics"
    },
    "air conditioner": {
        "relevant_products": ["Energy Efficient Air Conditioner", "Solar Powered AC Unit", 
                             "Eco-Friendly Inverter AC", "Carbon Neutral Air Conditioner"],
        "category": "Electronics"
    },
    "backpack": {
        "relevant_products": ["Sustainable Backpack", "Eco Backpack", "Green Backpack"],
        "category": "Sports & Outdoors"
    },
    "organic coffee": {
        "relevant_products": ["Organic Coffee", "Natural Coffee", "Sustainable Coffee"],
        "category": "Food & Beverages"
    },
    "toothpaste": {
        "relevant_products": ["Sustainable Toothpaste Cream", "Eco Toothpaste", 
                             "Organic Toothpaste", "Natural Toothpaste"],
        "category": "Beauty & Personal Care"
    }
}


def calculate_precision_at_k(relevant_items: Set[str], recommended_items: List[str], k: int) -> float:
    """Calculate Precision@K"""
    if k == 0:
        return 0.0
    recommended_k = recommended_items[:k]
    relevant_in_k = sum(1 for item in recommended_k if item in relevant_items)
    return relevant_in_k / k


def calculate_recall_at_k(relevant_items: Set[str], recommended_items: List[str], k: int) -> float:
    """Calculate Recall@K"""
    if len(relevant_items) == 0:
        return 0.0
    recommended_k = recommended_items[:k]
    relevant_in_k = sum(1 for item in recommended_k if item in relevant_items)
    return relevant_in_k / len(relevant_items)


def calculate_ndcg_at_k(relevant_items: Set[str], recommended_items: List[str], k: int) -> float:
    """Calculate NDCG@K"""
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            rel = 1  # Binary relevance
            dcg += (2**rel - 1) / np.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    num_relevant = min(len(relevant_items), k)
    idcg = sum([(2**1 - 1) / np.log2(i + 2) for i in range(num_relevant)])
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_map_at_k(relevant_items: Set[str], recommended_items: List[str], k: int) -> float:
    """Calculate MAP@K"""
    if len(relevant_items) == 0:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    
    return sum(precisions) / len(relevant_items) if precisions else 0.0


def calculate_mrr(relevant_items: Set[str], recommended_items: List[str]) -> float:
    """Calculate Mean Reciprocal Rank"""
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_recommender(recommender: EcoRecommender, test_queries: Dict) -> Dict:
    """Evaluate recommender system on test queries"""
    results = {
        'precision': {'k5': [], 'k10': [], 'k20': []},
        'recall': {'k5': [], 'k10': [], 'k20': []},
        'ndcg': {'k5': [], 'k10': [], 'k20': []},
        'map': {'k5': [], 'k10': [], 'k20': []},
        'mrr': []
    }
    
    for query, query_info in test_queries.items():
        # Get recommendations
        recommendations = recommender.score_products(query, top_k=20)
        recommended_names = recommendations['name'].tolist()
        
        # Get relevant items
        relevant_names = set(query_info['relevant_products'])
        
        # Also include products from same category with high eco-score as relevant
        category = query_info['category']
        category_products = recommender.filter_by_category(category)
        high_eco = category_products[category_products['eco_score'] > 50]
        for name in high_eco['name'].tolist():
            relevant_names.add(name)
        
        # Calculate metrics for different K values
        for k in [5, 10, 20]:
            results['precision'][f'k{k}'].append(
                calculate_precision_at_k(relevant_names, recommended_names, k)
            )
            results['recall'][f'k{k}'].append(
                calculate_recall_at_k(relevant_names, recommended_names, k)
            )
            results['ndcg'][f'k{k}'].append(
                calculate_ndcg_at_k(relevant_names, recommended_names, k)
            )
            results['map'][f'k{k}'].append(
                calculate_map_at_k(relevant_names, recommended_names, k)
            )
        
        # Calculate MRR
        results['mrr'].append(calculate_mrr(relevant_names, recommended_names))
    
    # Calculate averages
    summary = {
        'precision@5': np.mean(results['precision']['k5']),
        'precision@10': np.mean(results['precision']['k10']),
        'precision@20': np.mean(results['precision']['k20']),
        'recall@5': np.mean(results['recall']['k5']),
        'recall@10': np.mean(results['recall']['k10']),
        'recall@20': np.mean(results['recall']['k20']),
        'ndcg@5': np.mean(results['ndcg']['k5']),
        'ndcg@10': np.mean(results['ndcg']['k10']),
        'ndcg@20': np.mean(results['ndcg']['k20']),
        'map@5': np.mean(results['map']['k5']),
        'map@10': np.mean(results['map']['k10']),
        'map@20': np.mean(results['map']['k20']),
        'mrr': np.mean(results['mrr'])
    }
    
    return summary, results


def print_evaluation_results(summary: Dict):
    """Print formatted evaluation results"""
    print("="*80)
    print("EVALUATION METRICS SUMMARY")
    print("="*80)
    print()
    
    print("Precision@K:")
    print(f"  K=5:  {summary['precision@5']:.3f}")
    print(f"  K=10: {summary['precision@10']:.3f}")
    print(f"  K=20: {summary['precision@20']:.3f}")
    print()
    
    print("Recall@K:")
    print(f"  K=5:  {summary['recall@5']:.3f}")
    print(f"  K=10: {summary['recall@10']:.3f}")
    print(f"  K=20: {summary['recall@20']:.3f}")
    print()
    
    print("NDCG@K:")
    print(f"  K=5:  {summary['ndcg@5']:.3f}")
    print(f"  K=10: {summary['ndcg@10']:.3f}")
    print(f"  K=20: {summary['ndcg@20']:.3f}")
    print()
    
    print("MAP@K:")
    print(f"  K=5:  {summary['map@5']:.3f}")
    print(f"  K=10: {summary['map@10']:.3f}")
    print(f"  K=20: {summary['map@20']:.3f}")
    print()
    
    print(f"MRR: {summary['mrr']:.3f}")
    print()
    print("="*80)


def category_wise_evaluation(recommender: EcoRecommender, test_queries: Dict) -> Dict:
    """Evaluate performance by category"""
    category_results = {}
    
    for query, query_info in test_queries.items():
        category = query_info['category']
        if category not in category_results:
            category_results[category] = {'ndcg': []}
        
        recommendations = recommender.score_products(query, top_k=10)
        recommended_names = recommendations['name'].tolist()
        relevant_names = set(query_info['relevant_products'])
        
        ndcg = calculate_ndcg_at_k(relevant_names, recommended_names, 10)
        category_results[category]['ndcg'].append(ndcg)
    
    # Calculate averages
    category_summary = {}
    for category, metrics in category_results.items():
        category_summary[category] = np.mean(metrics['ndcg'])
    
    return category_summary


if __name__ == "__main__":
    print("Loading recommender system...")
    recommender = EcoRecommender("products_dataset.csv")
    
    print("\nEvaluating recommender system...")
    summary, detailed_results = evaluate_recommender(recommender, TEST_QUERIES)
    
    print_evaluation_results(summary)
    
    print("\nCategory-wise NDCG@10:")
    print("="*80)
    category_summary = category_wise_evaluation(recommender, TEST_QUERIES)
    for category, ndcg in sorted(category_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"{category:25s}: {ndcg:.3f}")
    print("="*80)
    
    # Save results to CSV for LaTeX table generation
    results_df = pd.DataFrame({
        'Metric': ['Precision@5', 'Precision@10', 'Precision@20',
                   'Recall@5', 'Recall@10', 'Recall@20',
                   'NDCG@5', 'NDCG@10', 'NDCG@20',
                   'MAP@5', 'MAP@10', 'MAP@20', 'MRR'],
        'Value': [summary['precision@5'], summary['precision@10'], summary['precision@20'],
                  summary['recall@5'], summary['recall@10'], summary['recall@20'],
                  summary['ndcg@5'], summary['ndcg@10'], summary['ndcg@20'],
                  summary['map@5'], summary['map@10'], summary['map@20'],
                  summary['mrr']]
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nResults saved to evaluation_results.csv")

