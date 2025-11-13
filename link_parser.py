"""
Link parser for extracting product information from URLs.
Includes mock URL metadata parsing and dataset matching utilities.
"""

import re
import urllib.parse
from typing import Dict, Optional, List, Tuple
from urllib.parse import urlparse, parse_qs
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

try:
    from bs4 import BeautifulSoup
    BEAUTIFUL_SOUP_AVAILABLE = True
except ImportError:
    BEAUTIFUL_SOUP_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Using mock parsing only.")


class LinkParser:
    """Parser for extracting product information from URLs."""
    
    def __init__(self, dataset_path: str = "products_dataset.csv"):
        """Initialize link parser with product dataset."""
        self.dataset_path = dataset_path
        self.df = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load product dataset for matching."""
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Loaded {len(self.df)} products for matching")
        except FileNotFoundError:
            print(f"Dataset not found: {self.dataset_path}")
            self.df = pd.DataFrame()
    
    def extract_url_metadata(self, url: str) -> Dict[str, any]:
        """Extract metadata from URL with improved Amazon URL parsing."""
        parsed = urlparse(url)
        
        metadata = {
            'url': url,
            'domain': parsed.netloc,
            'path': parsed.path,
            'query_params': parse_qs(parsed.query),
            'fragment': parsed.fragment
        }
        
        url_lower = url.lower()
        path_parts = [p for p in parsed.path.split('/') if p]
        
        # Amazon-specific URL parsing
        if 'amazon' in parsed.netloc.lower():
            # Amazon URLs often have pattern: /product-name/dp/PRODUCT_ID
            # Extract keywords from the path
            keywords = []
            for part in path_parts:
                # Skip common Amazon path segments
                if part.lower() not in ['dp', 'gp', 'product', 'ref', 's']:
                    # Clean and split the part
                    cleaned = part.replace('-', ' ').replace('_', ' ')
                    keywords.extend(cleaned.split())
            
            # Extract product name from keywords
            if keywords:
                # Filter out common words and keep meaningful ones
                meaningful_keywords = [k for k in keywords if len(k) > 2 and k.lower() not in ['www', 'com', 'in', 'co', 'uk']]
                
                # Add inferred product types based on keywords
                # Check for TV indicators
                if any(k.lower() in ['inches', 'inch', 'lr', 'smart'] for k in meaningful_keywords):
                    if 'tv' not in [k.lower() for k in meaningful_keywords]:
                        meaningful_keywords.insert(0, 'TV')  # Add TV as first keyword
                
                # Check for fridge indicators
                if any(k.lower() in ['refrigerator', 'fridge'] for k in meaningful_keywords):
                    if 'fridge' not in [k.lower() for k in meaningful_keywords]:
                        meaningful_keywords.insert(0, 'fridge')
                
                # Check for AC indicators
                if any(k.lower() in ['ac', 'conditioner', 'cooling'] for k in meaningful_keywords):
                    if 'air conditioner' not in ' '.join([k.lower() for k in meaningful_keywords]):
                        meaningful_keywords.insert(0, 'air conditioner')
                
                if meaningful_keywords:
                    metadata['extracted_name'] = ' '.join(meaningful_keywords[:5])  # Take first 5 keywords
                    metadata['keywords'] = meaningful_keywords
        
        # Generic URL parsing for other sites
        else:
            if path_parts:
                # Common patterns: /product/name, /p/name, /item/name
                if 'product' in path_parts or 'p' in path_parts or 'item' in path_parts:
                    product_name_idx = path_parts.index(next((p for p in path_parts if p in ['product', 'p', 'item']), path_parts[-1]))
                    if product_name_idx + 1 < len(path_parts):
                        metadata['extracted_name'] = path_parts[product_name_idx + 1].replace('-', ' ').replace('_', ' ')
        
        # Extract from query params
        if 'name' in metadata['query_params']:
            metadata['extracted_name'] = metadata['query_params']['name'][0]
        elif 'product' in metadata['query_params']:
            metadata['extracted_name'] = metadata['query_params']['product'][0]
        elif 'title' in metadata['query_params']:
            metadata['extracted_name'] = metadata['query_params']['title'][0]
        
        # Improved category extraction with more keywords
        category_keywords = {
            'Electronics': ['tv', 'television', 'fridge', 'refrigerator', 'ac', 'air conditioner', 
                          'laptop', 'phone', 'tablet', 'speaker', 'charger', 'electronic', 'smart tv',
                          'lg', 'samsung', 'sony', 'panasonic'],
            'Clothing': ['clothing', 'clothes', 'shirt', 'tee', 'dress', 'pants', 'apparel'],
            'Home & Garden': ['home', 'garden', 'furniture', 'appliance', 'kitchen', 'bedroom'],
            'Food & Beverages': ['food', 'beverage', 'drink', 'coffee', 'tea', 'snack'],
            'Beauty & Personal Care': ['beauty', 'personal care', 'shampoo', 'soap', 'cosmetic'],
            'Sports & Outdoors': ['sports', 'outdoor', 'fitness', 'exercise', 'gym'],
            'Books': ['book', 'guide', 'handbook', 'reading'],
            'Toys & Games': ['toy', 'game', 'puzzle', 'board']
        }
        
        url_text = ' '.join([url_lower] + (metadata.get('keywords', []) or []))
        best_match_score = 0
        best_category = 'Unknown'
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in url_text)
            if score > best_match_score:
                best_match_score = score
                best_category = category
        
        metadata['extracted_category'] = best_category
        
        return metadata
    
    def mock_extract_text_from_url(self, url: str) -> Dict[str, str]:
        """Mock function to extract text content from URL (simulates web scraping)."""
        # In real implementation, this would fetch and parse HTML
        # For now, return mock data based on URL patterns
        
        metadata = self.extract_url_metadata(url)
        
        # Build a better product name from extracted keywords
        extracted_name = metadata.get('extracted_name', 'Product from URL')
        keywords = metadata.get('keywords', [])
        
        # If we have keywords, create a more descriptive name
        if keywords:
            # Filter for product-relevant keywords (brands, types, features)
            product_keywords = [k for k in keywords if len(k) > 2]
            if product_keywords:
                extracted_name = ' '.join(product_keywords[:4])  # Use top 4 keywords
        
        mock_text = {
            'title': extracted_name,
            'description': f"{extracted_name} from {metadata['domain']}. This product is available for purchase.",
            'price': None,
            'category': metadata.get('extracted_category', 'Unknown'),
            'keywords': keywords
        }
        
        # Generate better description based on category and keywords
        category = metadata.get('extracted_category', 'Unknown')
        if category != 'Unknown':
            mock_text['description'] = (
                f"{extracted_name} - {category} product available from {metadata['domain']}. "
                f"Check our eco-friendly alternatives below for sustainable options."
            )
        
        return mock_text
    
    def fuzzy_match_product(self, query_text: str, top_k: int = 5, 
                          threshold: int = 60) -> List[Tuple[str, float, Dict]]:
        """Fuzzy match query text against product dataset."""
        if self.df.empty:
            return []
        
        query_lower = query_text.lower()
        
        # Extract product type keywords from query
        product_type_keywords = ['tv', 'television', 'fridge', 'refrigerator', 'ac', 'air conditioner']
        query_product_type = None
        for keyword in product_type_keywords:
            if keyword in query_lower:
                query_product_type = keyword
                break
        
        # Calculate similarity scores for each product
        results = []
        for idx, row in self.df.iterrows():
            # Combine searchable text fields for this product
            searchable_text = (
                str(row['name']) + ' ' +
                str(row.get('description', '')) + ' ' +
                str(row.get('category', ''))
            )
            searchable_lower = searchable_text.lower()
            
            # Calculate fuzzy match score
            score = fuzz.token_sort_ratio(query_text, searchable_text)
            
            # Boost score if product type matches
            if query_product_type:
                if query_product_type in searchable_lower:
                    score += 20  # Boost for product type match
                # Extra boost for exact product type in name
                if query_product_type in str(row['name']).lower():
                    score += 15
            
            # Additional boost for "Smart" keyword if present in query
            if 'smart' in query_lower and 'smart' in searchable_lower:
                score += 10
            
            # Boost for high eco-score products (encourage green alternatives)
            if row.get('eco_score', 0) >= 70:
                score += 5
            
            if score >= threshold:
                product = row.to_dict()
                # Store original score and eco score for sorting
                product['_match_score'] = score
                product['_eco_score'] = product.get('eco_score', 0)
                results.append((searchable_text, score, product))
        
        # Sort by match score first, then by eco score for tie-breaking
        # This ensures high eco-score products appear when match scores are similar
        results.sort(key=lambda x: (x[1], x[2].get('_eco_score', 0)), reverse=True)
        return results[:top_k]
    
    def match_url_to_product(self, url: str, top_k: int = 5) -> List[Dict]:
        """Match URL to products in dataset."""
        # Extract metadata from URL
        metadata = self.extract_url_metadata(url)
        
        # Extract text (mock)
        extracted_text = self.mock_extract_text_from_url(url)
        
        # Build query from extracted information, prioritizing keywords
        query_parts = []
        
        # Use keywords if available (better for Amazon URLs)
        keywords = metadata.get('keywords', []) or extracted_text.get('keywords', [])
        if keywords:
            # Filter for product-relevant keywords
            product_keywords = [k for k in keywords if len(k) > 2 and k.lower() not in ['www', 'com', 'in', 'co', 'uk', 'dp', 'gp']]
            if product_keywords:
                query_parts.extend(product_keywords[:5])  # Use top 5 keywords
        
        # Add extracted name if available
        if 'extracted_name' in metadata and metadata['extracted_name']:
            query_parts.append(metadata['extracted_name'])
        
        # Add category for better matching
        category = extracted_text.get('category') or metadata.get('extracted_category')
        if category and category != 'Unknown':
            query_parts.append(category)
        
        # Build final query
        query = ' '.join(query_parts) if query_parts else url
        
        # Lower threshold for URL matching to catch more potential matches
        matches = self.fuzzy_match_product(query, top_k=top_k, threshold=40)
        
        # Format results
        results = []
        for match_text, score, product in matches:
            product['match_score'] = score
            product['match_text'] = match_text
            results.append(product)
        
        return results
    
    def estimate_eco_score_for_url(self, url: str, extracted_data: Dict) -> float:
        """Estimate eco score for a product from URL (when not in catalog)."""
        # Base score
        base_score = 50.0
        
        # Check for eco keywords in URL and extracted data
        eco_keywords = [
            'organic', 'eco', 'green', 'sustainable', 'recycled',
            'biodegradable', 'fair trade', 'carbon neutral', 'renewable'
        ]
        
        url_lower = url.lower()
        text_lower = ' '.join([
            str(extracted_data.get('title', '')),
            str(extracted_data.get('description', ''))
        ]).lower()
        
        combined_text = url_lower + ' ' + text_lower
        
        # Add points for each eco keyword found
        keyword_bonus = sum(10 for keyword in eco_keywords if keyword in combined_text)
        
        # Category bonus
        category = extracted_data.get('category', '')
        category_bonuses = {
            'Food & Beverages': 15,
            'Home & Garden': 12,
            'Clothing': 10,
            'Beauty & Personal Care': 8,
            'Electronics': 5
        }
        category_bonus = category_bonuses.get(category, 0)
        
        estimated_score = min(100.0, base_score + keyword_bonus + category_bonus)
        return round(estimated_score, 2)
    
    def parse_and_match(self, url: str) -> Dict:
        """Parse URL and match to products, with fallback estimation."""
        # Try to match to existing products - get more matches for alternatives
        matches = self.match_url_to_product(url, top_k=10)
        
        if matches and matches[0]['match_score'] >= 80:
            # Good match found - return top match and all other matches as alternatives
            return {
                'matched': True,
                'product': matches[0],
                'alternatives': matches[1:9] if len(matches) > 1 else [],  # Return up to 9 alternatives
                'estimated': False
            }
        else:
            # No good match, estimate from URL
            metadata = self.extract_url_metadata(url)
            extracted_text = self.mock_extract_text_from_url(url)
            
            estimated_score = self.estimate_eco_score_for_url(url, extracted_text)
            
            return {
                'matched': False,
                'product': {
                    'product_id': 'URL_PRODUCT',
                    'name': extracted_text.get('title', 'Product from URL'),
                    'category': extracted_text.get('category', 'Unknown'),
                    'description': extracted_text.get('description', ''),
                    'eco_score': estimated_score,
                    'url': url
                },
                'alternatives': matches[:9] if matches else [],  # Return up to 9 alternatives
                'estimated': True
            }


if __name__ == "__main__":
    # Test link parser
    print("Testing Link Parser...")
    
    parser = LinkParser("products_dataset.csv")
    
    # Test URLs
    test_urls = [
        "https://example.com/product/organic-cotton-shirt",
        "https://shop.com/p/eco-friendly-laptop",
        "https://store.com/item/sustainable-coffee-beans?category=food",
        "https://marketplace.com/products/green-energy-device"
    ]
    
    for url in test_urls:
        print("\n" + "="*80)
        print(f"Testing URL: {url}")
        print("="*80)
        
        result = parser.parse_and_match(url)
        
        if result['matched']:
            print(f"✓ Matched to product: {result['product']['name']}")
            print(f"  Match score: {result['product']['match_score']}")
            print(f"  Eco score: {result['product']['eco_score']}")
        else:
            print(f"✗ No match found, estimated product")
            print(f"  Estimated name: {result['product']['name']}")
            print(f"  Estimated eco score: {result['product']['eco_score']}")
        
        if result['alternatives']:
            print(f"\n  Alternatives ({len(result['alternatives'])}):")
            for alt in result['alternatives'][:2]:
                print(f"    - {alt['name']} (score: {alt.get('match_score', 'N/A')})")
