"""
LLM engine for generating eco-friendly product explanations using Google Gemini API.
Includes caching stub and offline mode fallback.
"""

import os
import json
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Using offline mode.")


class LLMEngine:
    """LLM engine for generating product explanations."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """Initialize LLM engine."""
        # Default API key if none provided
        default_api_key = "AIzaSyCSU1EyM7HFHOkmgpkuw4-mhSS1PhJqQ18"
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or default_api_key
        self.model_name = model_name
        self.model = None
        self.cache = {}  # Simple in-memory cache
        self.offline_mode = False
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                print(f"LLM Engine initialized with {model_name}")
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")
                print("Falling back to offline mode")
                self.offline_mode = True
        else:
            print("LLM Engine running in offline mode (no API key or library)")
            self.offline_mode = True
    
    def build_prompt(self, product_data: Dict[str, Any], query: Optional[str] = None) -> str:
        """Build prompt for LLM explanation."""
        prompt_parts = [
            "You are an eco-friendly shopping assistant. Explain why this product is recommended:",
            "",
            f"Product Name: {product_data.get('name', 'N/A')}",
            f"Category: {product_data.get('category', 'N/A')}",
            f"Brand: {product_data.get('brand', 'N/A')}",
            f"Eco Score: {product_data.get('eco_score', 'N/A')}/100",
            f"Eco Attributes: {product_data.get('eco_attributes', 'N/A')}",
            f"Description: {product_data.get('description', 'N/A')}",
        ]
        
        if query:
            prompt_parts.append(f"\nUser Query: {query}")
        
        prompt_parts.extend([
            "",
            "Provide a concise, friendly explanation (2-3 sentences) highlighting:",
            "1. Why this product matches the user's needs",
            "2. Key eco-friendly benefits",
            "3. Why it's a good choice for sustainable shopping",
        ])
        
        return "\n".join(prompt_parts)
    
    def generate_explanation(self, product_data: Dict[str, Any], query: Optional[str] = None, 
                           use_cache: bool = True) -> str:
        """Generate explanation for a product recommendation."""
        # Check cache
        cache_key = f"{product_data.get('product_id', 'unknown')}_{query or 'no_query'}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.offline_mode:
            explanation = self._generate_offline_explanation(product_data, query)
        else:
            try:
                prompt = self.build_prompt(product_data, query)
                response = self.model.generate_content(prompt)
                explanation = response.text.strip()
            except Exception as e:
                # Handle rate limits and other API errors gracefully
                error_msg = str(e)
                if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    print(f"Rate limit reached, using offline explanation")
                elif "429" in error_msg:
                    print(f"API quota exceeded, using offline explanation")
                else:
                    print(f"API error: {error_msg[:100]}...")
                explanation = self._generate_offline_explanation(product_data, query)
        
        # Cache the result
        if use_cache:
            self.cache[cache_key] = explanation
        
        return explanation
    
    def _generate_offline_explanation(self, product_data: Dict[str, Any], 
                                     query: Optional[str] = None) -> str:
        """Generate explanation in offline mode (fallback)."""
        name = product_data.get('name', 'This product')
        category = product_data.get('category', 'product')
        eco_score = product_data.get('eco_score', 0)
        eco_attrs = product_data.get('eco_attributes', 'eco-friendly features')
        
        explanation_parts = [
            f"{name} is an excellent eco-friendly choice in the {category} category."
        ]
        
        if eco_score >= 80:
            explanation_parts.append(
                f"With an eco score of {eco_score}/100, it demonstrates strong commitment to sustainability."
            )
        elif eco_score >= 60:
            explanation_parts.append(
                f"With an eco score of {eco_score}/100, it offers good environmental credentials."
            )
        else:
            explanation_parts.append(
                f"With an eco score of {eco_score}/100, it includes some eco-friendly features."
            )
        
        explanation_parts.append(
            f"Key benefits include: {eco_attrs}. This makes it a responsible choice for conscious consumers."
        )
        
        if query:
            explanation_parts.insert(1, f"It aligns well with your search for '{query}'.")
        
        return " ".join(explanation_parts)
    
    def generate_batch_explanations(self, products: List[Dict[str, Any]], 
                                   query: Optional[str] = None) -> List[str]:
        """Generate explanations for multiple products."""
        explanations = []
        for product in products:
            explanation = self.generate_explanation(product, query)
            explanations.append(explanation)
            # Small delay to avoid rate limiting
            if not self.offline_mode:
                time.sleep(0.1)
        return explanations
    
    def clear_cache(self):
        """Clear the explanation cache."""
        self.cache.clear()
        print("Cache cleared")
    
    def save_cache(self, cache_path: str = "llm_cache.json"):
        """Save cache to file."""
        with open(cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)
        print(f"Cache saved to {cache_path}")
    
    def load_cache(self, cache_path: str = "llm_cache.json"):
        """Load cache from file."""
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.cache = json.load(f)
            print(f"Cache loaded from {cache_path}")
        else:
            print(f"Cache file not found: {cache_path}")


def get_default_prompt_template() -> str:
    """Get default prompt template for explanations."""
    return """You are an eco-friendly shopping assistant. Explain why this product is recommended:

Product Name: {name}
Category: {category}
Brand: {brand}
Eco Score: {eco_score}/100
Eco Attributes: {eco_attributes}
Description: {description}

{query_section}

Provide a concise, friendly explanation (2-3 sentences) highlighting:
1. Why this product matches the user's needs
2. Key eco-friendly benefits
3. Why it's a good choice for sustainable shopping
"""


if __name__ == "__main__":
    # Test LLM engine
    print("Testing LLM Engine...")
    
    # Initialize LLM with default API key
    llm = LLMEngine()
    
    sample_product = {
        'product_id': 'PROD_0001',
        'name': 'Organic Cotton T-Shirt',
        'category': 'Clothing',
        'brand': 'EcoBrand',
        'eco_score': 85.5,
        'eco_attributes': 'Organic, Fair Trade, Biodegradable',
        'description': 'Organic Cotton T-Shirt from EcoBrand. Category: Clothing. Features: Organic, Fair Trade, Biodegradable.'
    }
    
    print("\n" + "="*80)
    if llm.offline_mode:
        print("Sample Product Explanation (Offline Mode)")
    else:
        print(f"Sample Product Explanation (Using {llm.model_name})")
    print("="*80)
    
    try:
        explanation = llm.generate_explanation(sample_product, query="sustainable clothing")
        print(explanation)
        print(f"\n✓ Successfully generated explanation ({len(explanation)} characters)")
    except Exception as e:
        print(f"✗ Error generating explanation: {e}")
    
    # Show status
    print("\n" + "="*80)
    print("LLM Engine Status:")
    print("="*80)
    print(f"Model: {llm.model_name}")
    print(f"Offline Mode: {llm.offline_mode}")
    print(f"Model Initialized: {llm.model is not None}")
    print(f"API Key Set: {bool(llm.api_key)}")
