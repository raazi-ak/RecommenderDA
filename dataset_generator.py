"""
Synthetic dataset generator for EcoShopper Bot.
Generates product data with eco-friendly attributes, categories, and metadata.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import random

# Seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Product categories and eco attributes
CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Food & Beverages",
    "Beauty & Personal Care", "Sports & Outdoors", "Books", "Toys & Games"
]

ECO_ATTRIBUTES = [
    "Organic", "Recycled", "Biodegradable", "Energy Efficient",
    "Fair Trade", "Locally Sourced", "Carbon Neutral", "Minimal Packaging",
    "Renewable Materials", "Water Efficient", "Non-Toxic", "Sustainable"
]

BRANDS = [
    "EcoBrand", "GreenLife", "SustainableCo", "EarthFriendly", "NatureFirst",
    "PureEarth", "EcoTech", "GreenWave", "SustainableStyle", "EcoEssentials"
]

PRODUCT_NAMES_TEMPLATES = {
    "Electronics": ["Smart {attr} {item}", "Eco {item} Pro", "Green {item} Max"],
    "Clothing": ["{attr} {item} Shirt", "Sustainable {item}", "Organic {item} Tee"],
    "Home & Garden": ["{attr} {item} Set", "Eco {item}", "Green {item} Kit"],
    "Food & Beverages": ["{attr} {item}", "Organic {item}", "Natural {item}"],
    "Beauty & Personal Care": ["{attr} {item} Cream", "Natural {item}", "Eco {item}"],
    "Sports & Outdoors": ["{attr} {item} Gear", "Sustainable {item}", "Eco {item}"],
    "Books": ["{attr} Guide", "Sustainable Living", "Eco Handbook"],
    "Toys & Games": ["{attr} {item} Set", "Eco {item}", "Green {item}"]
}

ITEMS = {
    "Electronics": ["Phone", "Laptop", "Tablet", "Speaker", "Charger", "TV", "Television", "Fridge", "Refrigerator", "AC", "Air Conditioner"],
    "Clothing": ["Cotton", "Bamboo", "Hemp", "Linen", "Wool"],
    "Home & Garden": ["Garden", "Compost", "Solar", "LED", "Water", "AC", "Air Conditioner", "Fridge", "Refrigerator"],
    "Food & Beverages": ["Coffee", "Tea", "Snacks", "Juice", "Cereal"],
    "Beauty & Personal Care": ["Shampoo", "Soap", "Lotion", "Toothpaste", "Deodorant"],
    "Sports & Outdoors": ["Backpack", "Water Bottle", "Yoga Mat", "Running", "Camping"],
    "Books": ["Sustainability", "Climate", "Ecology", "Green Living"],
    "Toys & Games": ["Building Blocks", "Puzzle", "Board Game", "Art Supplies"]
}


def generate_product_description(category: str, brand: str, name: str, eco_attrs: List[str]) -> str:
    """Generate a product description based on attributes."""
    desc_parts = [
        f"{name} from {brand}.",
        f"Category: {category}.",
        f"Features: {', '.join(eco_attrs[:3])}."
    ]
    if len(eco_attrs) > 3:
        desc_parts.append(f"Additional benefits: {', '.join(eco_attrs[3:])}.")
    return " ".join(desc_parts)


def calculate_eco_score(eco_attrs: List[str], category: str) -> float:
    """Calculate eco score based on attributes and category."""
    base_score = len(eco_attrs) * 8.0
    category_bonus = {
        "Electronics": 5.0,
        "Clothing": 10.0,
        "Home & Garden": 12.0,
        "Food & Beverages": 15.0,
        "Beauty & Personal Care": 8.0,
        "Sports & Outdoors": 10.0,
        "Books": 5.0,
        "Toys & Games": 7.0
    }
    bonus = category_bonus.get(category, 5.0)
    score = min(100.0, base_score + bonus + np.random.normal(0, 5))
    return round(max(0.0, score), 2)


def generate_green_appliances() -> List[Dict]:
    """Generate specific green alternatives for TVs, fridges, and ACs."""
    green_appliances = [
        # TVs
        {
            "name": "Energy Efficient Smart TV Pro",
            "category": "Electronics",
            "brand": "EcoTech",
            "description": "Energy Efficient Smart TV Pro from EcoTech. Category: Electronics. Features: Energy Efficient, Carbon Neutral, Sustainable. Additional benefits: Low Power Consumption, LED Backlight Technology.",
            "eco_attributes": "Energy Efficient, Carbon Neutral, Sustainable, Low Power Consumption",
            "eco_score": 72.5,
            "price": 899.99,
            "rating": 4.6,
            "stock": 45
        },
        {
            "name": "Solar Powered LED TV",
            "category": "Electronics",
            "brand": "GreenWave",
            "description": "Solar Powered LED TV from GreenWave. Category: Electronics. Features: Energy Efficient, Renewable Materials, Carbon Neutral. Additional benefits: Solar Compatible, Ultra Low Power Mode.",
            "eco_attributes": "Energy Efficient, Renewable Materials, Carbon Neutral, Solar Compatible",
            "eco_score": 78.3,
            "price": 1299.99,
            "rating": 4.7,
            "stock": 32
        },
        {
            "name": "Recycled Materials Smart TV",
            "category": "Electronics",
            "brand": "SustainableCo",
            "description": "Recycled Materials Smart TV from SustainableCo. Category: Electronics. Features: Recycled, Energy Efficient, Sustainable. Additional benefits: Made from 80% Recycled Materials, Energy Star Certified.",
            "eco_attributes": "Recycled, Energy Efficient, Sustainable, Energy Star Certified",
            "eco_score": 75.8,
            "price": 799.99,
            "rating": 4.5,
            "stock": 28
        },
        # Fridges
        {
            "name": "Energy Efficient Refrigerator",
            "category": "Electronics",
            "brand": "EcoTech",
            "description": "Energy Efficient Refrigerator from EcoTech. Category: Electronics. Features: Energy Efficient, Carbon Neutral, Water Efficient. Additional benefits: Energy Star A+++ Rating, Low Noise Operation.",
            "eco_attributes": "Energy Efficient, Carbon Neutral, Water Efficient, Energy Star A+++",
            "eco_score": 81.2,
            "price": 1499.99,
            "rating": 4.8,
            "stock": 38
        },
        {
            "name": "Solar Compatible Eco Fridge",
            "category": "Electronics",
            "brand": "GreenLife",
            "description": "Solar Compatible Eco Fridge from GreenLife. Category: Electronics. Features: Energy Efficient, Renewable Materials, Carbon Neutral. Additional benefits: Solar Power Compatible, Ultra Efficient Cooling.",
            "eco_attributes": "Energy Efficient, Renewable Materials, Carbon Neutral, Solar Compatible",
            "eco_score": 83.7,
            "price": 1799.99,
            "rating": 4.9,
            "stock": 25
        },
        {
            "name": "Sustainable Refrigerator Pro",
            "category": "Electronics",
            "brand": "EarthFriendly",
            "description": "Sustainable Refrigerator Pro from EarthFriendly. Category: Electronics. Features: Sustainable, Energy Efficient, Non-Toxic. Additional benefits: CFC-Free Refrigerant, Recyclable Materials.",
            "eco_attributes": "Sustainable, Energy Efficient, Non-Toxic, CFC-Free",
            "eco_score": 79.5,
            "price": 1399.99,
            "rating": 4.7,
            "stock": 30
        },
        # Air Conditioners
        {
            "name": "Energy Efficient Air Conditioner",
            "category": "Electronics",
            "brand": "EcoTech",
            "description": "Energy Efficient Air Conditioner from EcoTech. Category: Electronics. Features: Energy Efficient, Carbon Neutral, Water Efficient. Additional benefits: Inverter Technology, Eco-Friendly Refrigerant.",
            "eco_attributes": "Energy Efficient, Carbon Neutral, Water Efficient, Inverter Technology",
            "eco_score": 76.4,
            "price": 899.99,
            "rating": 4.6,
            "stock": 42
        },
        {
            "name": "Solar Powered AC Unit",
            "category": "Electronics",
            "brand": "GreenWave",
            "description": "Solar Powered AC Unit from GreenWave. Category: Electronics. Features: Energy Efficient, Renewable Materials, Carbon Neutral. Additional benefits: Solar Compatible, Ultra Quiet Operation.",
            "eco_attributes": "Energy Efficient, Renewable Materials, Carbon Neutral, Solar Compatible",
            "eco_score": 82.1,
            "price": 1299.99,
            "rating": 4.8,
            "stock": 35
        },
        {
            "name": "Eco-Friendly Inverter AC",
            "category": "Electronics",
            "brand": "SustainableCo",
            "description": "Eco-Friendly Inverter AC from SustainableCo. Category: Electronics. Features: Energy Efficient, Sustainable, Non-Toxic. Additional benefits: R-32 Refrigerant, Smart Energy Management.",
            "eco_attributes": "Energy Efficient, Sustainable, Non-Toxic, R-32 Refrigerant",
            "eco_score": 77.9,
            "price": 999.99,
            "rating": 4.7,
            "stock": 40
        },
        {
            "name": "Carbon Neutral Air Conditioner",
            "category": "Home & Garden",
            "brand": "GreenLife",
            "description": "Carbon Neutral Air Conditioner from GreenLife. Category: Home & Garden. Features: Carbon Neutral, Energy Efficient, Sustainable. Additional benefits: Carbon Offset Program, High Efficiency Rating.",
            "eco_attributes": "Carbon Neutral, Energy Efficient, Sustainable, Carbon Offset",
            "eco_score": 80.6,
            "price": 1199.99,
            "rating": 4.8,
            "stock": 33
        }
    ]
    
    # Add product IDs
    for i, appliance in enumerate(green_appliances):
        appliance["product_id"] = f"GREEN_{i+1:03d}"
    
    return green_appliances


def generate_product_data(n_products: int = 500) -> pd.DataFrame:
    """Generate synthetic product dataset."""
    products = []
    
    for i in range(n_products):
        category = random.choice(CATEGORIES)
        brand = random.choice(BRANDS)
        
        # Select 2-5 eco attributes
        n_attrs = random.randint(2, 5)
        eco_attrs = random.sample(ECO_ATTRIBUTES, n_attrs)
        
        # Generate product name
        template = random.choice(PRODUCT_NAMES_TEMPLATES[category])
        item = random.choice(ITEMS[category])
        attr = random.choice(eco_attrs)
        name = template.format(attr=attr, item=item)
        
        # Generate description
        description = generate_product_description(category, brand, name, eco_attrs)
        
        # Calculate eco score
        eco_score = calculate_eco_score(eco_attrs, category)
        
        # Additional metadata
        price = round(np.random.lognormal(3.5, 0.8), 2)
        rating = round(np.random.uniform(3.5, 5.0), 1)
        stock = random.randint(0, 100)
        
        products.append({
            "product_id": f"PROD_{i+1:04d}",
            "name": name,
            "category": category,
            "brand": brand,
            "description": description,
            "eco_attributes": ", ".join(eco_attrs),
            "eco_score": eco_score,
            "price": price,
            "rating": rating,
            "stock": stock
        })
    
    df = pd.DataFrame(products)
    return df


def save_dataset(df: pd.DataFrame, output_path: str = "products_dataset.csv"):
    """Save dataset to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    return output_path


def preview_dataset(df: pd.DataFrame, n_samples: int = 5):
    """Print preview of dataset."""
    print("\n" + "="*80)
    print("Dataset Preview")
    print("="*80)
    print(f"\nTotal products: {len(df)}")
    print(f"\nFirst {n_samples} products:\n")
    print(df.head(n_samples).to_string())
    print("\n" + "="*80)
    print("\nDataset Statistics:")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Brands: {df['brand'].nunique()}")
    print(f"Eco Score Range: {df['eco_score'].min():.2f} - {df['eco_score'].max():.2f}")
    print(f"Average Eco Score: {df['eco_score'].mean():.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic product dataset...")
    df = generate_product_data(n_products=500)
    
    # Add green alternatives for TVs, fridges, and ACs
    print("\nAdding green alternatives for TVs, fridges, and ACs...")
    green_appliances = generate_green_appliances()
    green_df = pd.DataFrame(green_appliances)
    
    # Append to main dataset
    df = pd.concat([df, green_df], ignore_index=True)
    print(f"Added {len(green_appliances)} green appliance alternatives")
    
    # Save to CSV
    save_dataset(df, "products_dataset.csv")
    
    # Preview
    preview_dataset(df, n_samples=5)
    
    # Show green appliances
    print("\n" + "="*80)
    print("Green Appliance Alternatives Added:")
    print("="*80)
    print(green_df[['name', 'category', 'eco_score', 'price']].to_string(index=False))
    print("="*80 + "\n")
