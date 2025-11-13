"""
Streamlit app for EcoShopper Bot.
Interactive UI for product queries, URL parsing, recommendations, and visualizations.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import sys
import os

# Import our modules
from recommender import EcoRecommender
from llm_engine import LLMEngine
from link_parser import LinkParser

# Configure matplotlib for Streamlit
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_page():
    """Configure Streamlit page."""
    st.set_page_config(
        page_title="EcoShopper Bot",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def plot_eco_score_distribution(df: pd.DataFrame):
    """Plot distribution of eco scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['eco_score'], bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Eco Score', fontsize=12)
    ax.set_ylabel('Number of Products', fontsize=12)
    ax.set_title('Distribution of Eco Scores', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_category_eco_scores(df: pd.DataFrame):
    """Plot average eco scores by category."""
    category_scores = df.groupby('category')['eco_score'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(category_scores.index, category_scores.values, color='teal', alpha=0.7)
    ax.set_xlabel('Average Eco Score', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title('Average Eco Scores by Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, val) in enumerate(category_scores.items()):
        ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_top_products(df: pd.DataFrame, top_n: int = 10):
    """Plot top N products by eco score."""
    top_products = df.nlargest(top_n, 'eco_score')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_products)), top_products['eco_score'].values, color='darkgreen', alpha=0.7)
    ax.set_yticks(range(len(top_products)))
    ax.set_yticklabels([name[:40] + '...' if len(name) > 40 else name for name in top_products['name']], fontsize=9)
    ax.set_xlabel('Eco Score', fontsize=12)
    ax.set_title(f'Top {top_n} Products by Eco Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, val in enumerate(top_products['eco_score'].values):
        ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def display_recommendations(recommendations: pd.DataFrame, llm_engine: LLMEngine, 
                          query: Optional[str] = None, show_explanations: bool = True):
    """Display recommendations with explanations."""
    st.subheader("🌿 Recommended Products")
    
    if recommendations.empty:
        st.warning("No recommendations found.")
        return
    
    for idx, row in recommendations.iterrows():
        with st.expander(f"**{row['name']}** - Eco Score: {row['eco_score']:.1f}/100", expanded=(idx == 0)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Brand:** {row['brand']}")
                st.write(f"**Price:** ${row['price']:.2f}")
                st.write(f"**Rating:** {row['rating']:.1f} ⭐")
                st.write(f"**Stock:** {row['stock']} units")
                st.write(f"**Eco Attributes:** {row['eco_attributes']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Similarity Score:** {row.get('combined_score', 0):.3f}")
            
            with col2:
                st.metric("Eco Score", f"{row['eco_score']:.1f}", delta=f"{row['eco_score'] - 50:.1f}")
            
            if show_explanations:
                st.markdown("---")
                st.write("**🤖 AI Explanation:**")
                product_dict = row.to_dict()
                explanation = llm_engine.generate_explanation(product_dict, query)
                st.info(explanation)


def main_streamlit():
    """Main Streamlit application."""
    setup_page()
    
    st.title("🌱 EcoShopper Bot")
    st.markdown("### Your AI-Powered Eco-Friendly Shopping Assistant")
    
    # Initialize components
    if 'recommender' not in st.session_state:
        with st.spinner("Loading recommender system..."):
            try:
                st.session_state.recommender = EcoRecommender("products_dataset.csv")
            except FileNotFoundError:
                st.error("Dataset not found! Please run dataset_generator.py first.")
                st.stop()
    
    if 'llm_engine' not in st.session_state:
        # Try to get API key from secrets, but use default if not available
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except:
            # Secrets not configured, use default from LLMEngine
            api_key = None
        st.session_state.llm_engine = LLMEngine(api_key=api_key)
    
    if 'link_parser' not in st.session_state:
        st.session_state.link_parser = LinkParser("products_dataset.csv")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.radio(
            "Select Mode",
            ["Search Products", "Parse URL", "View Analytics"]
        )
        
        show_explanations = st.checkbox("Show AI Explanations", value=True)
        top_k = st.slider("Number of Recommendations", 3, 20, 10)
        
        st.markdown("---")
        st.header("📊 Quick Stats")
        df = st.session_state.recommender.df
        st.metric("Total Products", len(df))
        st.metric("Average Eco Score", f"{df['eco_score'].mean():.1f}")
        st.metric("Categories", df['category'].nunique())
    
    # Main content based on mode
    if mode == "Search Products":
        st.header("🔍 Search for Eco-Friendly Products")
        
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., organic cotton clothing, sustainable electronics..."
        )
        
        if st.button("Search", type="primary") or query:
            if query:
                with st.spinner("Finding the best eco-friendly products..."):
                    recommendations = st.session_state.recommender.score_products(query, top_k=top_k)
                    display_recommendations(
                        recommendations, 
                        st.session_state.llm_engine, 
                        query=query,
                        show_explanations=show_explanations
                    )
            else:
                st.info("Enter a search query to get recommendations.")
    
    elif mode == "Parse URL":
        st.header("🔗 Parse Product URL")
        
        url = st.text_input(
            "Enter product URL:",
            placeholder="https://example.com/product/..."
        )
        
        if st.button("Parse & Match", type="primary") or url:
            if url:
                with st.spinner("Parsing URL and finding matches..."):
                    result = st.session_state.link_parser.parse_and_match(url)
                    
                    if result['matched']:
                        st.success(f"✓ Matched to product in database!")
                        product = result['product']
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Name:** {product['name']}")
                            st.write(f"**Category:** {product['category']}")
                            st.write(f"**Eco Score:** {product['eco_score']:.1f}/100")
                            st.write(f"**Match Score:** {product['match_score']:.1f}%")
                        with col2:
                            st.metric("Eco Score", f"{product['eco_score']:.1f}")
                        
                        if show_explanations:
                            st.markdown("---")
                            st.write("**🤖 AI Explanation:**")
                            explanation = st.session_state.llm_engine.generate_explanation(product)
                            st.info(explanation)
                        
                        # Show alternatives list
                        if result.get('alternatives'):
                            st.markdown("---")
                            st.subheader("🌿 Alternative Eco-Friendly Products")
                            st.write(f"Found {len(result['alternatives'])} similar eco-friendly alternatives:")
                            
                            for i, alt in enumerate(result['alternatives'], 1):
                                with st.expander(f"**{i}. {alt['name']}** - Eco Score: {alt['eco_score']:.1f}/100 (Match: {alt.get('match_score', 0):.1f}%)", expanded=(i == 1)):
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        st.write(f"**Category:** {alt.get('category', 'N/A')}")
                                        st.write(f"**Brand:** {alt.get('brand', 'N/A')}")
                                        st.write(f"**Price:** ${alt.get('price', 0):.2f}" if alt.get('price') else "**Price:** N/A")
                                        st.write(f"**Eco Attributes:** {alt.get('eco_attributes', 'N/A')}")
                                    with col2:
                                        st.metric("Eco Score", f"{alt['eco_score']:.1f}")
                                    
                                    if show_explanations:
                                        st.markdown("---")
                                        st.write("**🤖 AI Explanation:**")
                                        alt_explanation = st.session_state.llm_engine.generate_explanation(alt)
                                        st.info(alt_explanation)
                    else:
                        st.warning("⚠️ Product not found in database. Estimated information:")
                        product = result['product']
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Estimated Name:** {product['name']}")
                            st.write(f"**Category:** {product['category']}")
                            st.write(f"**Estimated Eco Score:** {product['eco_score']:.1f}/100")
                        with col2:
                            st.metric("Eco Score", f"{product['eco_score']:.1f}")
                        
                        if result['alternatives']:
                            st.markdown("---")
                            st.write("**Similar products in database:**")
                            for alt in result['alternatives']:
                                st.write(f"- {alt['name']} (match: {alt.get('match_score', 'N/A')}%)")
            else:
                st.info("Enter a URL to parse and match.")
    
    elif mode == "View Analytics":
        st.header("📊 Product Analytics")
        
        df = st.session_state.recommender.df
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Products", len(df))
        with col2:
            st.metric("Avg Eco Score", f"{df['eco_score'].mean():.1f}")
        with col3:
            st.metric("Categories", df['category'].nunique())
        with col4:
            st.metric("Top Score", f"{df['eco_score'].max():.1f}")
        
        st.markdown("---")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Eco Score Distribution")
            fig1 = plot_eco_score_distribution(df)
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            st.subheader("Average Scores by Category")
            fig2 = plot_category_eco_scores(df)
            st.pyplot(fig2)
            plt.close(fig2)
        
        st.markdown("---")
        st.subheader("Top Products by Eco Score")
        top_n = st.slider("Number of top products", 5, 20, 10, key="top_n_slider")
        fig3 = plot_top_products(df, top_n=top_n)
        st.pyplot(fig3)
        plt.close(fig3)
        
        # Data table
        st.markdown("---")
        st.subheader("Product Data")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=csv,
            file_name="products_dataset.csv",
            mime="text/csv"
        )


def main_cli():
    """CLI fallback when Streamlit is not available."""
    print("="*80)
    print("EcoShopper Bot - CLI Mode")
    print("="*80)
    print("\nThis is a CLI fallback. For full functionality, run with Streamlit:")
    print("  streamlit run app_demo.py")
    print("\n" + "="*80)
    
    # Basic CLI functionality
    try:
        recommender = EcoRecommender("products_dataset.csv")
        llm_engine = LLMEngine()
        
        print("\nEnter a search query (or 'quit' to exit):")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                print(f"\nSearching for: {query}")
                results = recommender.score_products(query, top_k=5)
                print("\nTop 5 Results:")
                print(results[['name', 'category', 'eco_score', 'combined_score']].to_string())
    except FileNotFoundError:
        print("Error: Dataset not found. Please run dataset_generator.py first.")


# When Streamlit runs this file, it executes the entire script
# We call main_streamlit() here so it runs automatically
# The CLI code is only for direct Python execution (python app_demo.py)
if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT"):
    # Running under Streamlit - call main_streamlit
    main_streamlit()
elif __name__ == "__main__":
    # Running directly with Python - offer CLI mode
    try:
        import streamlit
        print("To run the Streamlit app, use: streamlit run app_demo.py")
        print("Running in CLI mode instead...")
        main_cli()
    except ImportError:
        print("Streamlit not available, running in CLI mode...")
        main_cli()
