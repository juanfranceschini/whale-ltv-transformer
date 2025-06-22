"""
Data preparation module for Whale LTV Transformer.

Processes Olist e-commerce data to create customer-level event sequences
and labels for LTV prediction and whale classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OlistDataProcessor:
    """Process Olist e-commerce data into customer event sequences."""
    
    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all Olist CSV files."""
        logger.info("Loading Olist dataset files...")
        
        files = {
            'customers': 'olist_customers_dataset.csv',
            'orders': 'olist_orders_dataset.csv', 
            'order_items': 'olist_order_items_dataset.csv',
            'order_payments': 'olist_order_payments_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'geolocation': 'olist_geolocation_dataset.csv',
            'order_reviews': 'olist_order_reviews_dataset.csv',
            'product_category_translation': 'product_category_name_translation.csv'
        }
        
        data = {}
        for key, filename in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                data[key] = pd.read_csv(filepath)
                logger.info(f"Loaded {key}: {data[key].shape}")
            else:
                logger.warning(f"File not found: {filepath}")
                
        return data
    
    def create_customer_sequences(
        self, 
        data: Dict[str, pd.DataFrame],
        early_window_days: int = 14,
        max_orders: int = 3
    ) -> pd.DataFrame:
        """Create customer event sequences from early window data."""
        customers = data['customers']
        orders = data['orders']
        order_items = data['order_items']
        order_payments = data['order_payments']
        
        # Convert timestamps
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        customers['customer_zip_code_prefix'] = customers['customer_zip_code_prefix'].astype(str)
        
        # Merge customer and order data
        customer_orders = customers.merge(orders, on='customer_id', how='inner')
        
        # Add order items and payments
        customer_orders = customer_orders.merge(order_items, on='order_id', how='left')
        customer_orders = customer_orders.merge(order_payments, on='order_id', how='left')
        
        # Calculate order values and dates
        customer_orders['order_value'] = customer_orders['price'] + customer_orders['freight_value']
        customer_orders['order_date'] = customer_orders['order_purchase_timestamp'].dt.date
        
        # Group by customer and create sequences
        sequences = []
        
        for customer_id, customer_data in customer_orders.groupby('customer_id'):
            # Sort by order date
            customer_data = customer_data.sort_values('order_purchase_timestamp')
            
            # Get first order date
            first_order_date = customer_data['order_purchase_timestamp'].min()
            
            # Filter to early window
            early_orders = customer_data[
                customer_data['order_purchase_timestamp'] <= first_order_date + pd.Timedelta(days=early_window_days)
            ]
            
            # Limit to max orders
            if len(early_orders) > max_orders:
                early_orders = early_orders.head(max_orders)
            
            # Create events list
            events = []
            for idx, (_, order) in enumerate(early_orders.iterrows(), 1):
                days_since_signup = (order['order_purchase_timestamp'] - first_order_date).days
                
                event = {
                    'order_value': order['order_value'],
                    'days_since_signup': days_since_signup,
                    'order_rank': idx
                }
                events.append(event)
            
            # Calculate features
            total_spend = early_orders['order_value'].sum()
            num_orders = len(early_orders)
            avg_order_value = total_spend / num_orders if num_orders > 0 else 0
            std_order_value = early_orders['order_value'].std() if num_orders > 1 else 0
            
            # Calculate LTV (90-day window)
            ninety_day_orders = customer_data[
                customer_data['order_purchase_timestamp'] <= first_order_date + pd.Timedelta(days=90)
            ]
            ltv_90d = ninety_day_orders['order_value'].sum()
            
            # Determine if whale (top 5% by LTV)
            sequences.append({
                'customer_id': customer_id,
                'signup_date': first_order_date,
                'num_orders': num_orders,
                'total_spend': total_spend,
                'avg_order_value': avg_order_value,
                'std_order_value': std_order_value,
                'first_order_day': 0,
                'last_order_day': days_since_signup if events else 0,
                'avg_days_between': 0,  # Simplified for now
                'events': events,
                'ltv_90d': ltv_90d,
                'is_whale': 0  # Will be set later
            })
        
        sequences_df = pd.DataFrame(sequences)
        
        # Mark whales (top 5% by LTV)
        ltv_threshold = sequences_df['ltv_90d'].quantile(0.95)
        sequences_df['is_whale'] = (sequences_df['ltv_90d'] >= ltv_threshold).astype(int)
        
        return sequences_df
    
    def create_labels(self, sequences: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create LTV and whale labels for each customer."""
        logger.info("Creating LTV and whale labels...")
        
        # Since we already calculated LTV in create_customer_sequences, 
        # we just need to ensure the whale labels are correct
        sequences = sequences.copy()
        
        # Calculate whale threshold (top 5%)
        whale_threshold = sequences['ltv_90d'].quantile(0.95)
        sequences['is_whale'] = (sequences['ltv_90d'] >= whale_threshold).astype(int)
        
        logger.info(f"Created labels for {len(sequences)} customers")
        logger.info(f"Whale threshold: {whale_threshold:.2f}")
        logger.info(f"Whales: {sequences['is_whale'].sum()} ({sequences['is_whale'].mean():.1%})")
        
        return sequences
    
    def save_processed_data(self, data: pd.DataFrame, filename: str = "customer_sequences.parquet"):
        """Save processed data to Parquet format."""
        output_path = self.output_dir / filename
        data.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        return output_path
    
    def process(self, early_window_days: int = 14, max_orders: int = 3) -> pd.DataFrame:
        """Complete data processing pipeline."""
        logger.info("Starting data processing pipeline...")
        
        # Load data
        data = self.load_data()
        
        # Create sequences
        sequences = self.create_customer_sequences(data, early_window_days, max_orders)
        
        # Create labels
        labeled_data = self.create_labels(sequences, data)
        
        # Save processed data
        self.save_processed_data(labeled_data)
        
        logger.info("Data processing complete!")
        return labeled_data


def main():
    """Main function to run data processing."""
    processor = OlistDataProcessor()
    data = processor.process()
    print(f"Processed {len(data)} customer sequences")


if __name__ == "__main__":
    main() 