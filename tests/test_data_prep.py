"""
Unit tests for data preparation module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data_prep import OlistDataProcessor


class TestOlistDataProcessor:
    """Test cases for OlistDataProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.raw_dir.mkdir()
        self.processed_dir.mkdir()
        
        # Create mock data
        self.create_mock_data()
        
        # Initialize processor
        self.processor = OlistDataProcessor(
            data_dir=str(self.raw_dir),
            output_dir=str(self.processed_dir)
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_data(self):
        """Create mock Olist dataset files for testing."""
        # Mock customers data
        customers_data = {
            'customer_id': ['c1', 'c2', 'c3'],
            'customer_unique_id': ['cu1', 'cu2', 'cu3'],
            'customer_zip_code_prefix': ['12345', '23456', '34567'],
            'customer_city': ['City1', 'City2', 'City3'],
            'customer_state': ['ST1', 'ST2', 'ST3']
        }
        pd.DataFrame(customers_data).to_csv(self.raw_dir / 'olist_customers_dataset.csv', index=False)
        
        # Mock orders data
        orders_data = {
            'order_id': ['o1', 'o2', 'o3', 'o4'],
            'customer_id': ['c1', 'c1', 'c2', 'c3'],
            'order_status': ['delivered', 'delivered', 'delivered', 'delivered'],
            'order_purchase_date': ['2020-01-01', '2020-01-05', '2020-01-02', '2020-01-03'],
            'order_approved_at': ['2020-01-01', '2020-01-05', '2020-01-02', '2020-01-03'],
            'order_delivered_carrier_date': ['2020-01-02', '2020-01-06', '2020-01-03', '2020-01-04'],
            'order_delivered_customer_date': ['2020-01-03', '2020-01-07', '2020-01-04', '2020-01-05'],
            'order_estimated_delivery_date': ['2020-01-05', '2020-01-09', '2020-01-06', '2020-01-07']
        }
        pd.DataFrame(orders_data).to_csv(self.raw_dir / 'olist_orders_dataset.csv', index=False)
        
        # Mock order items data
        order_items_data = {
            'order_id': ['o1', 'o2', 'o3', 'o4'],
            'order_item_id': [1, 1, 1, 1],
            'product_id': ['p1', 'p2', 'p3', 'p4'],
            'seller_id': ['s1', 's2', 's3', 's4'],
            'shipping_limit_date': ['2020-01-02', '2020-01-06', '2020-01-03', '2020-01-04'],
            'price': [100.0, 200.0, 150.0, 300.0],
            'freight_value': [10.0, 20.0, 15.0, 30.0]
        }
        pd.DataFrame(order_items_data).to_csv(self.raw_dir / 'olist_order_items_dataset.csv', index=False)
        
        # Mock order payments data
        order_payments_data = {
            'order_id': ['o1', 'o2', 'o3', 'o4'],
            'payment_sequential': [1, 1, 1, 1],
            'payment_type': ['credit_card', 'credit_card', 'credit_card', 'credit_card'],
            'payment_installments': [1, 1, 1, 1],
            'payment_value': [110.0, 220.0, 165.0, 330.0]
        }
        pd.DataFrame(order_payments_data).to_csv(self.raw_dir / 'olist_order_payments_dataset.csv', index=False)
        
        # Mock products data
        products_data = {
            'product_id': ['p1', 'p2', 'p3', 'p4'],
            'product_category_name': ['electronics', 'books', 'home', 'sports'],
            'product_name_lenght': [10, 8, 6, 12],
            'product_description_lenght': [100, 80, 60, 120],
            'product_photos_qty': [1, 1, 1, 1],
            'product_weight_g': [500, 300, 1000, 800],
            'product_length_cm': [20, 15, 30, 25],
            'product_height_cm': [10, 5, 15, 12],
            'product_width_cm': [15, 10, 20, 18]
        }
        pd.DataFrame(products_data).to_csv(self.raw_dir / 'olist_products_dataset.csv', index=False)
        
        # Mock sellers data
        sellers_data = {
            'seller_id': ['s1', 's2', 's3', 's4'],
            'seller_zip_code_prefix': ['12345', '23456', '34567', '45678'],
            'seller_city': ['City1', 'City2', 'City3', 'City4'],
            'seller_state': ['ST1', 'ST2', 'ST3', 'ST4']
        }
        pd.DataFrame(sellers_data).to_csv(self.raw_dir / 'olist_sellers_dataset.csv', index=False)
    
    def test_load_data(self):
        """Test data loading functionality."""
        data = self.processor.load_data()
        
        assert isinstance(data, dict)
        assert 'customers' in data
        assert 'orders' in data
        assert 'order_items' in data
        assert 'order_payments' in data
        assert 'products' in data
        assert 'sellers' in data
        
        # Check data shapes
        assert len(data['customers']) == 3
        assert len(data['orders']) == 4
        assert len(data['order_items']) == 4
        assert len(data['order_payments']) == 4
        assert len(data['products']) == 4
        assert len(data['sellers']) == 4
    
    def test_create_customer_sequences(self):
        """Test customer sequence creation."""
        data = self.processor.load_data()
        sequences = self.processor.create_customer_sequences(data, early_window_days=14, max_orders=3)
        
        assert isinstance(sequences, pd.DataFrame)
        assert len(sequences) > 0
        assert 'customer_id' in sequences.columns
        assert 'events' in sequences.columns
        assert 'num_orders' in sequences.columns
        assert 'total_spend' in sequences.columns
    
    def test_create_labels(self):
        """Test label creation."""
        data = self.processor.load_data()
        sequences = self.processor.create_customer_sequences(data)
        labeled_data = self.processor.create_labels(sequences, data)
        
        assert isinstance(labeled_data, pd.DataFrame)
        assert 'ltv_90d' in labeled_data.columns
        assert 'is_whale' in labeled_data.columns
        assert len(labeled_data) > 0
        
        # Check whale labels are binary
        assert labeled_data['is_whale'].isin([0, 1]).all()
    
    def test_save_processed_data(self):
        """Test data saving functionality."""
        data = self.processor.load_data()
        sequences = self.processor.create_customer_sequences(data)
        labeled_data = self.processor.create_labels(sequences, data)
        
        output_path = self.processor.save_processed_data(labeled_data, "test_data.parquet")
        
        assert output_path.exists()
        assert output_path.suffix == '.parquet'
        
        # Test loading saved data
        loaded_data = pd.read_parquet(output_path)
        assert len(loaded_data) == len(labeled_data)
        assert all(col in loaded_data.columns for col in labeled_data.columns)
    
    def test_full_pipeline(self):
        """Test the complete data processing pipeline."""
        result = self.processor.process(early_window_days=14, max_orders=3)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'customer_id' in result.columns
        assert 'ltv_90d' in result.columns
        assert 'is_whale' in result.columns
        assert 'events' in result.columns
        
        # Check that output file was created
        output_file = self.processed_dir / "customer_sequences.parquet"
        assert output_file.exists()


if __name__ == "__main__":
    pytest.main([__file__]) 