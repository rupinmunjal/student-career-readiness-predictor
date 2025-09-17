#!/usr/bin/env python3
"""
Validation script for the data pipeline.
Tests that all datasets can be loaded and merged successfully.
"""

import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import prepare_training_dataset


def validate_pipeline() -> bool:
    """Validate the data pipeline by loading and merging all available datasets."""
    try:
        print("🔍 Validating data pipeline...")
        
        result = prepare_training_dataset(
            base_dataset_path="data/Student Attitude and Behavior.csv",
            data_dir="data"
        )
        
        print(f"✅ Successfully loaded {len(result.used_files)} dataset(s)")
        print(f"📊 Total records: {len(result.data)}")
        print(f"📋 Dataset counts: {result.dataset_counts}")
        
        from src.data_pipeline import MODEL_FEATURES, DOMAIN_COLUMNS, TARGET_COLUMN
        required_cols = MODEL_FEATURES + DOMAIN_COLUMNS + [TARGET_COLUMN]
        
        missing_cols = [col for col in required_cols if col not in result.data.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        missing_values = result.data[MODEL_FEATURES].isna().sum().sum()
        if missing_values > 0:
            print(f"❌ Found {missing_values} missing values in features")
            return False
        
        if result.data[TARGET_COLUMN].isna().any():
            print("❌ Target column has missing values")
            return False
        
        print("✅ All validations passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_pipeline()
    sys.exit(0 if success else 1)