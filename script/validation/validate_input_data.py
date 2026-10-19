#!/usr/bin/env python3
"""
Validate input Excel data file before running the workflow.

This script checks that your input data file has the required columns
and format before preprocessing.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    'Bait': 'The protein being pulled down',
    'PreyGene': 'The co-purifying protein (or specify different column name)',
    'SaintScore': 'Confidence score (0.0 to 1.0)',
    'Spec': 'Spectral counts for experimental replicates (pipe-separated)',
    'ctrlCounts': 'Control spectral counts (pipe-separated)',
}

OPTIONAL_COLUMNS = {
    'Prey': 'Alternative prey column name',
}


def check_file_exists(filepath: Path) -> bool:
    """Check if file exists and is readable."""
    if not filepath.exists():
        logger.error(f"✗ File not found: {filepath}")
        return False
    
    if not filepath.is_file():
        logger.error(f"✗ Not a file: {filepath}")
        return False
    
    logger.info(f"✓ File exists: {filepath}")
    return True


def check_excel_format(filepath: Path) -> bool:
    """Check if file is a valid Excel file."""
    try:
        excel_file = pd.ExcelFile(filepath)
        logger.info(f"✓ Valid Excel file with {len(excel_file.sheet_names)} sheet(s)")
        return True
    except Exception as e:
        logger.error(f"✗ Not a valid Excel file: {e}")
        return False


def check_columns(df: pd.DataFrame, required_cols: list, prey_colname: str = 'PreyGene'):
    """Check if DataFrame has required columns."""
    missing = []
    found = []
    
    # Check required columns (with prey column flexibility)
    for col in required_cols:
        if col == 'PreyGene':
            # Check for PreyGene or Prey
            if prey_colname in df.columns:
                found.append(prey_colname)
            elif 'Prey' in df.columns:
                found.append('Prey')
                logger.warning(f"⚠ Found 'Prey' column instead of 'PreyGene'. Use --prey-column Prey if needed.")
            else:
                missing.append(col)
        elif col in df.columns:
            found.append(col)
        else:
            missing.append(col)
    
    return len(missing) == 0, missing


def validate_column_data(df: pd.DataFrame, colname: str, description: str) -> bool:
    """Validate that a column has appropriate data."""
    if colname not in df.columns:
        return False
    
    col = df[colname]
    
    # Check for empty values
    n_empty = col.isna().sum()
    if n_empty > 0:
        logger.warning(f"⚠ {colname}: {n_empty} empty values found")
    
    # Column-specific validations
    if colname == 'Bait':
        # Should be strings
        n_non_string = (~col.astype(str).str.strip().astype(bool)).sum()
        if n_non_string > 0:
            logger.warning(f"⚠ {colname}: {n_non_string} non-string values")
    
    elif colname in ['PreyGene', 'Prey']:
        # Should be strings
        n_non_string = (~col.astype(str).str.strip().astype(bool)).sum()
        if n_non_string > 0:
            logger.warning(f"⚠ {colname}: {n_non_string} non-string values")
    
    elif colname == 'SaintScore':
        # Should be numeric, between 0 and 1
        try:
            scores = pd.to_numeric(col, errors='coerce')
            n_invalid = scores.isna().sum()
            if n_invalid > 0:
                logger.warning(f"⚠ {colname}: {n_invalid} non-numeric values")
            
            out_of_range = ((scores < 0) | (scores > 1)).sum()
            if out_of_range > 0:
                logger.warning(f"⚠ {colname}: {out_of_range} values outside [0, 1] range")
        except Exception as e:
            logger.warning(f"⚠ {colname}: Could not validate numeric values: {e}")
    
    elif colname in ['Spec', 'ctrlCounts']:
        # Should contain pipe-separated values
        n_no_pipe = (~col.astype(str).str.contains('|', na=False)).sum()
        if n_no_pipe > 0:
            logger.warning(f"⚠ {colname}: {n_no_pipe} rows without pipe-separated values")
    
    return True


def validate_data_structure(filepath: Path, sheet_nums: int = 3, prey_colname: str = 'PreyGene') -> bool:
    """Validate the structure of the Excel data."""
    excel_file = pd.ExcelFile(filepath)
    total_sheets = len(excel_file.sheet_names)
    
    if sheet_nums is None:
        sheet_nums = total_sheets
    elif sheet_nums > total_sheets:
        logger.warning(f"⚠ Requested {sheet_nums} sheets but only {total_sheets} available")
        sheet_nums = total_sheets
    
    all_valid = True
    required_cols = list(REQUIRED_COLUMNS.keys())
    
    for sheet_num in range(min(sheet_nums, total_sheets)):
        logger.info(f"\nChecking sheet {sheet_num} ({excel_file.sheet_names[sheet_num]})...")
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_num)
            logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Check columns
            has_cols, missing = check_columns(df, required_cols, prey_colname)
            if not has_cols:
                logger.error(f"  ✗ Missing required columns: {', '.join(missing)}")
                all_valid = False
            else:
                logger.info(f"  ✓ All required columns present")
            
            # Validate column data
            for col in required_cols:
                actual_col = prey_colname if col == 'PreyGene' else col
                if actual_col in df.columns or (col == 'PreyGene' and 'Prey' in df.columns):
                    validate_column_data(df, actual_col if actual_col in df.columns else 'Prey', 
                                        REQUIRED_COLUMNS[col])
            
            # Check for corrupt rows (non-string bait/prey)
            bait_col = 'Bait'
            prey_col = prey_colname if prey_colname in df.columns else 'Prey'
            
            if bait_col in df.columns and prey_col in df.columns:
                corrupt = ~(df[bait_col].astype(str).str.strip().astype(bool) & 
                           df[prey_col].astype(str).str.strip().astype(bool))
                n_corrupt = corrupt.sum()
                if n_corrupt > 0:
                    logger.warning(f"  ⚠ {n_corrupt} rows with invalid bait/prey values (will be removed during preprocessing)")
            
            # Summary stats
            if 'Bait' in df.columns:
                n_baits = df['Bait'].nunique()
                logger.info(f"  Unique baits: {n_baits}")
            
            if prey_col in df.columns:
                n_prey = df[prey_col].nunique()
                logger.info(f"  Unique prey: {n_prey}")
            
            if 'SaintScore' in df.columns:
                try:
                    scores = pd.to_numeric(df['SaintScore'], errors='coerce')
                    mean_score = scores.mean()
                    logger.info(f"  Mean SAINT score: {mean_score:.3f}")
                except:
                    pass
        
        except Exception as e:
            logger.error(f"  ✗ Error reading sheet {sheet_num}: {e}")
            all_valid = False
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description='Validate input Excel data file before running workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a data file
  python validate_input_data.py data.xlsx
  
  # Validate with custom prey column name
  python validate_input_data.py data.xlsx --prey-column Prey
  
  # Validate specific number of sheets
  python validate_input_data.py data.xlsx --sheets 5
        """
    )
    
    parser.add_argument('data_file', type=Path, help='Path to input Excel file')
    parser.add_argument('--prey-column', type=str, default='PreyGene',
                       help='Name of prey column (default: PreyGene)')
    parser.add_argument('--sheets', type=int, default=3,
                       help='Number of sheets to check (default: 3)')
    parser.add_argument('--strict', action='store_true',
                       help='Exit with error on warnings')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Input Data Validation")
    print("=" * 60)
    print()
    
    # Step 1: Check file exists
    if not check_file_exists(args.data_file):
        sys.exit(1)
    
    # Step 2: Check Excel format
    if not check_excel_format(args.data_file):
        sys.exit(1)
    
    # Step 3: Validate data structure
    print()
    is_valid = validate_data_structure(args.data_file, args.sheets, args.prey_column)
    
    print()
    print("=" * 60)
    if is_valid:
        print("✓ Validation passed! Data file is ready for processing.")
        print()
        print("Next steps:")
        print("  1. Create a config file (or use existing template)")
        print("  2. Run: python run.py --config config.yaml")
        sys.exit(0)
    else:
        print("✗ Validation failed! Please fix the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

