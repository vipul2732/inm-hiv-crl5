#!/usr/bin/env python3
"""
Generate a statistics chart showing database sizes and validation overlaps.

This script creates a visualization showing:
1. Total pairs in BioGRID and CORUM databases
2. PDB counts for each cullin dataset
3. Overlap between predictions and databases (BioGRID, CORUM)

Usage:
    python generate_database_stats_chart.py \
        --input-dir validation_pdb_and_databases \
        --biogrid database/BIOGRID-ALL-5.0.253.mitab.txt \
        --corum database/corum_humanComplexes.tsv \
        --output validation_pdb_and_databases/database_statistics_chart.png
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Dataset name mapping for display
DATASET_DISPLAY_NAMES = {
    'WT': 'HIV WT type infection',
    'MOCK': 'mock infection',
    'VIF': 'HIV Δ VIF infection'  # Using delta symbol
}


def load_biogrid_interactions(biogrid_file: Path) -> Set[Tuple[str, str]]:
    """Load interactions from BioGRID MITAB file."""
    logger.info(f"Loading BioGRID interactions from {biogrid_file}")
    interactions = set()
    
    if not biogrid_file.exists():
        logger.warning(f"BioGRID file not found: {biogrid_file}")
        return interactions
    
    try:
        uniprot_pattern = re.compile(r'uniprot/(?:swiss-prot|trEMBL):([A-Z0-9-]+)')
        
        with open(biogrid_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            if not first_line.startswith('#'):
                f.seek(0)
            
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 3:
                    continue
                
                alt_ids_a = fields[2] if len(fields) > 2 else ''
                alt_ids_b = fields[3] if len(fields) > 3 else ''
                
                uniprot_ids_a = set(uniprot_pattern.findall(alt_ids_a))
                uniprot_ids_b = set(uniprot_pattern.findall(alt_ids_b))
                
                for up_a in uniprot_ids_a:
                    for up_b in uniprot_ids_b:
                        if up_a and up_b and up_a != up_b:
                            pair = tuple(sorted([up_a, up_b]))
                            interactions.add(pair)
        
        logger.info(f"Loaded {len(interactions)} unique BioGRID interactions")
        
    except Exception as e:
        logger.error(f"Error loading BioGRID: {e}")
    
    return interactions


def load_corum_interactions(corum_file: Path) -> Set[Tuple[str, str]]:
    """Load interactions from CORUM complexes file."""
    logger.info(f"Loading CORUM interactions from {corum_file}")
    interactions = set()
    
    if not corum_file.exists():
        logger.warning(f"CORUM file not found: {corum_file}")
        return interactions
    
    try:
        df = pd.read_csv(corum_file, sep='\t', low_memory=False)
        
        if 'subunits_uniprot_id' not in df.columns:
            logger.warning("CORUM file missing 'subunits_uniprot_id' column")
            return interactions
        
        for _, row in df.iterrows():
            subunits_str = str(row['subunits_uniprot_id'])
            if pd.isna(subunits_str) or subunits_str == 'nan':
                continue
            
            subunits = [s.strip() for s in subunits_str.split(';') if s.strip()]
            
            for i, sub1 in enumerate(subunits):
                for sub2 in subunits[i+1:]:
                    if sub1 and sub2 and sub1 != sub2:
                        pair = tuple(sorted([sub1, sub2]))
                        interactions.add(pair)
        
        logger.info(f"Loaded {len(interactions)} unique CORUM interactions")
        
    except Exception as e:
        logger.error(f"Error loading CORUM: {e}")
    
    return interactions


def load_validation_data(validation_file: Path) -> Dict:
    """Load validation data and compute statistics."""
    logger.info(f"Loading validation data from {validation_file}")
    
    if not validation_file.exists():
        logger.warning(f"Validation file not found: {validation_file}")
        return {
            'total_pairs': 0,
            'pdb_count': 0,
            'biogrid_count': 0,
            'corum_count': 0,
            'any_database_count': 0
        }
    
    try:
        df = pd.read_csv(validation_file, sep='\t')
        
        # Filter out pairs without valid UniProt IDs
        df = df[
            (df['Protein1_UniProt'].notna()) & 
            (df['Protein2_UniProt'].notna()) &
            (df['Protein1_UniProt'] != 'nan') &
            (df['Protein2_UniProt'] != 'nan')
        ].copy()
        
        stats = {
            'total_pairs': len(df),
            'pdb_count': df['Has_Shared_PDB'].sum() if 'Has_Shared_PDB' in df.columns else 0,
            'biogrid_count': df['In_BioGRID'].sum() if 'In_BioGRID' in df.columns else 0,
            'corum_count': df['In_CORUM'].sum() if 'In_CORUM' in df.columns else 0,
            'any_database_count': df['In_Any_Database'].sum() if 'In_Any_Database' in df.columns else 0
        }
        
        logger.info(f"  Total pairs: {stats['total_pairs']}")
        logger.info(f"  PDB validated: {stats['pdb_count']}")
        logger.info(f"  In BioGRID: {stats['biogrid_count']}")
        logger.info(f"  In CORUM: {stats['corum_count']}")
        logger.info(f"  In any database: {stats['any_database_count']}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return {
            'total_pairs': 0,
            'pdb_count': 0,
            'biogrid_count': 0,
            'corum_count': 0,
            'any_database_count': 0
        }


def set_pub_style():
    """Set publication-quality styling."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.6,
        "lines.linewidth": 3.0,
        "lines.markersize": 8,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": True,
    })


def create_database_stats_chart(
    datasets: Dict[str, Dict],
    biogrid_total: int,
    corum_total: int,
    output_file: Path
):
    """
    Create a multi-panel chart showing database statistics.
    
    Panel 1: Database sizes (BioGRID, CORUM) - total pairs in databases
    Panel 2: Combined view - Total sampled pairs (transparent) with mapped pairs (BioGRID, CORUM, PDB) inside
    """
    set_pub_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors
    dataset_colors = {'WT': '#1f77b4', 'MOCK': '#ff7f0e', 'VIF': '#2ca02c'}
    db_colors = {'BioGRID': '#4472C4', 'CORUM': '#ED7D31', 'PDB': '#70AD47'}
    
    dataset_names = list(datasets.keys())
    # Create display names for x-axis labels
    dataset_display_names = [DATASET_DISPLAY_NAMES.get(name, name) for name in dataset_names]
    
    # Panel 1: Database sizes as pie chart
    ax1 = axes[0]
    db_names = ['BioGRID', 'CORUM']
    db_counts = [biogrid_total, corum_total]
    
    # Create pie chart
    wedges, texts, autotexts = ax1.pie(db_counts, 
                                       labels=db_names,
                                       colors=[db_colors['BioGRID'], db_colors['CORUM']],
                                       autopct=lambda pct: '',
                                       startangle=90,
                                       textprops={'fontsize': 18, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    # Format label text (database names)
    for text in texts:
        text.set_fontsize(18)
        text.set_fontweight('bold')
    
    # Add count labels with better visibility
    for i, (wedge, count, name) in enumerate(zip(wedges, db_counts, db_names)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        # Position count at center of each wedge
        x = 0.5 * np.cos(np.deg2rad(angle))
        y = 0.5 * np.sin(np.deg2rad(angle))
        # Add white background for better visibility
        ax1.text(x, y, f'{count:,}', ha='center', va='center',
                fontsize=16, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='white', linewidth=2))
    
    ax1.set_title('Database Sizes', fontsize=18, fontweight='bold', pad=15)
    
    # Panel 2: Combined - Total sampled pairs (transparent) with mapped pairs inside
    ax2 = axes[1]
    x = np.arange(len(dataset_names))
    width = 0.6
    
    total_counts = [datasets[name]['total_pairs'] for name in dataset_names]
    biogrid_counts = [datasets[name]['biogrid_count'] for name in dataset_names]
    corum_counts = [datasets[name]['corum_count'] for name in dataset_names]
    pdb_counts = [datasets[name]['pdb_count'] for name in dataset_names]
    
    # Draw transparent total bars with outline
    bars_total = ax2.bar(x, total_counts, width,
                        color='none', edgecolor='black', linewidth=2.5,
                        linestyle='--', alpha=0.3, label='Total Sampled')
    
    # Draw mapped pairs as grouped bars inside (centered within total bars)
    inner_width = width * 0.7
    # Center the inner bars: start at x - inner_width/2
    inner_start = -inner_width / 2
    
    bars_bg = ax2.bar(x + inner_start, biogrid_counts, inner_width / 3,
                     label='BioGRID', color=db_colors['BioGRID'], alpha=0.8,
                     edgecolor='black', linewidth=1.5)
    bars_corum = ax2.bar(x + inner_start + inner_width/3, corum_counts, inner_width / 3,
                        label='CORUM', color=db_colors['CORUM'], alpha=0.8,
                        edgecolor='black', linewidth=1.5)
    bars_pdb = ax2.bar(x + inner_start + 2*inner_width/3, pdb_counts, inner_width / 3,
                      label='PDB', color=db_colors['PDB'], alpha=0.8,
                      edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Number of Pairs', fontsize=16, fontweight='bold')
    ax2.set_title('Sampled Pairs and Mappings', fontsize=18, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_display_names, fontsize=11, rotation=15, ha='right')
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels for total (on top)
    for bar, count in zip(bars_total, total_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'Total: {int(count):,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='black')
    
    # Add value labels for mapped pairs
    for bars, label_prefix in [(bars_bg, 'BG:'), (bars_corum, 'COR:'), (bars_pdb, 'PDB:')]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    logger.info(f"Saved database statistics chart to {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate database statistics visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('validation_pdb_and_databases'),
        help='Directory containing validation TSV files'
    )
    parser.add_argument(
        '--biogrid',
        type=Path,
        default=Path('database/BIOGRID-ALL-5.0.253.mitab.txt'),
        help='Path to BioGRID MITAB file'
    )
    parser.add_argument(
        '--corum',
        type=Path,
        default=Path('database/corum_humanComplexes.tsv'),
        help='Path to CORUM TSV file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('validation_pdb_and_databases/database_statistics_chart.png'),
        help='Path to output PNG file'
    )
    
    args = parser.parse_args()
    
    # Load database sizes
    logger.info("Loading database sizes...")
    biogrid_interactions = load_biogrid_interactions(args.biogrid)
    corum_interactions = load_corum_interactions(args.corum)
    
    biogrid_total = len(biogrid_interactions)
    corum_total = len(corum_interactions)
    
    logger.info(f"BioGRID total pairs: {biogrid_total:,}")
    logger.info(f"CORUM total pairs: {corum_total:,}")
    
    # Load validation data for each dataset
    datasets = {}
    for dataset_name in ['WT', 'MOCK', 'VIF']:
        validation_file = args.input_dir / f'cullin_{dataset_name.lower()}_database_comparison.tsv'
        stats = load_validation_data(validation_file)
        datasets[dataset_name] = stats
    
    # Create chart
    create_database_stats_chart(
        datasets,
        biogrid_total,
        corum_total,
        args.output
    )
    
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
