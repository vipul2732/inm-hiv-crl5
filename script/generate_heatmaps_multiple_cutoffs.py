#!/usr/bin/env python3
"""
Generate clustered heatmaps at multiple probability cutoffs for VIF KO and Mock conditions.
"""

import sys
import logging
from pathlib import Path

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from script.plot_vif_ko_heatmap import (
    load_vif_ko_interactions,
    create_interaction_matrix,
    plot_heatmap_clustered as plot_vif_ko_heatmap_clustered,
)
from script.plot_mock_heatmap import (
    load_mock_interactions,
    create_interaction_matrix as create_mock_interaction_matrix,
    plot_heatmap_clustered as plot_mock_heatmap_clustered,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    output_dir = Path('wt_vif_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vif_file = Path('results/cullin_vif_config_multi_seeds/all_pairs_with_uniprot.tsv')
    mock_file = Path('results/cullin_mock_config_multi_seeds/all_pairs_with_uniprot.tsv')
    
    cutoffs = [0.66, 0.67]
    
    logger.info("=" * 80)
    logger.info("Generating heatmaps at multiple cutoffs")
    logger.info("=" * 80)
    
    for cutoff in cutoffs:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing cutoff >= {cutoff}")
        logger.info(f"{'=' * 80}")
        
        # VIF KO heatmap
        logger.info(f"\nVIF KO (cutoff >= {cutoff}):")
        df_vif = load_vif_ko_interactions(vif_file, min_prob=cutoff)
        if len(df_vif) > 0:
            matrix_vif, protein_list_vif, _ = create_interaction_matrix(df_vif)
            output_file_vif = output_dir / f'vif_ko_heatmap_clustered_cutoff_{cutoff:.2f}.png'
            title_vif = f'HIV Δ VIF infection (cutoff >= {cutoff})'
            plot_vif_ko_heatmap_clustered(matrix_vif, protein_list_vif, output_file_vif, title=title_vif)
            logger.info(f"  Saved: {output_file_vif}")
            logger.info(f"  Interactions: {len(df_vif)}, Proteins: {len(protein_list_vif)}")
        else:
            logger.warning(f"  No interactions found at cutoff >= {cutoff}")
        
        # Mock heatmap
        logger.info(f"\nMock (cutoff >= {cutoff}):")
        df_mock = load_mock_interactions(mock_file, min_prob=cutoff)
        if len(df_mock) > 0:
            matrix_mock, protein_list_mock, _ = create_mock_interaction_matrix(df_mock)
            output_file_mock = output_dir / f'mock_heatmap_clustered_cutoff_{cutoff:.2f}.png'
            title_mock = f'mock infection (cutoff >= {cutoff})'
            plot_mock_heatmap_clustered(matrix_mock, protein_list_mock, output_file_mock, title=title_mock)
            logger.info(f"  Saved: {output_file_mock}")
            logger.info(f"  Interactions: {len(df_mock)}, Proteins: {len(protein_list_mock)}")
        else:
            logger.warning(f"  No interactions found at cutoff >= {cutoff}")
    
    logger.info("\n" + "=" * 80)
    logger.info("All heatmaps generated successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
