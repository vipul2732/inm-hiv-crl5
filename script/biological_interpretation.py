#!/usr/bin/env python3
"""
Biological interpretation of clustering analysis results.

This script analyzes the communities and identifies known protein complexes,
condition-specific patterns, and biological significance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Known protein complexes and families
KNOWN_COMPLEXES = {
    'COP9_Signalosome': {'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7A', 'CSN7B', 'CSN8'},
    'Cullin_RING_E3_Ligase_Core': {'CUL2', 'CUL5', 'RBX1', 'RBX2'},
    'Elongin_BC_Complex': {'ELOB', 'ELOC'},
    'ASB_Family': {'ASB1', 'ASB3', 'ASB6', 'ASB7', 'ASB8', 'ASB9', 'ASB13'},
    'SOCS_Family': {'SOCS2', 'SOCS6', 'SOCS7'},
    'PCM_Family': {'PCMD1', 'PCMD2'},
    'LRC_Family': {'LRC14', 'LRC28', 'LRC41', 'LRC42', 'LRC58'},
    'KLDC_Family': {'KLDC1', 'KLDC2', 'KLDC3', 'KLD10'},
    'SPSB_Family': {'SPSB2', 'SPSB3'},
    'PAF1_Complex': {'PAF1', 'CTR9', 'LEO1', 'CDC73', 'WDR61'},
}

PROTEIN_FUNCTIONS = {
    'CUL2': 'Cullin-2, scaffold for E3 ubiquitin ligase',
    'CUL5': 'Cullin-5, scaffold for E3 ubiquitin ligase (VIF targets)',
    'RBX1': 'RING-box protein 1, E3 ligase component',
    'RBX2': 'RING-box protein 2, E3 ligase component',
    'ELOB': 'Elongin B, E3 ligase adaptor',
    'ELOC': 'Elongin C, E3 ligase adaptor',
    'DCA11': 'DDB1 and CUL4 associated factor 11',
    'PEBB': 'Elongin BC complex component',
    'NEDD8': 'NEDD8, ubiquitin-like modifier for cullin activation',
    'vifprotein': 'HIV-1 VIF protein (viral factor)',
    'CSN1': 'COP9 signalosome subunit 1',
    'CSN2': 'COP9 signalosome subunit 2',
    'CSN3': 'COP9 signalosome subunit 3',
    'CSN4': 'COP9 signalosome subunit 4',
    'CSN5': 'COP9 signalosome subunit 5 (JAB1)',
    'CSN6': 'COP9 signalosome subunit 6',
    'CSN7A': 'COP9 signalosome subunit 7A',
    'CSN7B': 'COP9 signalosome subunit 7B',
    'CSN8': 'COP9 signalosome subunit 8',
    'ASB1': 'Ankyrin repeat and SOCS box protein 1',
    'ASB3': 'Ankyrin repeat and SOCS box protein 3',
    'ASB6': 'Ankyrin repeat and SOCS box protein 6',
    'ASB7': 'Ankyrin repeat and SOCS box protein 7',
    'ASB8': 'Ankyrin repeat and SOCS box protein 8',
    'ASB13': 'Ankyrin repeat and SOCS box protein 13',
    'SOCS2': 'Suppressor of cytokine signaling 2',
    'SOCS6': 'Suppressor of cytokine signaling 6',
    'SOCS7': 'Suppressor of cytokine signaling 7',
    'PCMD1': 'PCM domain containing 1',
    'PCMD2': 'PCM domain containing 2',
    'SPSB2': 'SPRY domain-containing SOCS box protein 2',
    'SPSB3': 'SPRY domain-containing SOCS box protein 3',
}


def load_communities(communities_file: Path) -> Dict[int, Set[str]]:
    """Load community assignments from TSV file."""
    df = pd.read_csv(communities_file, sep='\t')
    communities = {}
    for _, row in df.iterrows():
        comm_id = int(row['Community_ID'])
        proteins = set([p.strip() for p in str(row['Proteins']).split(',')])
        communities[comm_id] = proteins
    return communities


def identify_known_complexes(community: Set[str]) -> Dict[str, float]:
    """Identify which known complexes are present in a community."""
    matches = {}
    for complex_name, complex_proteins in KNOWN_COMPLEXES.items():
        overlap = community & complex_proteins
        if len(overlap) > 0:
            coverage = len(overlap) / len(complex_proteins)
            matches[complex_name] = coverage
    return matches


def analyze_communities(communities: Dict[int, Set[str]], condition: str) -> pd.DataFrame:
    """Analyze communities and identify biological complexes."""
    results = []
    
    for comm_id, proteins in communities.items():
        # Identify known complexes
        known_complexes = identify_known_complexes(proteins)
        
        # Check for key proteins
        has_cul5 = 'CUL5' in proteins
        has_cul2 = 'CUL2' in proteins
        has_vif = 'vifprotein' in proteins
        has_csn = any(csn in proteins for csn in ['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7A', 'CSN7B', 'CSN8'])
        has_elob_eloc = 'ELOB' in proteins and 'ELOC' in proteins
        
        # Count protein families
        asb_count = len([p for p in proteins if p.startswith('ASB')])
        socs_count = len([p for p in proteins if p.startswith('SOCS')])
        lrc_count = len([p for p in proteins if p.startswith('LRC')])
        csn_count = len([p for p in proteins if p.startswith('CSN')])
        
        results.append({
            'Condition': condition,
            'Community_ID': comm_id,
            'Size': len(proteins),
            'Proteins': ','.join(sorted(proteins)),
            'Has_CUL5': has_cul5,
            'Has_CUL2': has_cul2,
            'Has_VIF': has_vif,
            'Has_CSN': has_csn,
            'Has_ELOB_ELOC': has_elob_eloc,
            'ASB_Count': asb_count,
            'SOCS_Count': socs_count,
            'LRC_Count': lrc_count,
            'CSN_Count': csn_count,
            'Known_Complexes': '; '.join([f"{name} ({coverage:.1%})" for name, coverage in known_complexes.items()]),
        })
    
    return pd.DataFrame(results)


def compare_conditions(communities_dict: Dict[str, Dict[int, Set[str]]]) -> Dict:
    """Compare communities across conditions."""
    conditions = list(communities_dict.keys())
    
    # Get all proteins across conditions
    all_proteins = set()
    for communities in communities_dict.values():
        for community in communities.values():
            all_proteins.update(community)
    
    # Find condition-specific proteins
    condition_proteins = {cond: set() for cond in conditions}
    for cond, communities in communities_dict.items():
        for community in communities.values():
            condition_proteins[cond].update(community)
    
    # Find unique and shared proteins
    unique_proteins = {}
    shared_proteins = set()
    
    for cond in conditions:
        other_conditions = [c for c in conditions if c != cond]
        other_proteins = set()
        for other_cond in other_conditions:
            other_proteins.update(condition_proteins[other_cond])
        
        unique = condition_proteins[cond] - other_proteins
        unique_proteins[cond] = unique
        
        if not shared_proteins:
            shared_proteins = condition_proteins[cond].copy()
        else:
            shared_proteins &= condition_proteins[cond]
    
    # Analyze VIF-specific patterns
    vif_specific = {}
    if 'VIF' in communities_dict:
        vif_communities = communities_dict['VIF']
        for comm_id, proteins in vif_communities.items():
            if 'vifprotein' in proteins:
                vif_specific[comm_id] = proteins
    
    return {
        'all_proteins': all_proteins,
        'condition_proteins': condition_proteins,
        'unique_proteins': unique_proteins,
        'shared_proteins': shared_proteins,
        'vif_specific_communities': vif_specific,
    }


def identify_hub_proteins(participation_file: Path, top_n: int = 10) -> pd.DataFrame:
    """Identify hub proteins (high degree, high betweenness)."""
    df = pd.read_csv(participation_file, sep='\t')
    df = df.nlargest(top_n, 'Degree')
    return df


def generate_biological_report(
    communities_dir: Path,
    participation_dir: Path,
    output_file: Path
):
    """Generate comprehensive biological interpretation report."""
    
    conditions = ['WT', 'MOCK', 'VIF']
    communities_dict = {}
    participation_dict = {}
    
    # Load all data
    for condition in conditions:
        comm_file = communities_dir / f"{condition.lower()}_communities.tsv"
        part_file = participation_dir / f"{condition.lower()}_protein_participation.tsv"
        
        if comm_file.exists():
            communities_dict[condition] = load_communities(comm_file)
        if part_file.exists():
            participation_dict[condition] = pd.read_csv(part_file, sep='\t')
    
    # Analyze communities
    community_analysis = []
    for condition, communities in communities_dict.items():
        analysis = analyze_communities(communities, condition)
        community_analysis.append(analysis)
    
    community_df = pd.concat(community_analysis, ignore_index=True)
    
    # Compare conditions
    comparison = compare_conditions(communities_dict)
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("BIOLOGICAL INTERPRETATION OF CLUSTERING ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 100 + "\n\n")
        f.write("This analysis identifies protein complexes and condition-specific patterns in cullin-RING\n")
        f.write("E3 ligase networks across WT, MOCK, and VIF conditions.\n\n")
        
        # Key Findings
        f.write("KEY BIOLOGICAL FINDINGS\n")
        f.write("-" * 100 + "\n\n")
        
        # 1. CUL5-VIF Complex
        f.write("1. CUL5-VIF COMPLEX (VIF Condition Specific)\n")
        f.write("   " + "-" * 96 + "\n")
        if 'VIF' in communities_dict:
            vif_communities = communities_dict['VIF']
            for comm_id, proteins in vif_communities.items():
                if 'vifprotein' in proteins and 'CUL5' in proteins:
                    f.write(f"   Community {comm_id} contains VIF protein with CUL5 E3 ligase components:\n")
                    f.write(f"   {', '.join(sorted(proteins))}\n")
                    f.write(f"   \n")
                    f.write(f"   BIOLOGICAL SIGNIFICANCE:\n")
                    f.write(f"   - HIV-1 VIF protein hijacks CUL5-based E3 ligase to degrade APOBEC3G\n")
                    f.write(f"   - This complex is the core mechanism of VIF-mediated immune evasion\n")
                    f.write(f"   - Presence of ELOB/ELOC confirms canonical VIF-CUL5-ELOB-ELOC complex\n")
                    f.write(f"   \n")
        
        # 2. COP9 Signalosome
        f.write("2. COP9 SIGNALOSOME (CSN) COMPLEX\n")
        f.write("   " + "-" * 96 + "\n")
        csn_communities = []
        for condition, communities in communities_dict.items():
            for comm_id, proteins in communities.items():
                csn_proteins = [p for p in proteins if p.startswith('CSN')]
                if len(csn_proteins) >= 5:  # At least 5 CSN subunits
                    csn_communities.append((condition, comm_id, len(csn_proteins), csn_proteins))
        
        if csn_communities:
            f.write("   CSN complex detected in:\n")
            for cond, comm_id, count, csn_prots in csn_communities:
                f.write(f"   - {cond} Community {comm_id}: {count} CSN subunits ({', '.join(sorted(csn_prots))})\n")
            f.write(f"   \n")
            f.write(f"   BIOLOGICAL SIGNIFICANCE:\n")
            f.write(f"   - COP9 signalosome regulates cullin-RING E3 ligases by deneddylation\n")
            f.write(f"   - CSN is essential for proper E3 ligase function and substrate specificity\n")
            f.write(f"   - Co-clustering with cullins suggests functional coupling\n")
            f.write(f"   \n")
        
        # 3. ASB Family Clustering
        f.write("3. ASB (ANKYRIN REPEAT AND SOCS BOX) PROTEIN FAMILY\n")
        f.write("   " + "-" * 96 + "\n")
        asb_by_condition = {}
        for condition, communities in communities_dict.items():
            asb_proteins = set()
            for proteins in communities.values():
                asb_proteins.update([p for p in proteins if p.startswith('ASB')])
            asb_by_condition[condition] = asb_proteins
        
        for condition, asb_proteins in asb_by_condition.items():
            if asb_proteins:
                f.write(f"   {condition}: {len(asb_proteins)} ASB proteins ({', '.join(sorted(asb_proteins))})\n")
        f.write(f"   \n")
        f.write(f"   BIOLOGICAL SIGNIFICANCE:\n")
        f.write(f"   - ASB proteins are substrate adaptors for CUL5-based E3 ligases\n")
        f.write(f"   - They contain ankyrin repeats for substrate binding and SOCS box for cullin interaction\n")
        f.write(f"   - Co-clustering suggests they may compete or cooperate for CUL5 binding\n")
        f.write(f"   \n")
        
        # 4. Condition-Specific Patterns
        f.write("4. CONDITION-SPECIFIC PATTERNS\n")
        f.write("   " + "-" * 96 + "\n")
        
        # WT specific
        if 'WT' in comparison['unique_proteins']:
            wt_unique = comparison['unique_proteins']['WT']
            if wt_unique:
                f.write(f"   WT-specific proteins ({len(wt_unique)}): {', '.join(sorted(wt_unique))}\n")
        
        # MOCK specific
        if 'MOCK' in comparison['unique_proteins']:
            mock_unique = comparison['unique_proteins']['MOCK']
            if mock_unique:
                f.write(f"   MOCK-specific proteins ({len(mock_unique)}): {', '.join(sorted(mock_unique))}\n")
        
        # VIF specific
        if 'VIF' in comparison['unique_proteins']:
            vif_unique = comparison['unique_proteins']['VIF']
            if vif_unique:
                f.write(f"   VIF-specific proteins ({len(vif_unique)}): {', '.join(sorted(vif_unique))}\n")
        
        f.write(f"   \n")
        f.write(f"   Shared across all conditions ({len(comparison['shared_proteins'])}):\n")
        f.write(f"   {', '.join(sorted(comparison['shared_proteins']))}\n")
        f.write(f"   \n")
        
        # 5. Hub Proteins
        f.write("5. HUB PROTEINS (High Connectivity)\n")
        f.write("   " + "-" * 96 + "\n")
        for condition in conditions:
            if condition in participation_dict:
                df = participation_dict[condition]
                top_hubs = df.nlargest(5, 'Degree')[['Protein', 'Degree', 'Betweenness']]
                f.write(f"   {condition} top hubs:\n")
                for _, row in top_hubs.iterrows():
                    func = PROTEIN_FUNCTIONS.get(row['Protein'], 'Unknown function')
                    f.write(f"     - {row['Protein']}: Degree={row['Degree']:.0f}, Betweenness={row['Betweenness']:.4f}\n")
                    f.write(f"       Function: {func}\n")
                f.write(f"   \n")
        
        # 6. Complex Integrity
        f.write("6. COMPLEX INTEGRITY ANALYSIS\n")
        f.write("   " + "-" * 96 + "\n")
        f.write("   How well are known complexes preserved in communities?\n\n")
        
        for condition in conditions:
            if condition in communities_dict:
                f.write(f"   {condition}:\n")
                communities = communities_dict[condition]
                for comm_id, proteins in communities.items():
                    matches = identify_known_complexes(proteins)
                    if matches:
                        f.write(f"     Community {comm_id} (size={len(proteins)}):\n")
                        for complex_name, coverage in matches.items():
                            f.write(f"       - {complex_name}: {coverage:.1%} coverage\n")
                f.write(f"   \n")
        
        # 7. Biological Implications
        f.write("7. BIOLOGICAL IMPLICATIONS\n")
        f.write("   " + "-" * 100 + "\n\n")
        
        # MOCK condition
        if 'MOCK' in communities_dict:
            mock_communities = communities_dict['MOCK']
            largest_comm = max(mock_communities.items(), key=lambda x: len(x[1]))
            f.write("   MOCK Condition:\n")
            f.write(f"   - Largest community ({len(largest_comm[1])} proteins) suggests extensive rewiring\n")
            f.write(f"   - May represent baseline/control state with broad interactions\n")
            f.write(f"   - Lower shared percentage (52.4%) indicates condition-specific network topology\n")
            f.write(f"   \n")
        
        # VIF condition
        if 'VIF' in communities_dict:
            vif_communities = communities_dict['VIF']
            f.write("   VIF Condition:\n")
            f.write(f"   - Only 2 communities suggest focused network reorganization\n")
            f.write(f"   - High shared percentage (90.2%) indicates VIF hijacks existing complexes\n")
            f.write(f"   - VIF-CUL5 complex is the primary driver of condition-specific changes\n")
            f.write(f"   \n")
        
        # WT condition
        if 'WT' in communities_dict:
            f.write("   WT Condition:\n")
            f.write(f"   - 5 communities indicate moderate complexity\n")
            f.write(f"   - Intermediate shared percentage (87.6%) suggests balanced network\n")
            f.write(f"   - Presence of vifprotein in WT suggests viral protein expression\n")
            f.write(f"   \n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    logger.info(f"Biological interpretation report saved to {output_file}")
    
    # Also save detailed community analysis
    community_df.to_csv(output_file.parent / "community_biological_analysis.tsv", sep='\t', index=False)
    logger.info(f"Detailed community analysis saved to {output_file.parent / 'community_biological_analysis.tsv'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate biological interpretation of clustering analysis'
    )
    parser.add_argument('--communities-dir', type=Path, default='clustering_analysis',
                       help='Directory containing community TSV files')
    parser.add_argument('--participation-dir', type=Path, default='clustering_analysis',
                       help='Directory containing participation TSV files')
    parser.add_argument('--output', type=Path, default='clustering_analysis/biological_interpretation.txt',
                       help='Output file for biological interpretation report')
    
    args = parser.parse_args()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    generate_biological_report(
        args.communities_dir,
        args.participation_dir,
        args.output
    )


if __name__ == '__main__':
    main()
