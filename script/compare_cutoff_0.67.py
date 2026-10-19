#!/usr/bin/env python3
"""Compare Mock and VIF KO at cutoff 0.67"""

import pandas as pd
from collections import Counter

# Load data
vif_df = pd.read_csv('results/cullin_vif_config_multi_seeds/all_pairs_with_uniprot.tsv', sep='\t')
vif_df['Protein1'] = vif_df['Protein1'].replace('PEBB', 'CBFB')
vif_df['Protein2'] = vif_df['Protein2'].replace('PEBB', 'CBFB')

mock_df = pd.read_csv('results/cullin_mock_config_multi_seeds/all_pairs_with_uniprot.tsv', sep='\t')
mock_df['Protein1'] = mock_df['Protein1'].replace('PEBB', 'CBFB')
mock_df['Protein2'] = mock_df['Protein2'].replace('PEBB', 'CBFB')

# Filter at 0.67
cutoff = 0.67
vif_high = vif_df[vif_df['Probability'] >= cutoff].copy()
mock_high = mock_df[mock_df['Probability'] >= cutoff].copy()

# Create pair sets
vif_pairs = set(zip(vif_high['Protein1'], vif_high['Protein2']))
mock_pairs = set(zip(mock_high['Protein1'], mock_high['Protein2']))

# Find differences
vif_only = vif_pairs - mock_pairs
mock_only = mock_pairs - vif_pairs
shared = vif_pairs & mock_pairs

print('=' * 80)
print(f'COMPARISON AT CUTOFF >= {cutoff}')
print('=' * 80)
print(f'\nVIF KO: {len(vif_high)} interactions, {len(set(vif_high["Protein1"].tolist() + vif_high["Protein2"].tolist()))} proteins')
print(f'Mock:   {len(mock_high)} interactions, {len(set(mock_high["Protein1"].tolist() + mock_high["Protein2"].tolist()))} proteins')
print(f'\nShared interactions: {len(shared)}')
print(f'VIF KO-specific: {len(vif_only)}')
print(f'Mock-specific: {len(mock_only)}')

# Get protein frequencies
vif_proteins = Counter(vif_high['Protein1'].tolist() + vif_high['Protein2'].tolist())
mock_proteins = Counter(mock_high['Protein1'].tolist() + mock_high['Protein2'].tolist())

# Find proteins with different connectivity
all_proteins = set(vif_proteins.keys()) | set(mock_proteins.keys())
vif_only_proteins = set(vif_proteins.keys()) - set(mock_proteins.keys())
mock_only_proteins = set(mock_proteins.keys()) - set(vif_proteins.keys())

print(f'\nProteins in VIF KO only: {sorted(vif_only_proteins)}')
print(f'Proteins in Mock only: {sorted(mock_only_proteins)}')

# Key protein complexes
key_proteins = {'CUL5', 'CUL2', 'ELOB', 'ELOC', 'RBX1', 'RBX2', 'CBFB', 'NEDD8', 
                'CAND1', 'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7A', 
                'CSN7B', 'CSN8', 'PAF1', 'CTR9', 'LEO1', 'CDC73', 'WDR61', 
                'RUNX1', 'RUNX2', 'RUNX3', 'HIF1A', 'KAT3'}

print('\n' + '=' * 80)
print('KEY PROTEIN CONNECTIVITY (number of high-confidence interactions)')
print('=' * 80)
print(f'\n{"Protein":<12} {"VIF KO":<10} {"Mock":<10} {"Difference":<12}')
print('-' * 80)
for prot in sorted(key_proteins & all_proteins):
    vif_conn = vif_proteins.get(prot, 0)
    mock_conn = mock_proteins.get(prot, 0)
    diff = vif_conn - mock_conn
    if abs(diff) > 0 or vif_conn > 0 or mock_conn > 0:
        print(f'{prot:<12} {vif_conn:<10} {mock_conn:<10} {diff:+12}')

# Top interactions by probability
print('\n' + '=' * 80)
print('TOP 10 INTERACTIONS IN EACH CONDITION')
print('=' * 80)
print('\nVIF KO Top 10:')
vif_top = vif_high.nlargest(10, 'Probability')[['Protein1', 'Protein2', 'Probability']]
for idx, row in vif_top.iterrows():
    print(f'  {row["Protein1"]:12} - {row["Protein2"]:12} : {row["Probability"]:.4f}')

print('\nMock Top 10:')
mock_top = mock_high.nlargest(10, 'Probability')[['Protein1', 'Protein2', 'Probability']]
for idx, row in mock_top.iterrows():
    print(f'  {row["Protein1"]:12} - {row["Protein2"]:12} : {row["Probability"]:.4f}')

# VIF KO-specific interactions
if len(vif_only) > 0:
    print('\n' + '=' * 80)
    print('VIF KO-SPECIFIC INTERACTIONS (top 10 by probability)')
    print('=' * 80)
    vif_specific = []
    for p1, p2 in vif_only:
        prob = vif_high[(vif_high['Protein1'] == p1) & (vif_high['Protein2'] == p2)]['Probability'].values[0]
        vif_specific.append((p1, p2, prob))
    vif_specific.sort(key=lambda x: x[2], reverse=True)
    for p1, p2, prob in vif_specific[:10]:
        print(f'  {p1:12} - {p2:12} : {prob:.4f}')

# Mock-specific interactions
if len(mock_only) > 0:
    print('\n' + '=' * 80)
    print('MOCK-SPECIFIC INTERACTIONS (top 10 by probability)')
    print('=' * 80)
    mock_specific = []
    for p1, p2 in mock_only:
        prob = mock_high[(mock_high['Protein1'] == p1) & (mock_high['Protein2'] == p2)]['Probability'].values[0]
        mock_specific.append((p1, p2, prob))
    mock_specific.sort(key=lambda x: x[2], reverse=True)
    for p1, p2, prob in mock_specific[:10]:
        print(f'  {p1:12} - {p2:12} : {prob:.4f}')
