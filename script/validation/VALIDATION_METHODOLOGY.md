# Validation Methodology

## Key Principles

### 1. Do NOT treat non-annotated edges as negatives

**Databases provide a high-confidence but incomplete positive set.**

- Absence of annotation ≠ absence of interaction
- Global precision is not identifiable from current knowledge
- This is standard and defensible in interaction prediction

### 2. Evaluate as "enrichment / retrieval of known positives"

Instead of precision over all predictions, we evaluate:

#### A. Recall@K / Hit-rate@K

Sort edges by posterior probability and report:
- # known positives in top 50 / 100 / 200 / 500 / 1000
- Compare against a random ranking baseline

This answers: **"How enriched are known interactions among top predictions?"**

#### B. Fold-Enrichment@K

```
FE@K = (positives in top K / K) / (positives in all / N)
     = (#positives in top K / K) / (baseline_rate)
```

Where:
- `baseline_rate = #positives in all candidates / N`
- This accounts for sparse truth sets

#### C. Positive-set fraction (carefully worded)

**Avoid calling it "precision."** Instead call it:
- **"Positive-set fraction"**
- **"Database-supported fraction"**
- **"Fraction of predictions supported by reference positives"**

This measures: **"What fraction of top K predictions are validated by databases/PDB?"**

## Metrics Reported

1. **Recall@K**: Fraction of known positives recovered in top K predictions
2. **Fold-Enrichment@K**: Enrichment of known positives in top K vs. random baseline
3. **Positive-set fraction@K**: Fraction of top K predictions supported by reference positives
4. **Enrichment at percentages**: Fold-enrichment and recall at top 1%, 5%, 10%, 20%, 50%, etc.

## Terminology Notes

- ✅ **DO use**: "enrichment", "retrieval", "recall", "hit-rate", "positive-set fraction", "database-supported"
- ❌ **AVOID**: "precision" (implies negatives), "accuracy" (requires negatives)
