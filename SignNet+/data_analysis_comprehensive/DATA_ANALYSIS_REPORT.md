# SignNet Dataset Analysis Report

---

## Executive Summary

- **Total Samples**: 6,841
- **Total Frames**: 963,664
- **Total Glosses**: 77,271
- **Unique Glosses**: 1,295
- **Dataset**: RWTH-PHOENIX-2014
- **Feature Dimension**: 543 landmarks × 2 coordinates (X, Y)

## Dataset Splits

| Split | Samples | Frames | Glosses | Avg Frames/Sample |
|-------|---------|--------|---------|-------------------|
| Train | 5,672 | 799,006 | 65,227 | 140.9 |
| Dev | 540 | 75,186 | 5,540 | 139.2 |
| Test | 629 | 89,472 | 6,504 | 142.2 |

## Class Distribution

- **Class Imbalance Ratio**: 3619.0:1
- **Most Frequent Gloss**: 3,619 occurrences
- **Least Frequent Gloss**: 1 occurrence(s)

### Top-K Coverage

| Top-K | Coverage | Percentage |
|-------|----------|------------|
| 50 | 46,238 | 59.8% |
| 100 | 59,339 | 76.8% |
| 200 | 69,513 | 90.0% |
| 300 | 73,157 | 94.7% |
| 500 | 75,675 | 97.9% |

## Recommendations

### Model Training Strategy

1. **Sequence Length**: Use max_length=214 (covers 95% of samples)
2. **Class Selection**: Start with Top-200 glosses (90.0% coverage)
3. **Batch Size**: Start with batch_size=8 (memory permitting)

### Data Quality

- ✅ All samples passed quality checks
- ✅ X,Y coordinates validated
- ✅ Custom landmarks removed
- ✅ Z-dimension dropped (mostly near-zero)
