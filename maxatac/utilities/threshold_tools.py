import numpy as np
import pandas as pd
import pyBigWig
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


def import_blacklist_mask(bigwig_path, chromosome, chromosome_length, bin_count):
    """
        Import the chromosome signal from a blacklist bigwig file and convert to a numpy array to use to mask out
        the regions to exclude in the AUPR analysis
        :return: blacklist_mask: A np.array the has True for regions that should be excluded from analysis
        """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.array(input_bw.stats(chromosome,
                                       0,
                                       chromosome_length,
                                       type="max",
                                       nBins=bin_count
                                       ),
                        dtype=float  # need it to have NaN instead of None
                        ) != 1  # Convert to boolean array, select areas that are not 1


def import_GoldStandard_array(bigwig_path, chromosome, chromosome_length, bin_count):
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(np.array(input_bw.stats(chromosome,
                                                     0,
                                                     chromosome_length,
                                                     type="max",
                                                     nBins=bin_count,
                                                     exact=True
                                                     ),
                                      dtype=float  # need it to have NaN instead of None
                                      )
                             ) > 0  # to convert to boolean array


def calculate_AUC_per_rank(PR_CURVE_DF, threshold):
        """
        Calculate the AUC at each rank on the AUPRC curve
        """
        tmp_df = PR_CURVE_DF[PR_CURVE_DF["Threshold"] >= threshold]

        # If we only have 1 point do not calculate AUC
        if len(tmp_df["Threshold"].unique()) == 1:
            return 0
        else:
            return metrics.auc(y=tmp_df["Precision"], x=tmp_df["Recall"])


def compute_calibration_curve(goldstandard, prediction, gs_bins=None, rand_bins=None):
    """
    Build a monotonic Precision/Recall/Threshold calibration curve (plus F1) for a
    single goldstandard/prediction signal pair. Mirrors the median-curve calculation
    in run_thresholding, factored out so it can also be run per individual cell type
    for visualization (see threshold.py) and for the metric-binning pipeline below.

    log2FC is only added when both gs_bins and rand_bins are supplied (needed for
    plotting in run_thresholding); omit both to get just Precision/Recall/Threshold/F1.
    """
    precision, recall, thresholds = precision_recall_curve(goldstandard, prediction)

    P = np.maximum.accumulate(np.array(precision))
    R = np.minimum.accumulate(np.array(recall))

    curve_df = pd.DataFrame({'Precision': P, 'Recall': R, "Threshold": np.insert(thresholds, 0, 0)})
    # Duplicate the last row rather than forcing Threshold=1: these are quant models,
    # so the max predicted value (and therefore the max threshold) can exceed 1.0.
    new_row = curve_df.tail(n=1)
    curve_df = pd.concat([curve_df, new_row], ignore_index=True)

    if gs_bins is not None and rand_bins is not None:
        random_precision = gs_bins / rand_bins
        curve_df['log2FC'] = np.log2(curve_df['Precision'] / random_precision)

    curve_df['F1'] = 2 * (curve_df['Precision'] * curve_df['Recall']) / (curve_df['Precision'] + curve_df['Recall'])

    return curve_df


def bin_curve_by_metric(curve_df, metric):
    """
    Bin a single cell type's calibration curve (Precision/Recall/Threshold/F1, e.g.
    from compute_calibration_curve) by one of its metric columns into 0.01-wide
    steps from 0.00 to 1.00, logging the threshold for each bin -- the same
    technique hybrid_subset uses for its Precision grid, generalized to Precision,
    Recall, and F1.

    Precision/Recall are expected to already be monotonic w.r.t. Threshold. F1 is
    NOT monotonic -- it rises then falls -- and is binned on its raw values here,
    with no accumulate applied, so it isn't artificially flattened.

    OPEN ISSUE (flagged, not resolved here): since F1 isn't forced monotonic, a
    given F1 bin can genuinely be reachable at more than one threshold (once on
    the way up, once on the way down). Only one row per bin is kept -- whichever
    maximizes F1 within the bin (matching hybrid_subset's own convention) -- so
    the other occurrence is dropped rather than retained as a separate row.
    """
    df = curve_df.copy()

    df['_bin'] = ((df[metric].to_numpy() * 100).round().astype(int) / 100).clip(0, 1)

    binned = df.loc[df.groupby('_bin')['F1'].idxmax()].copy()
    binned = binned.rename(columns={'_bin': 'Bin'})
    binned['Metric'] = metric

    return binned[['Metric', 'Bin', 'Precision', 'Recall', 'Threshold', 'F1']].reset_index(drop=True)


def extend_bins_to_full_grid(binned_df, full_bins=None):
    """
    Extend a single cell type's bin_curve_by_metric() result to cover every bin in
    full_bins (default: 0.00 to 1.00 in 0.01 steps), even bins this cell type's own
    curve never actually reached. Uses the same step-lookup + extrapolation
    technique as sample_curve_at_thresholds, keyed on Bin instead of Threshold:
    a forward merge_asof (the smallest available bin >= target) handles bins below
    this cell type's own range, and ffill (holding the last real match) handles
    bins above it.

    This matters because bin_curve_by_metric's per-cell-type output is already
    monotonic in Threshold vs. Bin (Precision/Recall are monotonic w.r.t. Threshold
    before binning, so each bin's threshold-preimage is an ordered, non-overlapping
    interval, and picking one row per interval preserves order). But different
    cell types have gaps in different places, so without this extension,
    median_bins_across_samples would compute each bin's median over a different
    subset of cell types -- and per-cell-type ordering does not, in general, imply
    ordering of the median once the pointwise (i.e. same-index) matching between
    adjacent bins is broken. Extending every cell type to the same full bin grid
    restores that matching, so the median stays monotonic too.
    """
    if full_bins is None:
        full_bins = np.round(np.arange(0, 1.01, 0.01), 2)

    binned_df = binned_df.drop_duplicates(subset='Bin').sort_values('Bin').reset_index(drop=True)
    grid = pd.DataFrame({'Bin': np.sort(np.unique(full_bins))})

    extended = pd.merge_asof(grid, binned_df, on='Bin', direction='forward')
    extended = extended.ffill().bfill()

    return extended


def median_bins_across_samples(binned_tables, full_bins=None):
    """
    Combine one metric's binned tables across cell types (each from
    bin_curve_by_metric, for the same metric) by taking the median of every
    numeric column -- Precision, Recall, Threshold, F1 -- for each bin in
    full_bins (default: 0.00 to 1.00 in 0.01 steps).

    Each cell type's table is first extended to the full bin grid
    (extend_bins_to_full_grid) so every bin's median is computed over the exact
    same set of cell types -- see that function's docstring for why this is
    needed to keep the result monotonic.
    """
    metric = binned_tables[0]['Metric'].iloc[0]
    extended_tables = [extend_bins_to_full_grid(t, full_bins) for t in binned_tables]

    merged = pd.concat(extended_tables, ignore_index=True)
    combined = merged.groupby('Bin', as_index=False)[['Precision', 'Recall', 'Threshold', 'F1']].median()
    combined.insert(0, 'Metric', metric)

    return combined.sort_values('Bin').reset_index(drop=True)


def recompute_f1(median_table):
    """
    Recompute F1 = 2*Precision*Recall/(Precision+Recall) from a median-across-
    cell-type table's own Precision and Recall columns, replacing whatever F1
    value is currently there.

    median_bins_across_samples takes the median of Precision, Recall, and F1 as
    independent columns; since each can come from a different cell type, the
    resulting row is a synthetic point that doesn't generally satisfy
    F1 = 2PR/(P+R) on its own (e.g. cell type A at P=0.9,R=0.3,F1=0.45 and cell
    type B at P=0.3,R=0.9,F1=0.45 median to P=0.6,R=0.6, but 2*0.6*0.6/1.2=0.6,
    not 0.45). Recomputing F1 from that same row's own median P/R keeps it
    internally consistent, at the cost of no longer necessarily matching the
    median of each cell type's own actual F1.
    """
    out = median_table.copy()
    out['F1'] = 2 * out['Precision'] * out['Recall'] / (out['Precision'] + out['Recall'])
    return out


def bin_median_table_by_f1(median_tables):
    """
    Build the F1 grid from one or more already median-across-cell-type,
    F1-recalculated tables (e.g. the Recall-binned table after recompute_f1),
    rather than binning each cell type's own raw F1 independently. Pools the
    given tables' rows, bins the pool by F1 (0.01 steps, clipped to [0, 1]), and
    keeps the max-F1 row per bin -- same technique as bin_curve_by_metric, just
    applied to the pooled, already-consensus rows instead of one cell type's raw
    curve. Since every candidate row's F1 is already internally consistent with
    its own Precision/Recall, so is whichever one gets picked per bin.

    Pass a single table (e.g. just the Recall table) to keep Threshold ordering
    sourced from one consistent table; pooling multiple tables mixes in more than
    one independently-ordered source and can reintroduce non-monotonic jumps.
    """
    # Drop the source tables' own Bin/Metric (Precision- or Recall-bin, now stale)
    # before computing the new F1-based Bin -- otherwise the rename below collides
    # with the pre-existing 'Bin' column instead of replacing it.
    pooled = pd.concat(median_tables, ignore_index=True).drop(columns=['Bin', 'Metric'])
    pooled['Bin'] = ((pooled['F1'].to_numpy() * 100).round().astype(int) / 100).clip(0, 1)

    binned = pooled.loc[pooled.groupby('Bin')['F1'].idxmax()].copy()
    binned['Metric'] = 'F1'

    return binned[['Metric', 'Bin', 'Precision', 'Recall', 'Threshold', 'F1']].sort_values('Bin').reset_index(drop=True)


def merge_binned_metrics(median_tables):
    """
    Union the per-metric median-across-cell-type tables (one each for Precision,
    Recall, F1 binning), sorted by Threshold. No deduplication across metrics.

    Each table is already internally unique by construction: median_bins_across_samples
    extends every cell type to the full 0.00-1.00 bin grid before taking the
    median, so there's exactly one row per bin, with no gaps and no
    within-metric collisions left to dedupe. And since build_cross_cell_type_threshold_table
    now derives the F1 grid (bin_median_table_by_f1) by re-selecting rows from the
    already-complete Precision/Recall pool rather than deriving new (Threshold,
    Precision, Recall) points, most F1 rows are full-entry duplicates of a
    Precision or Recall row *by construction*. Deduping across metrics (the
    previous behavior) silently erased the entire F1 grid as a result -- every
    F1 row lost to an identical Precision/Recall row that happened to list first.
    Each row is still a legitimate, distinct answer to "what's the
    median-consensus point for this bin of this metric," even when it numerically
    coincides with a different metric's bin.
    """
    merged = pd.concat(median_tables, ignore_index=True)
    merged = merged.sort_values('Threshold').reset_index(drop=True)

    return merged


def build_cross_cell_type_threshold_table(curves):
    """
    End-to-end pipeline: given one calibration curve per cell type (DataFrames
    with Precision/Recall/Threshold/F1, e.g. from compute_calibration_curve):
      1. Bin each cell type's curve by Precision and by Recall separately
         (bin_curve_by_metric).
      2. Extend each cell type's binned table to the full 0.00-1.00 bin grid
         (extend_bins_to_full_grid, inside median_bins_across_samples) so every
         bin's median is computed over the same set of cell types, keeping the
         result monotonic.
      3. Combine each metric's bins across cell types via the median
         (median_bins_across_samples), then recompute F1 from each table's own
         median Precision/Recall (recompute_f1) instead of independently
         medianing F1 across cell types.
      4. Build the F1 grid by binning the F1-recalculated Recall table by F1
         (bin_median_table_by_f1) -- using only the Recall table, not Precision,
         as the source. (Recall is fully populated across all 101 bins and
         already the sole source of Threshold ordering here; pooling in the
         Precision table as well previously mixed in a second, independently-
         ordered source and produced a visible dip near the F1 peak.)
      5. Merge the three metrics' tables into one, deduped on the full entry
         (merge_binned_metrics).
    """
    precision_tables = [bin_curve_by_metric(curve, 'Precision') for curve in curves]
    recall_tables = [bin_curve_by_metric(curve, 'Recall') for curve in curves]

    precision_median = recompute_f1(median_bins_across_samples(precision_tables))
    recall_median = recompute_f1(median_bins_across_samples(recall_tables))

    f1_binned = bin_median_table_by_f1([recall_median])

    return merge_binned_metrics([precision_median, recall_median, f1_binned])


def sample_curve_at_thresholds(curve_df, threshold_values):
    """
    Evaluate a monotonic step-function calibration curve (as built by
    compute_calibration_curve) at a shared grid of threshold values. Uses a
    forward step lookup: the value at T is whatever the curve reports at the
    smallest recorded threshold >= T, matching '>= T' cutoff semantics.
    Threshold values beyond the curve's own range are extrapolated by holding
    its nearest boundary row.
    """
    curve_df = curve_df.drop_duplicates(subset='Threshold').sort_values('Threshold').reset_index(drop=True)
    shared = pd.DataFrame({'Threshold': np.sort(np.unique(threshold_values))})

    sampled = pd.merge_asof(shared, curve_df, on='Threshold', direction='forward')
    sampled = sampled.ffill().bfill()

    return sampled


def hybrid_subset(df, last_row_df_source, n_bins=300):
    """
    Downsample a threshold-calibration dataframe using a hybrid grid:
    - Precision bins at 0.01 steps (0.00 ... 1.00), keeping the max-F1 row per bin
    - Threshold bins (log-spaced, denser at small values), keeping the max-F1 row per bin
    last_row_df_source is re-appended before binning so its row is guaranteed to survive
    the downsampling (it otherwise loses out on max-F1 selection, e.g. the Recall=0 endpoint).
    Returns the union of both grids, sorted by Threshold.
    """
    df = df.copy()
    df = pd.concat([df, last_row_df_source], ignore_index=True)

    # Precision grid
    df['Precision_bin'] = (df['Monotonic_Precision'] * 100).round().astype(int) / 100
    df_precision = df.loc[df.groupby('Precision_bin')['Monotonic_F1'].idxmax()]

    # Threshold grid (log-spaced, denser at small values)
    threshold_bins = np.geomspace(df['Threshold'].min() + 1e-12, df['Threshold'].max(), n_bins)
    df['Threshold_bin'] = pd.cut(df['Threshold'], bins=threshold_bins)
    df_threshold = df.loc[df.groupby('Threshold_bin', observed=True)['Monotonic_F1'].idxmax()]

    # Combine. Dedupe on Threshold specifically (not drop_duplicates() over all
    # columns): df_precision lacks the Threshold_bin column added after it was
    # sliced off, so a row selected by both grids would otherwise compare as
    # non-identical (real Threshold_bin vs. NaN) and survive as a false duplicate.
    df_hybrid = pd.concat([df_precision, df_threshold], ignore_index=True)
    df_hybrid = df_hybrid.drop_duplicates(subset='Threshold').sort_values('Threshold').reset_index(drop=True)
    del df_hybrid['Precision_bin']
    del df_hybrid['Threshold_bin']

    return df_hybrid