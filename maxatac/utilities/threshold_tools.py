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


def compute_calibration_curve(goldstandard, prediction, gs_bins, rand_bins):
    """
    Build a monotonic Precision/Recall/Threshold/log2FC/F1 calibration curve for a
    single goldstandard/prediction signal pair. Mirrors the median-curve calculation
    in run_thresholding, factored out so it can also be run per individual cell type
    for visualization (see threshold.py).
    """
    precision, recall, thresholds = precision_recall_curve(goldstandard, prediction)

    P = np.maximum.accumulate(np.array(precision))
    R = np.minimum.accumulate(np.array(recall))

    curve_df = pd.DataFrame({'Precision': P, 'Recall': R, "Threshold": np.insert(thresholds, 0, 0)})
    # Duplicate the last row rather than forcing Threshold=1: these are quant models,
    # so the max predicted value (and therefore the max threshold) can exceed 1.0.
    new_row = curve_df.tail(n=1)
    curve_df = pd.concat([curve_df, new_row], ignore_index=True)

    random_precision = gs_bins / rand_bins
    curve_df['log2FC'] = np.log2(curve_df['Precision'] / random_precision)
    curve_df['F1'] = 2 * (curve_df['Precision'] * curve_df['Recall']) / (curve_df['Precision'] + curve_df['Recall'])

    return curve_df


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