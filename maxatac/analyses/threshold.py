import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool, Manager
import time
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank, hybrid_subset, compute_calibration_curve, bin_curve_by_metric, extend_bins_to_full_grid, bin_median_table_by_f1, merge_binned_metrics, build_cross_cell_type_threshold_table
from maxatac.utilities.plot import plot_threshold_calibration_stats
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import pybedtools
import logging


def extract_pred_gs_bw(bigwig_file, training_data_dict, chrom_name, chrom_length, bin_count):
    start = time.time()
    bw_name = bigwig_file.split("/")[-1]
    chrom_vals = get_bigwig_stats(bigwig_file, chrom_name, chrom_length, bin_count)
    print(training_data_dict[bigwig_file])
    
    
    predictions = np.empty(1, dtype=np.float64)
    gold_standard = np.empty(1, dtype=np.float64)
    goldstandard_array = import_GoldStandard_array(training_data_dict[bigwig_file], chrom_name, chrom_length, bin_count)
    
    tot_gs_bins = len(np.argwhere(goldstandard_array == True))

    predictions = np.concatenate([predictions, chrom_vals])
    gold_standard = np.concatenate([gold_standard, goldstandard_array])
    
    end = time.time()
    print('total time (s)= ' + str(end-start), "____________", bw_name)
    
    return predictions, gold_standard, tot_gs_bins

def run_thresholding(args):
    """
    :param args:
    :return:
    """
    # Make the output directory
    output_dir = get_dir(args.output_dir)

    chromosome_sizes_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    meta_DF = pd.read_table(args.meta_file)

    training_data_dict = pd.Series(meta_DF["Binding_File"].values,index=meta_DF["Prediction"]).to_dict()

    # results_filename only fed the old pooled/median .tsv write and, before the plot
    # was repointed at cross_celltype_filename, the plot's filename base. No longer
    # used by anything active; kept here commented out alongside the code that used it.
    # results_filename = os.path.join(output_dir, args.prefix + ".tsv")

    # Loop through the chromosomes and average the values across files
    OUT=[]
    for chrom_name, chrom_length in chromosome_sizes_dictionary.items():
        bin_count = int(int(chrom_length) / int(args.bin_size))  # need to floor the number
        
        blacklist_mask = import_blacklist_mask(args.blacklist_bw, chrom_name, chrom_length, bin_count)

        lst_of_bws=list(training_data_dict.keys())
        
        pool = Pool(int(multiprocessing.cpu_count())) 
        output = pool.starmap(
            extract_pred_gs_bw,
            [(bigwig, training_data_dict, chrom_name, chrom_length, bin_count) for bigwig in lst_of_bws]
                            )
        OUT.append(output)


    # Stack each cell type's Prediction/GoldStandard as a pair of columns, used below
    # to build each cell type's own calibration curve.
    DF=pd.DataFrame([])
    total_gs_bins = []
    for i in range(len(OUT[0])):
        df = pd.DataFrame([])

        df['Prediction'] = OUT[0][i][0][:bin_count].tolist()
        df['GoldStandard'] = OUT[0][i][1][:bin_count].tolist()

        gs_bins = OUT[0][i][2]
        DF = pd.concat([DF, df], axis=1, ignore_index=True)
        total_gs_bins.append(gs_bins)

    num_cell_types = len(OUT[0])

    # ------------------------------------------------------------------------
    # Old pooled/median aggregation + single-curve PR/F1/log2FC table + hybrid_subset
    # downsampling. Superseded by the cross-cell-type threshold table below (each
    # cell type's own curve, binned per-metric, combined via median-across-cell-type
    # thresholds -- see build_cross_cell_type_threshold_table). Kept here, commented
    # out, for reference.
    # ------------------------------------------------------------------------
    # if args.aggregation == "median":
    #     # Per-bin median Prediction/GoldStandard across cell types. GoldStandard is
    #     # kept only where the median is exactly 1, i.e. a majority of cell types agree.
    #     DF_median = pd.DataFrame([])
    #     DF_median['Prediction'] = np.nanmedian(DF[range(0, np.shape(DF)[1], 2)], axis=1)
    #     DF_median['GoldStandard'] = np.nanmedian(DF[range(1, np.shape(DF)[1], 2)], axis=1)
    #     DF_median.loc[DF_median['GoldStandard'] != 1, 'GoldStandard'] = 0
    #
    #     agg_goldstandard = DF_median['GoldStandard'][blacklist_mask]
    #     agg_prediction = DF_median['Prediction'][blacklist_mask]
    # else:
    #     # Default: pool every cell type's bins together (stacked as rows) into one
    #     # curve, rather than aggregating them into a single signal first.
    #     DF_pooled = pd.concat(
    #         [DF[[2 * i, 2 * i + 1]].rename(columns={2 * i: 'Prediction', 2 * i + 1: 'GoldStandard'})
    #          for i in range(num_cell_types)],
    #         ignore_index=True,
    #     )
    #     DF_pooled.loc[DF_pooled['GoldStandard'] != 1, 'GoldStandard'] = 0
    #     # DF_pooled is block-major (all of cell type 0's bins, then all of cell type
    #     # 1's, ...), so the mask must cycle once per block via np.tile, not np.repeat
    #     # (which would smear each bin's mask across a run of num_cell_types rows).
    #     blacklist_pooled = np.tile(blacklist_mask, num_cell_types)
    #
    #     agg_goldstandard = DF_pooled['GoldStandard'][blacklist_pooled]
    #     agg_prediction = DF_pooled['Prediction'][blacklist_pooled]
    #
    # precision, recall, thresholds = precision_recall_curve(agg_goldstandard, agg_prediction)
    #
    # # Create a dataframe from the results
    # df = pd.DataFrame({'Precision': precision, 'Recall': recall, "Threshold": np.insert(thresholds, 0, 0)})
    #
    # P = np.maximum.accumulate((np.array(df.Precision)))
    # R = np.minimum.accumulate((np.array(df.Recall)))
    #
    # PR_CURVE_DF = pd.DataFrame({'Precision': P, 'Recall': R, "Threshold": np.insert(thresholds, 0, 0)})
    #
    # # Duplicate the last row so hybrid_subset has a trailing row to drop and re-anchor
    # # on. Not forced to Threshold=1 here: these are quant models, so the max predicted
    # # value (and therefore the max threshold) routinely exceeds 1.0.
    # new_row = PR_CURVE_DF.tail(n=1)
    # PR_CURVE_DF = pd.concat([PR_CURVE_DF, new_row], ignore_index=True)
    #
    # # total_gs_bins: Total GS bins across for each CT
    # PR_CURVE_DF["Total_Avg_GoldStandard_Bins"] = int(np.mean(total_gs_bins))
    #
    # # Random Precision
    # PR_CURVE_DF['Random_Precision'] = PR_CURVE_DF['Total_Avg_GoldStandard_Bins']/rand_bins
    #
    # logging.info("Calculate log2FC for each threshold")
    #
    # # Log2FC
    # PR_CURVE_DF['log2FC_Precision_Random_Precision'] = np.log2(PR_CURVE_DF["Precision"]/PR_CURVE_DF["Random_Precision"])
    #
    # logging.info("Calculate F1 Score for each threshold")
    #
    # # F1 Score
    # PR_CURVE_DF['F1_Score'] = 2 * (PR_CURVE_DF["Precision"] * PR_CURVE_DF["Recall"]) / (PR_CURVE_DF["Precision"] + PR_CURVE_DF["Recall"])
    #
    # del PR_CURVE_DF['Total_Avg_GoldStandard_Bins']
    # del PR_CURVE_DF['Random_Precision']
    #
    # PR_CURVE_DF.columns = 	['Monotonic_Precision', 'Monotonic_Recall', 'Threshold', 'Monotonic_log2FC', 'Monotonic_F1']
    # PR_CURVE_DF.to_csv(results_filename, sep="\t", header=True, index=False)
    #
    # logging.info("Generating hybrid-subset (downsampled) threshold calibration table")
    #
    # # Drop the duplicate tail-extension row, then anchor on the row that follows it so
    # # the true endpoint is guaranteed to survive the max-F1 binning in hybrid_subset.
    # df_full = PR_CURVE_DF.drop(PR_CURVE_DF.index[-1])
    # last_row_df_source = df_full.iloc[-1:]
    #
    # hybrid_df = hybrid_subset(df_full, last_row_df_source, n_bins=300)
    #
    # hybrid_filename = os.path.join(output_dir, args.prefix + "_hybrid.tsv")
    # hybrid_df.to_csv(hybrid_filename, sep="\t", header=True, index=False)
    # ------------------------------------------------------------------------
    # End of old pooled/median + hybrid_subset code.
    # ------------------------------------------------------------------------

    # Create a bedtools object that is a windowed genome
    BED_df_bedtool = pybedtools.BedTool().window_maker(g=args.chrom_sizes, w=args.bin_size)

    # Create a blacklist object form the blacklist bed
    # TODO: I do not think we need to get the blacklist bed location like this since it is a part of the args now.
    blacklist_bed_location = ".".join([args.blacklist_bw.split(".")[0],'bed'])
    blacklist_bedtool = pybedtools.BedTool(blacklist_bed_location)

    # Remove the blacklisted regions from the windowed genome object
    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    # Create a dataframe from the BedTools object
    df = blacklisted_df.to_dataframe()

    # Rename the columns
    df.columns = ["chr", "start", "stop"]

    # Find the number of non-blacklisted bins in chr of interest. Still needed as the
    # log2FC denominator for each cell type's calibration curve below.
    rand_bins = df.query('chr == @args.chromosomes').shape[0]

    logging.info("Building per-cell-type calibration curves")

    raw_cell_type_curves = []
    for i in range(num_cell_types):
        ct_curve = compute_calibration_curve(
            DF[2 * i + 1][blacklist_mask], DF[2 * i][blacklist_mask], total_gs_bins[i], rand_bins
        )
        raw_cell_type_curves.append(ct_curve)

    logging.info("Building cross-cell-type threshold table (per-metric binning + median across cell types)")

    # For each cell type independently: bin its own curve by Precision, Recall, and
    # F1 (0.01 steps each), then take the median Threshold (and Precision/Recall/F1)
    # across cell types per bin, then merge the three metrics' tables, deduped on the
    # full entry. This sidesteps the "median" aggregation mode's majority-consensus
    # GoldStandard issue entirely, since it never aggregates the raw signal -- only
    # the resulting per-cell-type thresholds.
    cross_celltype_table = build_cross_cell_type_threshold_table(raw_cell_type_curves)
    cross_celltype_filename = os.path.join(output_dir, args.prefix + "_cross_celltype.tsv")
    cross_celltype_table.to_csv(cross_celltype_filename, sep="\t", header=True, index=False)

    logging.info("Plotting the validation statistics v. threshold values")

    # Plot each cell type's OWN binned+extended curve directly (bin_curve_by_metric +
    # extend_bins_to_full_grid -- the same per-cell-type step build_cross_cell_type_
    # threshold_table already does before taking the cross-cell-type median), rather
    # than resampling its raw curve at the median table's own consensus thresholds.
    # Resampling at a foreign (median) threshold meant one cell type's curve could get
    # evaluated at a threshold badly distorted by a *different* cell type's own gap:
    # when a cell type has no real data between two precision levels, extend_bins_to_
    # full_grid forward-fills a distant, much-higher threshold across that whole gap,
    # which then drags the cross-cell-type median threshold up for those bins too --
    # confirmed concretely in real data (ATF1: bin 0.83->0.84 jumped Precision
    # 0.830->0.918 and Threshold 10.04->12.74 in a single 0.01-bin step). Evaluating
    # the OTHER cell type's smooth curve at that distorted threshold overstated its
    # precision there. Using each cell type's own table means its line only ever
    # reflects its own genuine Bin/Threshold/Precision relationship, never
    # contaminated by another cell type's gap -- any crossing that remains reflects
    # real disagreement between cell types on threshold-for-precision, not a
    # resampling artifact.
    cell_type_curves = []
    for i, ct_curve in enumerate(raw_cell_type_curves):
        random_precision_i = total_gs_bins[i] / rand_bins
        precision_curve = extend_bins_to_full_grid(bin_curve_by_metric(ct_curve, 'Precision'))
        recall_curve = extend_bins_to_full_grid(bin_curve_by_metric(ct_curve, 'Recall'))

        # Save this cell type's own binned threshold table standalone, same schema
        # as cross_celltype.tsv (Metric/Bin/Precision/Recall/Threshold/F1), so
        # per-sample calibration data is available on its own, not just embedded in
        # the plot. F1 rows are picked via bin_median_table_by_f1 (same technique
        # the cross-cell-type table's own F1 block uses) rather than reusing
        # recall_curve verbatim: a saved table has no continuity implication the way
        # a plotted line does, so the fuller, properly-selected F1 rows are fine
        # here even though they're not used for the plotted F1 line below.
        sample_table = merge_binned_metrics([precision_curve, recall_curve, bin_median_table_by_f1([recall_curve])])
        sample_name = os.path.splitext(os.path.basename(lst_of_bws[i]))[0]
        sample_table.to_csv(os.path.join(output_dir, sample_name + ".tsv"), sep="\t", header=True, index=False)

        # F1 is not monotonic in Threshold (rises then falls). Re-binning it by its
        # own value -- even from a single, already-Threshold-monotonic source table --
        # still reintroduces non-monotonic jumps: near the peak, two different
        # thresholds (one on the rise, one on the fall) can land in the same F1 bin,
        # and idxmax can flip between them as the target bin shifts slightly (this
        # was tested directly and produced a visibly zigzagging line). Re-binning by
        # F1 only exists to build a *shared consensus* grid across multiple cell
        # types for the median table; a single cell type doesn't need that -- its own
        # Recall-binned curve already carries a real, Threshold-monotonic F1 column
        # straight from its own curve, so just reuse it directly.
        curves_by_metric = {'Precision': precision_curve, 'Recall': recall_curve, 'F1': recall_curve.copy()}
        for curve in curves_by_metric.values():
            curve['log2FC'] = np.log2(curve['Precision'] / random_precision_i)
        cell_type_curves.append({'name': os.path.basename(lst_of_bws[i]), 'curves': curves_by_metric})

    # plot_threshold_calibration_stats also needs a log2FC column, which
    # build_cross_cell_type_threshold_table doesn't produce (it only tracks
    # Precision/Recall/Threshold/F1). Add it here for the plot only -- not written to
    # cross_celltype.tsv -- using the same mean-gs-bins Random_Precision the old
    # pooled/median code above used.
    random_precision = np.mean(total_gs_bins) / rand_bins
    median_curve_for_plot = cross_celltype_table.copy()
    median_curve_for_plot['log2FC'] = np.log2(median_curve_for_plot['Precision'] / random_precision)

    plot_threshold_calibration_stats(median_curve_for_plot, cell_type_curves, cross_celltype_filename, args.prefix)


