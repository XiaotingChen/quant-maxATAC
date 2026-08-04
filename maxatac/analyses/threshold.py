import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool, Manager
import time
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank, hybrid_subset, compute_calibration_curve, sample_curve_at_thresholds, build_cross_cell_type_threshold_table
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

    # Plot each metric panel on that metric's *own* threshold grid, not the full
    # cross_celltype_table (which mixes Precision-, Recall-, and F1-binned
    # consensus thresholds together via merge_binned_metrics). Resampling every
    # cell type's raw curve at that mixed set meant, e.g., the Precision panel's
    # colored lines were evaluated at ~3x more (and differently-sourced) threshold
    # points than the black Precision-only median line -- a real, faithfully
    # reported difference in each cell type's own curve, but not one comparable
    # point-for-point against the median. Resampling per metric instead means each
    # panel's colored lines and its median line share exactly the same threshold
    # set (and, since each metric's table has one row per 0.01 bin, the same
    # order of resolution).
    metric_thresholds = {
        metric: cross_celltype_table.loc[cross_celltype_table['Metric'] == metric, 'Threshold'].to_numpy()
        for metric in ('Precision', 'Recall', 'F1')
    }
    cell_type_curves = []
    for i, ct_curve in enumerate(raw_cell_type_curves):
        curves_by_metric = {
            metric: sample_curve_at_thresholds(ct_curve, thresholds)
            for metric, thresholds in metric_thresholds.items()
        }
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


