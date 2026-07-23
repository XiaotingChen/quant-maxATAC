import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool, Manager
import time
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank, hybrid_subset, compute_calibration_curve, sample_curve_at_thresholds
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

    results_filename = os.path.join(output_dir, args.prefix + ".tsv")

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


    # Stack each cell type's Prediction/GoldStandard as a pair of columns. This layout
    # is used both to build per-cell-type curves for plotting, and (in "median" mode)
    # to aggregate across cell types before computing the curve.
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

    if args.aggregation == "median":
        # Per-bin median Prediction/GoldStandard across cell types. GoldStandard is
        # kept only where the median is exactly 1, i.e. a majority of cell types agree.
        DF_median = pd.DataFrame([])
        DF_median['Prediction'] = np.nanmedian(DF[range(0, np.shape(DF)[1], 2)], axis=1)
        DF_median['GoldStandard'] = np.nanmedian(DF[range(1, np.shape(DF)[1], 2)], axis=1)
        DF_median.loc[DF_median['GoldStandard'] != 1, 'GoldStandard'] = 0

        agg_goldstandard = DF_median['GoldStandard'][blacklist_mask]
        agg_prediction = DF_median['Prediction'][blacklist_mask]
    else:
        # Default: pool every cell type's bins together (stacked as rows) into one
        # curve, rather than aggregating them into a single signal first.
        DF_pooled = pd.concat(
            [DF[[2 * i, 2 * i + 1]].rename(columns={2 * i: 'Prediction', 2 * i + 1: 'GoldStandard'})
             for i in range(num_cell_types)],
            ignore_index=True,
        )
        DF_pooled.loc[DF_pooled['GoldStandard'] != 1, 'GoldStandard'] = 0
        # DF_pooled is block-major (all of cell type 0's bins, then all of cell type
        # 1's, ...), so the mask must cycle once per block via np.tile, not np.repeat
        # (which would smear each bin's mask across a run of num_cell_types rows).
        blacklist_pooled = np.tile(blacklist_mask, num_cell_types)

        agg_goldstandard = DF_pooled['GoldStandard'][blacklist_pooled]
        agg_prediction = DF_pooled['Prediction'][blacklist_pooled]

    precision, recall, thresholds = precision_recall_curve(agg_goldstandard, agg_prediction)
    
    
    # Create a dataframe from the results
    df = pd.DataFrame({'Precision': precision, 'Recall': recall, "Threshold": np.insert(thresholds, 0, 0)})
    
    
    P = np.maximum.accumulate((np.array(df.Precision)))
    R = np.minimum.accumulate((np.array(df.Recall)))
    
    PR_CURVE_DF = pd.DataFrame({'Precision': P, 'Recall': R, "Threshold": np.insert(thresholds, 0, 0)})
    
    # Duplicate the last row so hybrid_subset has a trailing row to drop and re-anchor
    # on. Not forced to Threshold=1 here: these are quant models, so the max predicted
    # value (and therefore the max threshold) routinely exceeds 1.0.
    new_row = PR_CURVE_DF.tail(n=1)
    PR_CURVE_DF = pd.concat([PR_CURVE_DF, new_row], ignore_index=True)
    
    # total_gs_bins: Total GS bins across for each CT
    PR_CURVE_DF["Total_Avg_GoldStandard_Bins"] = int(np.mean(total_gs_bins))
    
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
    
    # Find the number of non-blacklisted bins in chr of interest
    rand_bins = df.query('chr == @args.chromosomes').shape[0]
    
    # Random Precision
    PR_CURVE_DF['Random_Precision'] = PR_CURVE_DF['Total_Avg_GoldStandard_Bins']/rand_bins
    
    logging.info("Calculate log2FC for each threshold")   
    
    # Log2FC
    PR_CURVE_DF['log2FC_Precision_Random_Precision'] = np.log2(PR_CURVE_DF["Precision"]/PR_CURVE_DF["Random_Precision"])
    
    logging.info("Calculate F1 Score for each threshold")

    # F1 Score
    PR_CURVE_DF['F1_Score'] = 2 * (PR_CURVE_DF["Precision"] * PR_CURVE_DF["Recall"]) / (PR_CURVE_DF["Precision"] + PR_CURVE_DF["Recall"])
    
    del PR_CURVE_DF['Total_Avg_GoldStandard_Bins']
    del PR_CURVE_DF['Random_Precision']
    

    PR_CURVE_DF.columns = 	['Monotonic_Precision', 'Monotonic_Recall', 'Threshold', 'Monotonic_log2FC', 'Monotonic_F1']
    PR_CURVE_DF.to_csv(results_filename, sep="\t", header=True, index=False)

    logging.info("Generating hybrid-subset (downsampled) threshold calibration table")

    # Drop the duplicate tail-extension row, then anchor on the row that follows it so
    # the true endpoint is guaranteed to survive the max-F1 binning in hybrid_subset.
    df_full = PR_CURVE_DF.drop(PR_CURVE_DF.index[-1])
    last_row_df_source = df_full.iloc[-1:]

    hybrid_df = hybrid_subset(df_full, last_row_df_source, n_bins=300)

    hybrid_filename = os.path.join(output_dir, args.prefix + "_hybrid.tsv")
    hybrid_df.to_csv(hybrid_filename, sep="\t", header=True, index=False)

    logging.info("Building per-cell-type curves and plotting the validation statistics v. threshold values")

    # Plot on the hybrid-subset's threshold grid: build each cell type's own monotonic
    # curve, then resample it onto hybrid_df's unique Threshold values (step lookup,
    # extrapolating past a curve's own range) so every line shares the same x-axis.
    shared_thresholds = hybrid_df['Threshold'].to_numpy()
    cell_type_curves = []
    for i in range(len(OUT[0])):
        ct_curve = compute_calibration_curve(
            DF[2 * i + 1][blacklist_mask], DF[2 * i][blacklist_mask], total_gs_bins[i], rand_bins
        )
        ct_curve_sampled = sample_curve_at_thresholds(ct_curve, shared_thresholds)
        cell_type_curves.append({'name': os.path.basename(lst_of_bws[i]), 'curve': ct_curve_sampled})

    median_curve_for_plot = hybrid_df.rename(columns={
        'Monotonic_Precision': 'Precision',
        'Monotonic_Recall': 'Recall',
        'Monotonic_log2FC': 'log2FC',
        'Monotonic_F1': 'F1',
    })
    plot_threshold_calibration_stats(median_curve_for_plot, cell_type_curves, results_filename, args.prefix)



