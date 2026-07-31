import logging
import os
import timeit

import numpy as np
import pandas as pd

from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.genome_tools import load_bigwig, import_bigwig_stats_array
from maxatac.utilities.constants import INPUT_LENGTH


def _select_threshold(benchmark_tsv, select_by="precision", precision_level=0.7,
                       recall_level=0.25, threshold_override=None):
    if threshold_override is not None:
        return float(threshold_override)

    df = pd.read_csv(benchmark_tsv, sep="\t").sort_values("Threshold").reset_index(drop=True)

    # ChromosomeAUPRC's Precision/Recall are raw sklearn precision_recall_curve
    # output, not guaranteed monotonic in Threshold order (sampling noise near
    # ties/duplicate scores). Force the same monotone envelope threshold.py's
    # superseded pooled/median path used to apply -- precision never dips as
    # threshold rises, recall never blips upward -- so idxmax() below can't land
    # on a noisy local spike right at the constraint boundary.
    df["Precision"] = np.maximum.accumulate(df["Precision"])
    df["Recall"] = np.minimum.accumulate(df["Recall"])

    if select_by == "recall":
        valid = df[df["Recall"] >= recall_level]
        if valid.empty:
            logging.warning(
                "No threshold achieves recall >= %.2f; using threshold with highest recall"
                % recall_level
            )
            return float(df.loc[df["Recall"].idxmax(), "Threshold"])
        return float(valid.loc[valid["Precision"].idxmax(), "Threshold"])

    valid = df[df["Precision"] >= precision_level]
    if valid.empty:
        logging.warning(
            "No threshold achieves precision >= %.2f; using threshold with highest precision"
            % precision_level
        )
        return float(df.loc[df["Precision"].idxmax(), "Threshold"])
    return float(valid.loc[valid["Recall"].idxmax(), "Threshold"])


def _classify_indices(pred_array, gs_array, threshold):
    pred_pos = pred_array >= threshold
    gs_pos = gs_array > 0
    return {
        "TP": np.where(pred_pos & gs_pos)[0],
        "FP": np.where(pred_pos & ~gs_pos)[0],
        "FN": np.where(~pred_pos & gs_pos)[0],
        "TN": np.where(~pred_pos & ~gs_pos)[0],
    }


def _valid_windows(indices, bin_size, chrom_length):
    half = INPUT_LENGTH // 2
    bin_starts = indices * bin_size
    midpoints = bin_starts + bin_size // 2
    win_starts = midpoints - half
    win_ends = win_starts + INPUT_LENGTH
    valid = (win_starts >= 0) & (win_ends <= chrom_length)
    return win_starts[valid], win_ends[valid], bin_starts[valid]


def run_model_interpreting(args):
    start_time = timeit.default_timer()
    output_dir = get_dir(args.output_directory)

    threshold = _select_threshold(
        args.benchmark_tsv,
        select_by=args.select_by,
        precision_level=args.precision_level,
        recall_level=args.recall_level,
        threshold_override=args.threshold
    )
    logging.info("Using classification threshold: %.6f" % threshold)

    class_data = {
        label: {"prediction": [], "gs": [], "motif": [], "atac": [], "motif_max": []}
        for label in ("TP", "FP", "TN", "FN")
    }
    coord_rows = []

    with load_bigwig(args.prediction) as pred_bw, \
         load_bigwig(args.gold_standard) as gs_bw, \
         load_bigwig(args.motif) as motif_bw, \
         load_bigwig(args.atac) as atac_bw:

        chrom_sizes = pred_bw.chroms()
        chromosomes = [c for c in args.chromosomes if c in chrom_sizes]

        for chrom in chromosomes:
            chrom_length = chrom_sizes[chrom]
            bin_count = chrom_length // args.bin_size

            if bin_count == 0:
                continue

            logging.info("Processing %s (%d bins)" % (chrom, bin_count))

            pred_array = np.nan_to_num(
                import_bigwig_stats_array(pred_bw, chrom, chrom_length, args.agg_function, bin_count)
            )
            gs_array = np.nan_to_num(
                import_bigwig_stats_array(gs_bw, chrom, chrom_length, args.agg_function, bin_count)
            )
            # Mirror __import_prediction_array__ / __import_goldstandard_array__
            # (benchmarking_tools.py): under "sum" aggregation a bin's value is a
            # base-pair-integrated total, not a per-base score, so both arrays must be
            # divided by bin_size to stay on the same scale as max/mean aggregation
            # before pred_array is compared against the externally-sourced `threshold`
            # and gs_array is re-binarized against agg_threshold.
            if args.agg_function == "sum":
                pred_array = pred_array / args.bin_size
                gs_array = np.where((gs_array / args.bin_size) >= args.agg_threshold, 1.0, 0.0)

            class_indices = _classify_indices(pred_array, gs_array, threshold)

            for label, indices in class_indices.items():
                if len(indices) == 0:
                    continue

                win_starts, win_ends, bin_starts = _valid_windows(
                    indices, args.bin_size, chrom_length
                )

                windows = list(zip(win_starts.tolist(), win_ends.tolist(), bin_starts.tolist()))

                if label == "TN" and args.tn_limit is not None and len(windows) > args.tn_limit:
                    rng = np.random.default_rng(seed=42)
                    chosen = rng.choice(len(windows), size=args.tn_limit, replace=False)
                    windows = [windows[i] for i in sorted(chosen)]

                for win_start, win_end, bin_start in windows:
                    pred_arr = np.nan_to_num(
                        np.array(pred_bw.values(chrom, win_start, win_end), dtype=float)
                    )
                    gs_arr = np.nan_to_num(
                        np.array(gs_bw.values(chrom, win_start, win_end), dtype=float)
                    )
                    motif_arr = np.nan_to_num(
                        np.array(motif_bw.values(chrom, win_start, win_end), dtype=float)
                    )
                    atac_arr = np.nan_to_num(
                        np.array(atac_bw.values(chrom, win_start, win_end), dtype=float)
                    )

                    class_data[label]["prediction"].append(pred_arr)
                    class_data[label]["gs"].append(gs_arr)
                    class_data[label]["motif"].append(motif_arr)
                    class_data[label]["atac"].append(atac_arr)
                    class_data[label]["motif_max"].append(float(np.max(motif_arr)))

                    coord_rows.append({
                        "chr": chrom,
                        "start": win_start,
                        "stop": win_end,
                        "class": label,
                        "bin_start": bin_start,
                        "bin_stop": bin_start + args.bin_size,
                    })

    output_arrays = {}
    for label in ("TP", "FP", "TN", "FN"):
        d = class_data[label]
        if d["prediction"]:
            output_arrays["%s_prediction" % label] = np.array(d["prediction"])
            output_arrays["%s_gs" % label] = np.array(d["gs"])
            output_arrays["%s_motif" % label] = np.array(d["motif"])
            output_arrays["%s_atac" % label] = np.array(d["atac"])
            output_arrays["%s_motif_max" % label] = np.array(d["motif_max"])
            logging.info("%s: %d windows saved" % (label, len(d["prediction"])))

    out_npz = os.path.join(output_dir, "%s_model_interpreting.npz" % args.prefix)
    np.savez(out_npz, **output_arrays)
    logging.info("Signal matrices saved to %s" % out_npz)

    out_coords = os.path.join(output_dir, "%s_model_interpreting_coords.tsv" % args.prefix)
    pd.DataFrame(coord_rows).to_csv(out_coords, sep="\t", index=False)
    logging.info("Coordinates saved to %s" % out_coords)

    elapsed = timeit.default_timer() - start_time
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    logging.info("Total time: %d:%d:%d." % (hours, mins, secs))
