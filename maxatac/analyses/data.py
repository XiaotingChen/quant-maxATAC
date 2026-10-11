import glob
import shutil
import os
import sys
import stat
import logging
import subprocess
from maxatac.utilities.system_tools import get_dir, check_data_packages_installed
from maxatac.utilities.constants import V1_MODELS_WISHLIST_URL


def remove_tree(path):
    """Delete a directory tree, including read-only files such as git objects on Windows

    Args:
        path (str): Directory to delete
    """
    def make_writable_and_retry(func, failed_path, _):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable_and_retry)
    else:
        shutil.rmtree(path, onerror=make_writable_and_retry)


def describe_existing_data(data_dir):
    """Summarize an existing maxATAC data directory so the user knows what will be overwritten

    Args:
        data_dir (str): Path to the installed data directory. Default: ~/opt/maxatac/data

    Returns:
        str: A short human readable description
    """
    lines = [f"Existing maxATAC data found in: {data_dir}"]

    if os.path.isdir(os.path.join(data_dir, ".git")):
        source = subprocess.run(["git", "remote", "get-url", "origin"], cwd=data_dir,
                                capture_output=True, text=True).stdout.strip()

        if source:
            lines.append(f"  Installed from: {source}")

    model_files = glob.glob(os.path.join(data_dir, "models", "*", "*.h5"))

    if model_files:
        n_tfs = len({os.path.dirname(model_file) for model_file in model_files})

        # quant-maxATAC ships a <TF>_cross_celltype.tsv calibration table next to each model; v1 (binary) models do not
        n_v1 = len([model_file for model_file in model_files
                    if not glob.glob(os.path.join(os.path.dirname(model_file), "*_cross_celltype.tsv"))])

        lines.append(f"  Models: {len(model_files)} .h5 files for {n_tfs} TFs" +
                     (f", {n_v1} of which look like maxATAC v1 (binary) models" if n_v1 else ""))

    return "\n".join(lines)


def confirm_overwrite(data_dir, assume_yes):
    """Ask the user to confirm that an existing data directory, including any v1 models, will be replaced

    Args:
        data_dir (str): Path to the installed data directory
        assume_yes (bool): Skip the prompt (--yes)

    Returns:
        bool: True if the data directory may be overwritten
    """
    logging.warning(describe_existing_data(data_dir))

    if assume_yes:
        logging.info("--yes was set: overwriting without asking")
        return True

    if not sys.stdin.isatty():
        raise RuntimeError(f"{data_dir} already exists and would be overwritten, but there is no terminal to "
                           "confirm this. Re-run with --yes to overwrite it.")

    try:
        answer = input(f"\n{data_dir} will be overwritten, including any maxATAC v1 models it contains.\n"
                       f"quant-maxATAC does not use the v1 models; to ask for v1 support, see {V1_MODELS_WISHLIST_URL}\n"
                       "Continue? [y/N] ")

    except EOFError:
        # No answer could be read (e.g. stdin is a null device on Windows): treat it as "no"
        logging.error("No answer received. Re-run with --yes to overwrite without asking.")
        return False

    return answer.strip().lower() in ("y", "yes")


def run_data(args):
    """Download maxATAC reference data and the quant-maxATAC models

    Workflow overview

    1) If a data directory already exists, ask the user to confirm it will be overwritten (v1 models included)
    2) Clone the data repository into a temporary directory, so a failed clone leaves the old data in place
    3) Replace the old data directory with the new clone, keeping any downloaded .2bit files
    4) Download the UCSC .2bit genome sequence for each requested genome

    Args:
        args (Namespace): Arguments list for the reference genome, output directory, data repository, --yes and the
        log level.
    """
    check_data_packages_installed()

    ############### Parameters ###############
    # Path to the directory where the project data should be installed. Default: ~/opt
    base_dir = get_dir(args.output)

    # Path to the maxATAC directory in the install directory. Default: ~/opt/maxatac
    maxatac_dir = get_dir(os.path.join(base_dir, "maxatac"))

    # Final data directory. Default: ~/opt/maxatac/data
    maxatac_final_dir = os.path.join(maxatac_dir, "data")

    # Temporary clone directory, swapped in for the final directory once the clone succeeded
    maxatac_tmp_dir = os.path.join(maxatac_dir, ".maxatac_data.tmp")

    # The old data directory is renamed here before the new one is moved in, and deleted last
    maxatac_old_dir = os.path.join(maxatac_dir, ".maxatac_data.old")

    # Set all genomes if selected
    genomes = ["hg38", "hg19", "mm10"] if args.genome == ['all'] else args.genome

    ############### Body ###############
    if os.path.isdir(maxatac_final_dir) and os.listdir(maxatac_final_dir):
        if not confirm_overwrite(maxatac_final_dir, args.yes):
            logging.info("Aborted: nothing was changed.")
            return

    # Clear temporary directories left behind by an interrupted run
    for leftover_dir in (maxatac_tmp_dir, maxatac_old_dir):
        if os.path.exists(leftover_dir):
            remove_tree(leftover_dir)

    logging.info(f"Cloning {args.data_repo} \n" +
                 f"Final data will be placed in: {maxatac_final_dir}")

    try:
        subprocess.run(["git", "clone", "--depth", "1", args.data_repo, maxatac_tmp_dir], check=True)

    except subprocess.CalledProcessError:
        if os.path.exists(maxatac_tmp_dir):
            remove_tree(maxatac_tmp_dir)
        raise

    if os.path.isdir(maxatac_final_dir):
        # The .2bit files are not part of the repository and are large: carry them over instead of re-downloading
        for twobit in glob.glob(os.path.join(maxatac_final_dir, "*", "*.2bit")):
            new_genome_dir = os.path.join(maxatac_tmp_dir, os.path.basename(os.path.dirname(twobit)))

            if os.path.isdir(new_genome_dir) and not os.path.exists(os.path.join(new_genome_dir, os.path.basename(twobit))):
                shutil.move(twobit, new_genome_dir)

        # Swap by renaming, so the data directory is never left half deleted
        shutil.move(maxatac_final_dir, maxatac_old_dir)

    shutil.move(maxatac_tmp_dir, maxatac_final_dir)

    if os.path.exists(maxatac_old_dir):
        remove_tree(maxatac_old_dir)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=maxatac_final_dir,
                            capture_output=True, text=True).stdout.strip()

    logging.info(f"Installed {args.data_repo} at commit {commit}")

    for genome in genomes:
        genome_dir = os.path.join(maxatac_final_dir, genome)

        if not os.path.isdir(genome_dir):
            logging.warning(f"{args.data_repo} has no reference data for {genome}; skipping the {genome}.2bit download")
            continue

        if os.path.exists(os.path.join(genome_dir, f"{genome}.2bit")):
            logging.info(f"{genome}.2bit is already present in {genome_dir}")
            continue

        logging.info(f"Downloading {genome}.2bit into {genome_dir}")

        # Download under a temporary name so an interrupted download is not mistaken for a finished one
        subprocess.run(["wget", "--no-check-certificate",
                        f"https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.2bit",
                        "-O", f"{genome}.2bit.part"],
                       cwd=genome_dir, check=True)

        os.replace(os.path.join(genome_dir, f"{genome}.2bit.part"), os.path.join(genome_dir, f"{genome}.2bit"))

    logging.info("Finished!")
