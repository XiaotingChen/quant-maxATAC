# Data

The `data` function downloads the reference data and models that the other functions look for by default under `~/opt/maxatac/data`:

* the [quant-maxATAC_data](https://github.com/MiraldiLab/quant-maxATAC_data) repository (blacklists, chromosome sizes, quant-maxATAC TF models with their `*_cross_celltype.tsv` threshold calibration tables, and the `prepare` shell scripts), cloned into `~/opt/maxatac/data`
* the UCSC `.2bit` genome sequence for each requested genome, downloaded into `~/opt/maxatac/data/<genome>/`

`git` and `wget` must be on your PATH.

## Overwriting existing data and maxATAC v1 models

quant-maxATAC uses its own data repository instead of the maxATAC v1 [maxATAC_data](https://github.com/MiraldiLab/maxATAC_data) repository. If `~/opt/maxatac/data` already exists, `maxatac data` lists what is installed there (the source repository and how many models look like v1 binary models) and asks for confirmation before replacing it. The whole directory is replaced, **including any maxATAC v1 models**. Answer anything other than `y` to leave it untouched.

* The new data is cloned into a temporary directory first; the old directory is only removed once the clone has succeeded.
* `.2bit` files already present are kept, so they are not downloaded again.
* Without a terminal to answer the prompt (e.g. in a script or Docker build), `maxatac data` stops with an error unless `--yes` is given.

If you are still interested in using the maxATAC v1 models with quant-maxATAC, let us know on the [issue tracker](https://github.com/MiraldiLab/quant-maxATAC/issues).

## Example

```bash
maxatac data
maxatac data --genome hg38 hg19
maxatac data --yes
```

## Optional Arguments

### `--genome`

One or more genome builds to download the `.2bit` sequence for, or `all` for `hg38 hg19 mm10`. Only hg38 has been tested extensively. Default: `hg38`

### `-o`, `--output`

Base directory to install into; data is placed in `<output>/maxatac/data`. Default: `~/opt`

### `--data_repo`

Git repository to install the reference data and models from, e.g. a fork or mirror of quant-maxATAC_data. Default: `https://github.com/MiraldiLab/quant-maxATAC_data.git`

### `-y`, `--yes`

Overwrite an existing data directory, including any maxATAC v1 models, without asking for confirmation.

### `--loglevel`

Logging level (`fatal`, `error`, `warning`, `info`, `debug`). Default: `info`.
