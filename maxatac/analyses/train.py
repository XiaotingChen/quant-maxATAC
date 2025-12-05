import logging
import sys
import timeit

from tensorflow.keras.utils import OrderedEnqueuer

from maxatac.utilities.constants import TRAIN_MONITOR, INPUT_LENGTH
from maxatac.utilities.system_tools import Mute

with Mute():
    from maxatac.utilities.callbacks import get_callbacks
    from maxatac.utilities.training_tools import DataGenerator, MaxATACModel, ROIPool, SeqDataGenerator, model_selection, model_selection_v2
    from maxatac.utilities.plot import export_binary_metrics, export_loss_mse_coeff, export_model_structure

import wandb
import numpy as np
from wandb.integration.keras import WandbMetricsLogger
import tensorflow as tf
import gc

def run_training(args):
        """
        Train a maxATAC model using ATAC-seq and ChIP-seq data

        The primary input to the training function is a meta file that contains all of the information for the locations of
        ATAC-seq signal, ChIP-seq signal, TF, and Cell type.

        Example header for meta file. The meta file must be a tsv file, but the order of the columns does not matter. As
        long as the column names are the same:

        TF | Cell_Type | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks

        ## An example meta file is included in our repo

        _________________
        Workflow Overview

        1) Set up the directories and filenames
        2) Initialize the model based on the desired architectures
        3) Read in training and validation pools
        4) Initialize the training and validation generators
        5) Fit the models with the specific parameters

        :params args: arch, seed, output, prefix, output_activation, lrate, decay, weights,
        dense, batch_size, val_batch_size, train roi, validate roi, meta_file, sequence, average, threads, epochs, batches,
        tchroms, vchroms, shuffle_cell_type, rev_comp, multiprocessing, max_que_size

        :returns: Trained models saved after each epoch
        """
        # # Start Timer
        # startTime = timeit.default_timer()

        logging.info(f"Training Parameters:\n" +
                  f"Architecture: {args.arch} \n" +
                  f"Filename prefix: {args.prefix} \n" +
                  f"Output directory: {args.output} \n" +
                  f"Meta file: {args.meta_file} \n" +
                  f"Output activation: {args.output_activation} \n" +
                  f"Number of threads: {args.threads} \n" +
                  f"Use dense layer?: {args.dense} \n" +
                  f"Training ROI file (if provided): {args.train_roi} \n" +
                  f"Validation ROI file (if provided): {args.validate_roi} \n" +
                  f"2bit sequence file: {args.sequence} \n" +
                  "Restricting to chromosomes: \n   - " + "\n   - ".join(args.chroms) + "\n" +
                  "Restricting training to chromosomes: \n   - " + "\n   - ".join(args.tchroms) + "\n" +
                  "Restricting validation to chromosomes: \n   - " + "\n   - ".join(args.vchroms) + "\n" +
                  f"Number of batches: {args.batches} \n" +
                  f"Number of examples per batch: {args.batch_size} \n" +
                  f"Proportion of examples drawn randomly: {args.rand_ratio} \n" +
                  f"Shuffle training regions amongst cell types: {args.shuffle_cell_type} \n" +
                  f"Train with the reverse complement sequence: {args.rev_comp} \n" +
                  f"Number of epochs: {args.epochs} \n" +
                  f"Use multiprocessing?: {args.multiprocessing} \n" +
                  f"Max number of workers to queue: {args.max_queue_size} \n"
                  )

        # define wandb sweep
        sweep_configuration = {
            'method': 'grid',
            'name': args.wandb_sweep_name,
            'metric': {
                'goal': 'minimize',
                'name': 'val_loss'
            },
            'parameters': {
                "pearsonr_mse_alpha": {
                    "values": args.wandb_parameter_list,
                }
            },
            'run_cap': args.wandb_count,
        }

        def agent_func():
            # initialize wandb
            if args.wandb_run_id!=None:
                wandb_run=wandb.init(
                    # project="maxATAC_hparam_BO",
                    # track hyperparameters and run metadata with wandb.config
                    #name=args.wandb_name,
                    group=args.wandb_group_name,
                    notes=args.wandb_notes,
                    config={
                        'pearsonr_mse_alpha': 0.001,
                    },
                    tags=(args.wandb_tag,),
                    resume="allow",
                    id = args.wandb_run_id,
                )
            else:
                wandb_run = wandb.init(
                    # project="maxATAC_hparam_BO",
                    # track hyperparameters and run metadata with wandb.config
                    # name=args.wandb_name,
                    group=args.wandb_group_name,
                    notes=args.wandb_notes,
                    config={
                        'pearsonr_mse_alpha': 0.001,
                    },
                    tags=(args.wandb_tag,),
                )

            # Initialize the model with the architecture of choice
            maxatac_model = MaxATACModel(arch=args.arch,
                                         seed=args.seed,
                                         output_directory=args.output,
                                         prefix=args.prefix,
                                         threads=args.threads,
                                         meta_path=args.meta_file,
                                         quant=args.quant,
                                         output_activation=args.output_activation,
                                         target_scale_factor=args.target_scale_factor,
                                         dense=args.dense,
                                         weights=args.weights,
                                         loss=args.loss,
                                         wandb_config=wandb.config
                                         )

            logging.info("Import training regions")

            # Import training regions
            train_examples = ROIPool(chroms=args.tchroms,
                                     roi_file_path=args.train_roi,
                                     meta_file=args.meta_file,
                                     prefix=args.prefix,
                                     output_directory=maxatac_model.output_directory,
                                     blacklist=args.blacklist,
                                     region_length=INPUT_LENGTH,
                                     chrom_sizes_file=args.chrom_sizes
                                     )

            # Import validation regions
            validate_examples = ROIPool(chroms=args.vchroms,
                                        roi_file_path=args.validate_roi,
                                        meta_file=args.meta_file,
                                        prefix=args.prefix,
                                        output_directory=maxatac_model.output_directory,
                                        blacklist=args.blacklist,
                                        region_length=INPUT_LENGTH,
                                        chrom_sizes_file=args.chrom_sizes
                                        )

            logging.info("Initialize training data generator")

            # Initialize the training generator
            train_gen = DataGenerator(sequence=args.sequence,
                                      meta_table=maxatac_model.meta_dataframe,
                                      roi_pool=train_examples.ROI_pool,
                                      cell_type_list=maxatac_model.cell_types,
                                      rand_ratio=args.rand_ratio,
                                      chroms=args.tchroms,
                                      quant=args.quant,
                                      batch_size=args.batch_size,
                                      shuffle_cell_type=args.shuffle_cell_type,
                                      rev_comp_train=args.rev_comp,
                                      chrom_sizes=args.chrom_sizes
                                      )

            # Create keras.utils.sequence object from training generator
            seq_train_gen = SeqDataGenerator(batches=args.batches, generator=train_gen)

            # Specify max_que_size
            if args.max_queue_size:
                queue_size = int(args.max_queue_size)
                logging.info("User specified Max Queue Size: " + str(queue_size))
            else:
                queue_size = args.threads * 2
                logging.info("Max Queue Size found: " + str(queue_size))

            # Builds a Enqueuer from a Sequence.
            # Specify multiprocessing
            if args.multiprocessing:
                logging.info("Training with multiprocessing")
                train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=True)
                train_gen_enq.start(workers=args.threads, max_queue_size=queue_size)

            else:
                logging.info("Training without multiprocessing")
                train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=False)
                train_gen_enq.start(workers=1, max_queue_size=queue_size)

            enq_train_gen = train_gen_enq.get()

            logging.info("Initialize validation data generator")

            # Initialize the validation generator
            val_gen = DataGenerator(sequence=args.sequence,
                                    meta_table=maxatac_model.meta_dataframe,
                                    roi_pool=validate_examples.ROI_pool,
                                    cell_type_list=maxatac_model.cell_types,
                                    rand_ratio=args.rand_ratio,
                                    chroms=args.vchroms,
                                    quant=args.quant,
                                    batch_size=args.batch_size,
                                    shuffle_cell_type=args.shuffle_cell_type,
                                    rev_comp_train=args.rev_comp,
                                    chrom_sizes=args.chrom_sizes
                                    )

            # Create keras.utils.sequence object from validation generator
            seq_validate_gen = SeqDataGenerator(batches=args.batches, generator=val_gen)

            # Builds a Enqueuer from a Sequence.
            # Specify multiprocessing
            if args.multiprocessing:
                logging.info("Training with multiprocessing")
                val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=True)
                val_gen_enq.start(workers=args.threads, max_queue_size=queue_size)
            else:
                logging.info("Training without multiprocessing")
                val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=False)
                val_gen_enq.start(workers=1, max_queue_size=queue_size)

            enq_val_gen = val_gen_enq.get()


            logging.info("Fit model")

            # Fit the model
            training_history = maxatac_model.nn_model.fit(enq_train_gen,
                                                        validation_data=enq_val_gen,
                                                        steps_per_epoch=args.batches,
                                                        validation_steps=args.batches,
                                                        epochs=args.epochs,
                                                        callbacks=[
                                                            #WandbMetricsLogger(log_freq=5),
                                                            WandbMetricsLogger(log_freq="epoch"),
                                                        ],
                                                        # callbacks=get_callbacks(
                                                        #     model_location=maxatac_model.results_location,
                                                        #     log_location=maxatac_model.log_location,
                                                        #     tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
                                                        #     monitor=TRAIN_MONITOR
                                                        #     ),
                                                        max_queue_size=10,
                                                        use_multiprocessing=False,
                                                        workers=1,
                                                        verbose=1
                                                        )
            wandb_run.log(
                vars(args)
            )
            log_dict = {}
            target_metric = 'val_pearson'
            epoch_argmax = np.argmax(training_history.history[target_metric])
            for key in training_history.history.keys():
                log_dict['best_' + key] = training_history.history[key][epoch_argmax]
            for key in training_history.history.keys():
                log_dict[key] = training_history.history[key]
            wandb_run.log(log_dict)
            wandb_run.finish()
            # reset
            del maxatac_model
            del enq_train_gen,enq_valid_gen
            del seq_train_gen, seq_valid_gen
            del train_gen, val_gen
            del train_examples, validate_examples
            tf.compat.v1.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()

        # start sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_proj_name,)
        wandb.agent(sweep_id=sweep_id, function=agent_func, count=args.wandb_count)
