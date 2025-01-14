# Modular Analysis Framework for DCM Analyses

This is a framework for automating high-permutation, large scale ML analyses. 
While it was developed for research into prediction post-surgical outcomes for patients with DCM,
it can be easily extended to allow for the analysis of any tabular dataset.

## Set Up

1. Clone this repository to wherever you need it:
   * `git clone https://github.com/SomeoneInParticular/modular_optuna_ml.git`
2. Create a new Conda/Mamba environment with the dependencies needed:
   * `conda env create -f environment.yml`
   * `mamba env create -f environment.yml`
3. Activate said environment
   * `conda activate modular_optuna_ml`
   * `mamba activate modular_optuna_ml`
4. Done!
5. This only sets up the tool to be run; you will still need to create the configuration files for the analyses you want to run (see `testing` for an example).

## Running the Program

Four files are needed to run an analysis

* A tabular dataset, containing the metrics you want to run the analysis on
  * Should contain at least 1 independent and 1 target metric; unsupervised analyses are currently 
    not supported
* A data configuration file; this defines where a dataset is and what pre-processing methods
should be applied to its contents. An example, alongside the dataset it manages, can be found 
in `testing/iris_data/`
* A model configuration file; this defines which ML model to test, which hyper-parameters to tune,
and how to tune them. A few examples are available in `testing/model_configs/`
* A study configuration file; this defines which metrics to evaluated throughout the runtime of the
analysis, and where to save the results (currently only supports an SQLite DB output format). An
example is provided in `testing/testing_study_config.json`

Once all three have been created, and you have installed all dependencies (detailed in 
`environment.yml`) simply run the following command (replacing the values within the 
curly brackets with the corresponding file name):

`python run_ml_analysis.py -d {data_config} -m {model_config} -s {study_config}`

For example, if you downloaded the source code for this package, you can run the following to test that everything was set up correctly:

`python run_ml_analysis.py -d testing/iris_data/iris_config.json -m testing/model_configs/log_reg.json -s testing/testing_study_config.json`

**NOTE:** By default, if a new study would overwrite the results of an old one, it will crash out instead. To force study over-writing, add the `--overwrite` flag to your command.

## Method Details

The overall structure of the analysis can be broken down into the following broad steps:

1. **Configuration Loading:** All configuration files are loaded and checked for validity. 
2. **Dataset Loading:** The tabular dataset designated in the data configuration file is loaded
   * If a target column is specified, it is split off the dataset at this point to isolate it from 
   pre-processing (see below)
3. **Study Initialization:** An Optuna study is initialized, set up to run `n_trials` trials as specified 
in the study config file.
   * All steps past this point occur per-trial, sampling from the corresponding `Trial` instance to 
   determine the hyperparameters to use.
   * Configuration files denote a parameter as being "trial tunable" by placing a dictionary in the 
   place of a constant; an example of this can be seen in the `penalty` parameter for the 
   `testing/model_configs/log_reg.json` file.
   * Details on how hyper-parameters are sampled via Optuna Trials can be found 
   (here)[https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html].
4. **Universal Pre-Processing:** Any data processing hooks for which `"run_per_replicate": true` are 
run on the dataset in its entirety
   * If a data processing hook does not specify a `run_per_replicate` value, it defaults to `true`.
5. **In-Out Splits:** The dataset is split via a stratified k-fold split into in- and out-groups,
`n_replicates` times.
   * As the parameter name implies, each of these splits will make up an analytical "replicate"
   * Any post-split hooks for which `"run_per_replicate": true` will also run here, fitting to the 
   in-dataset and transforming both the in- and out-dataset if possible 
   * If a data processing hook does not specify a `run_per_replicate` value, it defaults to `true`.
   * NOTE: Despite this occurring per-trial, the RNG state being fixed prior to study start ensures that
   the in-out datasets are the same for all trials, so long as universal pre-processing did not delete
   and samples during its run-time
6. **Replicate Pre-Processing:** For each in-dataset, any data processing hooks for which 
`"run_per_cross": true` are run on the in-dataset.
   * If a data processing hook does not specify a `run_per_cross` value, it defaults to `false`.
7. **Train-Test Splits:** The validation dataset is split via a stratified k-fold split into 
`n_crosses` splits, as defined in the study configuration file.
   * As the parameter name implies, each of these splits will make up an analytical "cross"
   * Any post-split hooks for which `"run_per_cross": true` will also run here, fitting to the 
   train dataset and transforming both the train and test set if possible
   * If a data processing hook does not specify a `run_per_cross` value, it defaults to `false`.
8. **Cross-Validate Performance Reported:** Any metrics that the user requested be tracked are 
calculated. These metrics are defined in the study config like so.
   * `train`: Evaluate the metric on a model which has been trained on the training set, evaluating the 
   metric from the model itself, or from the model's output when applied to the test set.
     * As a result of this being run once per cross, each metric specified at this hook will result in
        `n_crosses` values being output (each denoted as `{metric_name} [{cross_idx}]`)
   * `validate`: Evaluate the metric on a model which has been trained on the in-dataset, evaluating the 
   metric from the model itself, or from the model's output when applied to the in-dataset.
   * `test`: Evaluate the metric on a model which has been trained on the in-dataset, evaluating the 
   metric from the model itself, or from the model's output when applied to the out-dataset.
   * `objective`: Evaluated identically to the `train` hook, but reported as an average both to you 
   and the study instance (allowing the study to guide the hyperparameter sampling in future trials)
     * Currently, only one `objective` metric can be defined due to this averaging.