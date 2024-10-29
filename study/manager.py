import logging
import sqlite3
from typing import Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from config.data import DataConfig
from config.model import ModelConfig
from config.study import StudyConfig
from data.utils import FeatureSplittableManager


class StudyManager(object):
    # TODO: Make this configurable by the user
    DB_KEYS = [
        'replicate',
        'trial',
        'objective'
    ]

    def __init__(self, data_config: DataConfig, model_config: ModelConfig, study_config: StudyConfig, debug: bool):
        # Track each of the configs for this analysis
        self.data_config = data_config
        self.model_config = model_config
        self.study_config = study_config

        # Track whether to run the model in debug mode or not
        self.debug = debug
        self.logger = self.init_logger()

        # Generate a unique label for the combination of configs in this analysis
        self.study_label = f"{self.study_config.label}__{self.model_config.label}__{self.data_config.label}"

        # Pull the objective function for this study -- TODO: Make this configurable
        self.objective = lambda m, x, y: log_loss(y, m.predict_proba(x))

        # Track the list of other metrics to measure and track -- TODO: make this configurable
        self.tracked_metrics = {
            "bacc": lambda m, x, y: balanced_accuracy_score(y, m.predict(x))
        }

        # Generate some null attributes to be filled later
        self.db_connection : Optional[sqlite3.Connection] = None
        self.db_cursor : Optional[sqlite3.Cursor] = None

    """ Logger Management """
    def init_logger(self):
        """Generates the logger for this study"""
        logger = logging.getLogger(self.study_config.label)

        # Define how messages for this logger are formatted
        msg_handler = logging.StreamHandler()
        msg_formatter = logging.Formatter(
            fmt=f"[{self.study_config.label}" + " {asctime} {levelname}] {message}",
            datefmt="%H:%M:%S",
            style='{'
        )
        msg_handler.setFormatter(msg_formatter)
        logger.addHandler(msg_handler)

        # Set the logger to debug mode if requested
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Return the result
        return logger

    """ DB Management """
    def init_db(self):
        # Create the requested database file if it does not already exist
        if not self.study_config.output_path.exists():
            self.study_config.output_path.touch()

        # Initiate a connection w/ a 'with' clause, to ensure the connection closes when the program does
        with sqlite3.connect(self.study_config.output_path) as con:
            # Initiate the cursor
            cur = con.cursor()

            # Generate a list of all the columns to place in the table
            col_vals = [*StudyManager.DB_KEYS, *self.tracked_metrics.keys()]

            # TODO: Allow for table resetting/overwriting via command line

            # Create the table for this study
            cur.execute(f"CREATE TABLE {self.study_label} ({', '.join(col_vals)})")

            # Return the result
            return con, cur

    def save_results(self, replicate_n, trial_n, metrics):
        # Generate the list of values to be saved to the DB
        new_vals = [replicate_n, trial_n, *metrics.values()]
        new_val_strs = [str(v) for v in new_vals]

        # Push the results to the db
        self.db_cursor.execute(f"INSERT INTO {self.study_label} VALUES ({', '.join(new_val_strs)})")
        self.db_connection.commit()

    """ ML Management """
    def calculate_metrics(self, model, test_x, test_y):
        # Calculate the objective function's value
        obj_val = self.objective(model, test_x, test_y)

        # Instantiate the set of values to be saved to the DB, starting with the metrics which are always saved
        metric_dict = {
            'objective': obj_val
        }

        # Calculate the rest
        for k, metric_func in self.tracked_metrics.items():
            metric_dict[k] = metric_func(model, test_x, test_y)

        # Return the objective function's value for re-use
        return obj_val, metric_dict

    def run(self):
        # Control for RNG before proceeding
        init_seed = self.study_config.random_seed
        np.random.seed(init_seed)

        # Get the DataManager from the config
        data_manager = self.data_config.data_manager

        # Generate the requested number of splits, so each replicate will have a unique validation group
        replicate_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=self.study_config.no_replicates)
        skf_splitter = StratifiedKFold(n_splits=self.study_config.no_replicates, random_state=init_seed, shuffle=True)

        # Process the dataframe with any operations that should be done pre-split
        data_manager = data_manager.process_pre_analysis()

        # Isolate the target column(s) from the dataset
        if isinstance(data_manager, FeatureSplittableManager):
            x = data_manager
            y = data_manager.pop_features(self.study_config.target)
        else:
            raise NotImplementedError("Unsupervised analyses are not currently supported")

        # Initiate the DB and create a table within it for the study's results
        self.db_connection, self.db_cursor = self.init_db()

        # Run the study once for each replicate
        for i, (train_idx, test_idx) in enumerate(skf_splitter.split(x.array(), y.array())):
            # Set up the workspace for this replicate
            s = replicate_seeds[i]
            np.random.seed(s)

            # If debugging, report the sizes
            self.logger.debug(f"Test/Train ratio (split {i}): {len(test_idx)}/{len(train_idx)}")

            # Split the data using the indices provided
            train_x, test_x = x.train_test_split(train_idx, test_idx)
            train_y, test_y = y[train_idx], y[test_idx]

            # Run a sub-study using this data
            self.run_supervised(i, train_x, train_y, test_x, test_y, s)

    def run_supervised(self, rep: int, train_x, train_y, test_x, test_y, seed):
        # Generate the study name for this run
        study_name = f"{self.study_label} [{rep}]"

        # Run the model specified by the model config on the data
        model_factory = self.model_config.model_factory

        # Define the function which will utilize a trial's parameters to generate models to-be-tested
        def opt_func(trial: optuna.Trial):
            # Generate and fit the model
            model = model_factory.build_model(trial)
            model.fit(train_x, train_y)

            # Calculate any metrics requested by the user, including the objective function
            obj_val, metric_vals = self.calculate_metrics(model=model, test_x=test_x, test_y=test_y)

            # Save the metric values to the DB
            self.save_results(rep, trial.number, metric_vals)

            # Return the objective function so Optuna can run optimization based on it
            return obj_val

        # Run the study with these parameters
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler
        )
        study.optimize(opt_func, n_trials=self.study_config.no_trials)
