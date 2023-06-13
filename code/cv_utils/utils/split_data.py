"""
Split data by split_columns such that distribution_columns
distribution is roughly the same across splits.

Adapted from
https://github.com/aramis-lab/AD-DL/blob/1feec979e308adcba43b3b53f9bb4a3ada2af797/clinicadl/clinicadl/tools/tsv/data_split.py#L75
"""
import argparse
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import yaml
from scipy.stats.mstats import mquantiles
from statsmodels.tools.sm_exceptions import PerfectSeparationError

logger = logging.getLogger(__name__)

DataSplitTuple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

def split_data(data: pd.DataFrame, mandatory_columns: List[str], split_ratio: Optional[List[float]] = [0.6, 0.2, 0.2], label: Optional[Dict[str, str]] = None, id: Optional[str] = None, balance_columns: Optional[Dict[str, str]] = None, drop_columns: Optional[List[str]] = None, drop_duplicates: Optional[bool] = False, filter_columns: Optional[Dict[str, Any]] = None, missing_values: Optional[str] = None, oversampling: Optional[bool] = False, undersampling: Optional[bool] = False, shuffle: Optional[bool] = True, seed: Optional[int] = 42, save_folder: Optional[str] = None, **kwargs: Optional[Dict[str, Any]]) -> DataSplitTuple:
    """Split data into train, validation and test using supplied split ratio. Data can be cleaned, filtered and balanced before split.

    Args:
        data (pd.DataFrame): Data to be split.
        mandatory_columns (List[str]): Mandatory columns present in each data split.
        split_ratio (Optional[List[float]]): Ratio of data to be split into train, val and test.
        label (Optional[Dict[str, str]]): Needed for balanced sampling and over- / undersampling. Key: Label name, Value: Label type (one of categorical or numerical). Only a single label is supported.
        id (Optional[str]): Needed for balanced sampling. Name of the id column.
        balance_columns (Optional[Dict[str, str]]): Keys: Columns to distribute evenly across the data, Values: formula for patsy to treat columns. More information on supported formulas: https://patsy.readthedocs.io/en/latest/formulas.html, https://patsy.readthedocs.io/en/latest/categorical-coding.html, https://patsy.readthedocs.io/en/latest/spline-regression.html#natural-and-cyclic-cubic-regression-splines.
        drop_columns (Optional[List[str]]): Columns to drop using pandas.drop().
        drop_duplicates (Optional[bool]): Whether to drop duplicate rows using pandas.drop_duplicates().
        filter_columns (Optional[Dict[str, Any]]): Columns to filter, e.g. exclude some values of categorical samples. 
         Key: column to apply filter on or query to use DataFrame.query() or eval to use pd.eval()
         Values: list of values, string, substring, range and all pandas.eval() (https://pandas.pydata.org/docs/reference/api/pandas.eval.html) and DataFrame.query() (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) options.
        missing_values (Optional[Union[None, str]]): One of None, 'fill', "drop". Using pandas function fillna (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) or dropna (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html). Pass wanted parameters using kwargs.
        oversampling: (Optional[bool]): Whether to oversample the data.
        undersampling: (Optional[bool]): Whether to undersample the data.
        shuffle (Optional[bool]): Whether to shuffle the data or not.
        seed (Optional[int]): Seed value for the shuffle operation. Default: 42 -> random shuffle.
        save_folder (Optional[str]): Where to save the data to. None to don't save the data.
        kwargs (Optional[Dict[str, Any]]): Dict to pass additional arguments for missing_values function.
    
    Raises: 
        ValueError: If mandatory_columns are not present in data.
        ValueError: If filter_columns are specified but not in mandatory_columns.
        ValueError: If mandatory_columns and drop_columns intersect.
        ValueError: If split_ratio doesn't sum up to 1.
        ValueError: If split_ratio does not contain 3 values.
        ValueError: If no label is specified when data should be balanced.
        ValueError: If no label is specified when data should be over- or undersampled.
        ValueError: If more than one label is specified.
        ValueError: If unallowed label type is specified.
        ValueError: If id column is not present in data.
        ValueError: If mandatory columns contain missing values after cleaning step.
        ValueError: If over- and undersampling is set to true.

    Returns:
        data (DataSplitTuple): Tuple containing train, validation and test data.
    """

    missing_cols = set(mandatory_columns) - set(data.keys())
    if missing_cols != set():
            raise ValueError(f"Missing mandatory columns {missing_cols} in data.")

    if filter_columns:
        missing_cols = set(filter_columns.keys()) - set(mandatory_columns)
        if missing_cols != set():
            raise ValueError(f"Filter columns {missing_cols} must be mandatory aswell.")

    if drop_columns:
        intersecting_cols = set(drop_columns) - set(mandatory_columns)
        if intersecting_cols != set(drop_columns):
            raise ValueError(f"Can't drop {intersecting_cols} because they are mandatory.")

    if sum(split_ratio) != 1.0:
        raise ValueError(f"split_ratio needs to sum up to 1, but got {sum(split_ratio)}.")

    if len(split_ratio) != 3:
        raise ValueError(f"split_ratio needs to be of length 3, but got {len(split_ratio)}.")

    if balance_columns and (not label or not id):
        raise ValueError("Label and id must be specified if data should be balanced.")

    if (oversampling or undersampling) and not label:
        raise ValueError("Label must be specified if data should be over- or undersampled.")

    if oversampling and undersampling:
        raise ValueError("Can either over- or undersample data, but not both.")

    if label and len(list(label.keys())) != 1 and len(list(label.values())) != 1:
        raise ValueError(f"Only a single label is supported, but got {list(label.keys())}.")

    diff = set(label.values()) - set(["numerical", "categorical"])
    if diff != set():
        raise ValueError(f"Label type can only be numerical or categorical, but got {diff}.")

    if not id in data.keys():
        raise ValueError(f"id column needs to be in data, but got {id}.")

    logger.info("Starting to split data.")

    if drop_columns:
        logger.debug(f"Dropping columns {drop_columns}.")
        data.drop(labels=drop_columns, axis=1, inplace=True)

    if drop_duplicates:
        logger.debug(f"Dropping duplicate rows.")
        data.drop_duplicates(inplace=True)
    
    if filter_columns:
        logger.debug(f"Filter columns {filter_columns}.")
        indices = set(data.index)
        for key, value in filter_columns.items():
            if type(value) == type([]):
                filtered_data = data.loc[data[key].isin(value)]
            elif key == "query":
                filtered_data = data.query(value)
            elif key == "eval":
                filtered_data = pd.eval(value)
            elif type(value) == type(str):
                filtered_data = data[data[key].str.contains(value)]
            elif type(value) == type(range(0)):
                filtered_data = data[data[key].between(min(value), max(value))]
                # filtered_data = data[(data[key] >= min(value)) & (data[key] < max(value))]
            else:
                filtered_data = data.loc[data[key] == value]

            indices = indices.intersection(set(filtered_data.index))

        data = data.iloc[list(indices)]

    if missing_values == "fill":
        logger.debug("Filling missing values.")
        cleaned_data = data.fillna(axis=0, **kwargs)
        if data[mandatory_columns].isnull().values.any():
            logger.warning(f"Some values of mandatory_columns were filled.")
    elif missing_values == "drop":
        logger.debug("Dropping missing values.")
        cleaned_data = data.dropna(subset=mandatory_columns)
        if len(data) != len(cleaned_data):
            logger.warning("Some rows of mandatory_columns were dropped.")
        cleaned_data.dropna(axis=1, inplace=True, **kwargs)
        dropped = set(data.keys) - set(cleaned_data.keys())
        logger.debug(f"The following columns were dropped {dropped}.")
    else:
        logger.debug("Don't handle missing values.")
        cleaned_data = data.copy(True)

    if cleaned_data[mandatory_columns].isnull().any().any():
        nan_cols = list(cleaned_data[mandatory_columns].keys()[cleaned_data[mandatory_columns].isnull().any()])
        raise ValueError(f"Following mandatory columns of cleaned data still contain Nan values: {nan_cols}. Please clean data or use a different method for missing_values.")
    elif cleaned_data.isnull().any().any():
        nan_cols = list(cleaned_data.keys()[cleaned_data.isnull().any()])
        logger.warning(f"Cleaned data still contains Nan values in following columns: {nan_cols}. Consider using a different method for missing_values, cleaning them and run script again or clean afterwards.")
        
    if shuffle:
        logger.debug(f"Shuffling data using seed {seed}.")
        cleaned_data = cleaned_data.sample(frac=1, random_state=seed, ignore_index=True)
    
    if balance_columns:
        logger.debug(f"Balancing data according to columns: {balance_columns} and split ratio: {split_ratio}.")
        if "max_iter" in kwargs.keys():
            max_iter = kwargs["max_iter"]
        else:
            max_iter = 1000

        logger.info(f"Splitting data into training and valid_test.\n")
        # distribute according to Splitter
        splitter = BalancedStratifiedSplitter(label=label, id=id, balance_columns=balance_columns, n_test=1-split_ratio[0], max_iter=max_iter, random_state=seed)
        train_data, valid_test = splitter.split(cleaned_data)

        logger.info("")
        logger.info(f"Splitting valid_test into validation and test data.\n")
        splitter.n_test = split_ratio[2] / (1-split_ratio[0])
        val_data, test_data = splitter.split(valid_test)
    else:
        logger.debug(f"Only splitting columns without balancing according to split ratio: {split_ratio}.")
        # split according to split_ratio
        index = math.ceil(split_ratio[0]*len(cleaned_data))
        train_data = cleaned_data.iloc[:index]
        index2 = math.floor(split_ratio[1]*len(cleaned_data))
        val_data = cleaned_data.iloc[index:index+index2]
        index3 = math.floor(split_ratio[2]*len(cleaned_data))
        test_data = cleaned_data.iloc[-index3:]

    if oversampling:
        label_name = list(label.keys())[0]
        train_data = oversample(train_data, label_name)

    if undersampling:
        label_name = list(label.keys())[0]
        train_data = undersample(train_data, label_name)

    if save_folder:
        logger.debug(f"Saving data to folder: {save_folder}.")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        filename = os.path.join(save_folder, "train.csv")
        train_data.to_csv(filename)
        filename = os.path.join(save_folder, "validation.csv")
        val_data.to_csv(filename)
        filename = os.path.join(save_folder, "test.csv")
        test_data.to_csv(filename)

    # check even distribution
    if balance_columns:
        logger.debug("Checking the balancing.")
        check_distribution([train_data, val_data, test_data], balance_columns=balance_columns)

    logger.info("Finished data splitting.")

    return (train_data, val_data, test_data)


def oversample(data: pd.DataFrame, label_name: str) -> pd.DataFrame:

    max_count = data[label_name].value_counts().max()
    logger.debug(f"Oversampling data for label {label_name} to classes of size {max_count}.")

    lst = [data]
    for class_index, group in data.groupby(label_name):
        lst.append(group.sample(max_count-len(group), replace=True))
    oversampled_data = pd.concat(lst)

    return oversampled_data


def undersample(data: pd.DataFrame, label_name: str) -> pd.DataFrame:

    min_count = data[label_name].value_counts().min()

    logger.debug(f"Undersampling data for label {label_name} to classes of size {min_count}.")

    lst = []
    for class_index, group in data.groupby(label_name):
        lst.append(group.sample(min_count, replace=False))
    undersampled_data = pd.concat(lst)

    return undersampled_data


def check_distribution(data: List[pd.DataFrame], balance_columns: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_summary = pd.DataFrame()
    cat_summary = pd.DataFrame()
    cat_summary2 = pd.DataFrame()
    for i, df in enumerate(data):
        df = df[list(balance_columns.keys())]
        df = df.add_suffix("_" + str(i))
        # split into categorical and non categorical
        df_numerical = df.select_dtypes(include=['int64', "float"])
        df_categorical = df.select_dtypes(include=['object', "bool"])
            
        if not df_numerical.empty:
            num_summary = pd.concat([num_summary, df_numerical.describe()], axis=1)
        if not df_categorical.empty:
            cat_summary = pd.concat([cat_summary, df_categorical.describe()], axis=1)

            cat_summary2 = cat_summary2.append(df_categorical.value_counts(sort=False), ignore_index=True)

    if not num_summary.empty:
        logger.info(f"\n{num_summary}")
    
    if not cat_summary.empty:
        logger.info(f"\n{cat_summary}")
        logger.info(f"\n{cat_summary2}")
    return (num_summary, cat_summary, cat_summary2)


class BalancedStratifiedSplitter:
    """
    Performs a single split for each categorical label independently on the subject level.
    We do **not** compare the balance_columns distributions between the two sets based on significance tests (T-test and chi-square).
    Instead, we follow [1]_ and first estimate the propensity score by logistic regression. Next, we construct a empirical quantile-quantile plot from which we compute the maximum deviation from the 45-degree line as a measure of imbalance.

    Parameters
    ==========
    n_test : float
        If > 1, number of subjects to put in test set
        If < 1, proportion of subjects to put in test set

    max_iter : int
        Number of iterations to search for best split.

    label : Dict
        Name of label column as key and type of label as value.

    id : str
        Name of the id column.

    balance_columns : Dict
        Name of columns to balance.

    random_state : int
        Random number seed.

    References
    ----------
    .. [1] Ho, D. E., Imai, K., King, G., & Stuart, E. A. (2007).
           Matching as Nonparametric Preprocessing for Reducing Model
           Dependence in Parametric Causal Inference.
           Political Analysis, 15(3), 199–236.
           https://doi.org/10.1093/pan/mpl013
    """
    def __init__(
        self,
        label: Dict,
        id: str,
        balance_columns: Dict,
        n_test: Union[int, float] = 0.2,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_test = n_test
        self.label_column = list(label.keys())[0]
        self.label_type = label[self.label_column]
        self.id = id
        self.balance_columns = balance_columns
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)


    def _complementary_list(self, total_list: np.ndarray, sub_list: np.ndarray) -> np.ndarray:
        mask = np.ones(total_list.shape[0], dtype=bool)
        mask[sub_list] = False
        return np.flatnonzero(mask)


    def _drop_samples_with_missing(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        null_mask = data.loc[:, columns].isnull().any(axis=1)
        n_before = data.shape[0]
        out_data = data.drop(null_mask.loc[null_mask].index)
        if n_before > out_data.shape[0]:
            logger.info("Dropping %d samples due to missing %s",
                        n_before - out_data.shape[0], ", ".join(columns))
        return out_data


    def _create_baseline_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data at the subject-level in training and test set with equivalent age and sex distributions

        Parameters
        ==========
        df : pd.DataFrame
            DataFrame

        Returns
        =======
        train_df : pd.DataFrame
            subjects in the train set
        test_df : pd.DataFrame
            subjects in the test set
        """
        if self.n_test > 1:
            n_test = int(self.n_test)
        else:
            n_test = int(self.n_test * len(df))
        assert n_test > 0

        idx = np.arange(len(df))

        # convert every possible column to numeric
        df = df.apply(pd.to_numeric, errors='ignore')

        # self.balance_columns nutzen um Formula zu errechnen
        formula = ""
        for key, value in self.balance_columns.items():
            formula += value + " + "
        formula += "1"
        # formula = "cr(__AGE, df=4) + C(__PTGENDER)" 
        # cr -> cubic spline of freedom df
        # C -> force categorical
        
        logger.info("Fitting propensity score model: %r", formula)
        confounders = patsy.dmatrix(
            formula,
            df,
            return_type="dataframe")

        y = pd.Series(np.empty(idx.shape[0]),
                      index=confounders.index,
                      name="split",
                      copy=False)

        best_balance = np.inf
        best_indices = None
        for _ in range(self.max_iter):
            idx_test = self.random_state.choice(idx, size=n_test, replace=False)
            idx_test.sort(kind="mergesort")
            idx_train = self._complementary_list(idx, idx_test)

            y.iloc[idx_test] = 1.0
            y.iloc[idx_train] = 0.0

            dist = self._check_balance(y, confounders, idx_train, idx_test)
            if dist < best_balance:
                best_balance = dist
                best_indices = (idx_test, idx_train)

        logger.info("Best split after %d tries has balance statistic: %f",
                 self.max_iter, best_balance)

        idx_test, idx_train = best_indices
        test_df = df.iloc[idx_test]
        train_df = df.iloc[idx_train]

        return train_df, test_df


    def _check_balance(
            self,
            ys: pd.Series,
            X: pd.DataFrame,
            idx_train: np.ndarray,
            idx_test: np.ndarray,
        ) -> np.ndarray:
        # estimate propensity scores
        glm = sm.GLM(ys, X, family=sm.families.Binomial())
        try:
            res = glm.fit(full_output=False)
        except PerfectSeparationError:
            stats = np.inf
        else:
            ps = res.fittedvalues
            ps_test = ps.iloc[idx_test].values
            ps_train = ps.iloc[idx_train].values

            # compute sample quantiles
            # see https://github.com/statsmodels/statsmodels/blob/160911ace8119eefe0e66998ea56d24e590fc415/statsmodels/graphics/gofplots.py#L391
            ps_test.sort(kind="mergesort")
            n_test_p1 = idx_test.shape[0] + 1
            pos = np.arange(1., n_test_p1) / n_test_p1
            q = mquantiles(ps_train, pos)

            # compute maximum deviation between two distributions
            dist = np.absolute(ps_test - q)
            stats = np.max(dist)
        return stats


    def split(self, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs a single split for each label independently on the subject level.

        Parameters
        ==========
        merged_df : pd.DataFrame

        Returns
        =======
        train_df : pd.DataFrame
            subjects in the train set
        test_df : pd.DataFrame
            subjects in the test set
        """

        columns = merged_df.keys()

        # retrieve baseline visit
        baseline_df = merged_df.copy(True)

        train_data = []
        test_data = []

        if self.label_type == "categorical":
            # Get baseline splits for each label separately
            for values, df in baseline_df.groupby(self.label_column):
                train_df, test_df = self._create_baseline_split(df)
                train_df = train_df.loc[:, columns]
                test_df = test_df.loc[:, columns]

                train_data.append(train_df)
                test_data.append(test_df)
        else:
            train_df, test_df = self._create_baseline_split(baseline_df)

            train_data.append(train_df)
            test_data.append(test_df)

        train_data = pd.concat(train_data)
        test_data = pd.concat(test_data)

        logger.info("")
        logger.info("Split %d samples into %d for training, and %d for testing.",
                 merged_df.shape[0], train_data.shape[0], test_data.shape[0])
        self._log_stats(train_data, "Training")
        self._log_stats(test_data, "Testing")

        return train_data, test_data


    def _log_stats(self, data, name):
            y = (data.loc[:, self.label_column]
                 .value_counts(dropna=False, sort=False)
                 .sort_index())
                 
            logger.info("%s data has %d unique subjects and %d labels.",
                     name, data.loc[:, self.id].nunique(), y.shape[0]
                     )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False,
                        help="Path to data.")
    parser.add_argument("--mandatory_columns", nargs="*", type=str, required=False, help="columns required in the dataset.")
    # if first two not supplied config is required
    parser.add_argument("-c", "--config", type=str, required=False,
                    help="JSON configuration string for this operation")

    args = parser.parse_args(args=args)

    if args.config is None and (args.mandatory_columns is None or args.data is None):
        parser.error("--config is required if --mandatory_columns and --data is missing.")

    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if args.data is not None:
        data_path = args.data
    elif "data" in config.keys():
        data_path = config["data"]
    else:
        raise ValueError("data path must be either specified as cli argument or via the config.")

    if args.mandatory_columns is not None:
        mandatory_columns = args.mandatory_columns
    elif "mandatory_columns" in config.keys():
        mandatory_columns = config["mandatory_columns"]
    else:
        raise ValueError("mandatory_columns must be either specified as cli argument or via the config.")

    data = pd.read_csv(data_path, low_memory=False)

    args = config.copy()
    args.pop('data', None)
    args.pop('mandatory_columns', None)
    kwargs = args.pop("kwargs", None)
    kwargs = {**args, **kwargs}

    datasets = split_data(data, mandatory_columns, **kwargs)

    return datasets

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    np.seterr(all="warn")
    datasets = main()
