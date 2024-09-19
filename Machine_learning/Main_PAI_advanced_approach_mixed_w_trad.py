# -*- coding: utf-8 -*-
"""
Created in February 2024
by authors Charlotte Meinke, Kevin Hilbert & Silvan Hornstein
"""

# %% Import packages
# Standard packages
import argparse
import numpy as np
import os
import copy
import pandas as pd
import pickle
import shap
import sklearn
import sys
import time
import warnings
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pandas import read_csv
from scipy.sparse import SparseEfficiencyWarning
from scipy.stats import uniform
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Ridge, ElasticNetCV, RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
import xgboost as xgb

# Custom packages 
import sys
sys.path.append("C:\\Users\\meinkcha.PSYCHOLOGIE\\Documents\\GitHub\\PAI_Paper_Analysis_new\\Machine_learning\\")  # Append the directory containing the script
from Simple_biased_script import  add_interaction_features, apply_stepwise_regression_feat_selection, create_counterfactual_dataset

from library.Evaluating_PAI import calc_PAI_metrics_across_reps_notpertreat, summarize_PAI_metrics_across_reps
from library.Evaluating_feat_importance import summarize_features, collect_shaps, make_shap_plots
from library.Evaluating_modelperformance import calc_modelperformance_metrics, get_modelperformance_metrics_across_folds, summarize_modelperformance_metrics_across_folds
from library.html_script import PAI_to_HTML
from library.Imputing import MiceModeImputer_pipe
from library.Organizing import create_folder_to_save_results, get_categorical_variables
from library.Preprocessing import FeatureSelector
from library.Scaling import ZScalerDimVars


# %% General settings

def set_options_and_paths():
    """ Set options and paths based on command-line or inline arguments depending on the use of command line or the IDE.

    Returns:
    - args: An object containing parsed command-line arguments.
    - PATH_RESULTS: Path to save results.
    """

    def generate_and_create_results_path(args):
        model_name = f"{args.NAME_RESULTS_FOLDER}"
        if args.HP_TUNING == "True":
            model_name += "_hp_tuned_grid"
        PATH_RESULTS = os.path.join(args.PATH_RESULTS_BASE, model_name)
        create_folder_to_save_results(PATH_RESULTS)
        PATH_RESULTS_PLOTS = os.path.join(PATH_RESULTS, "plots")
        create_folder_to_save_results(PATH_RESULTS_PLOTS)
        PATHS = {
            "RESULT": PATH_RESULTS,
            "RESULT_PLOTS": PATH_RESULTS_PLOTS
        }

        return PATHS

    # Argparser
    parser = argparse.ArgumentParser(
        description='Advanced script to calculate the PAI')
    parser.add_argument('--PATH_INPUT_DATA', type=str,
                        help='Path to input data')
    parser.add_argument('--NAME_RESULTS_FOLDER', type=str,
                        help='Name result folder')
    parser.add_argument('--PATH_RESULTS_BASE', type=str,
                        help='Path to save results')
    parser.add_argument('--NUMBER_FOLDS', type=int, default=5,
                        help='Number of folds in the cross-validation')
    parser.add_argument('--NUMBER_REPETITIONS', type=int, default=100,
                        help='Number of repetitions of the cross-validation')
    parser.add_argument('--CLASSIFIER', type=str,
                        help='Classifier to use, set ridge_regression, random_forest or xgboost')
    parser.add_argument('--HP_TUNING', type=str, default="False",
                        help='Should hyperparameter tuning be applied? Set False or True')

    args = parser.parse_args()

    try:
        PATHS = generate_and_create_results_path(args)
        print("Using arguments given via terminal or GUI")
    except:
        print("Using arguments given in the script")
        working_directory = os.getcwd()
        args = parser.parse_args([
            '--PATH_INPUT_DATA', "Y:\\PsyThera\\Projekte_Meinke\\PAI_Personalized_Advantage_Index\\4_Analysis_CM\\Datenaufbereitung_Protect",
            '--NAME_RESULTS_FOLDER', "Protect_trad_adv_3",
            '--PATH_RESULTS_BASE', working_directory,
            '--NUMBER_FOLDS', '5',
            '--NUMBER_REPETITIONS', '10',
            '--HP_TUNING', 'False'
        ])
        PATHS = generate_and_create_results_path(args)

    return args, PATHS


def generate_treatstratified_splits(PATH_INPUT_DATA, n_folds, n_repeats):
    """Generate splits startitied for treatment groups.

    Args:
    PATH_INPUT_DATA (str): Path to the input data directory.
    n_folds (int): Number of folds in the cross-validation.
    n_repeats (int): Number of repetitions of the cross-validation.

    Returns:
    splits (list of tuples): List containing train and test indices.
    """
    groups_import_path = os.path.join(PATH_INPUT_DATA, "groups.txt")
    groups = read_csv(groups_import_path, sep="\t", header=0)
    y = np.array(groups)
    sfk = RepeatedStratifiedKFold(n_splits=n_folds,
                                  n_repeats=n_repeats,
                                  random_state=0)
    splits = list(sfk.split(np.zeros(len(y)), y))
    return splits

# %% Handle Warnings 
# FutureWarning cannot be addressed directly since it is a library-level warning. Make sure seaborn and pandas are up-to-date! 
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# SparseEfficiencyWarning
# Also seems to be on library-level
# related to how scikit-learn or other libraries handle sparse matrices internally
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)

# RuntimeWarning
# Warning = np.nanmean is called on a slice of an array that contains only NaN values
# RuntimeWarning just informs that whole slices of the array contain NaNs and skips over them with no change to the subsequent analysis
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %% Procedure for one iteration/split in the repeated stratified cross-validation

def procedure_per_iter(split, PATH_RESULTS, PATH_INPUT_DATA, args):
    """
    Perform a single iteration in the repeated k-fold cross-valdiation

    Parameters:
    - split: Tuple containing train and test indices.
    - PATH_RESULTS: Path to save results.
    - PATH_INPUT_DATA: Path to the input data folder.
    - args: Argument object containing arguments for running the script

    Returns:
    - results_single_iter: Dictionary containing results for the iteration.
    """

    random_state_seed = 0

    # Load dataset
    features_import_path = os.path.join(PATH_INPUT_DATA, "features.txt")
    labels_import_path = os.path.join(PATH_INPUT_DATA, "labels.txt")
    groups_import_path = os.path.join(PATH_INPUT_DATA, "groups.txt")
    #catvars_import_path = os.path.join(PATH_INPUT_DATA, "categorical_vars.txt")

    features_import = read_csv(features_import_path, sep="\t", header=0)
    labels_import = read_csv(labels_import_path, sep="\t", header=0)
    # Sanity check
    # features_import["outcome"] = labels_import
    name_groups_id_import = read_csv(
        groups_import_path, sep="\t", header=0)
    
    # NEW: Recode groups
    groups = np.array(name_groups_id_import)
    counts_groups = Counter(groups.flatten())
    group_labels = sorted(counts_groups, key=counts_groups.get, reverse=True)
    groups_recoded = np.where(
        groups == group_labels[0], -
        0.5, np.where(groups == group_labels[1], 0.5, groups)
    )
    groups_recoded = groups_recoded.astype(float)
    #names_categorical_vars = get_categorical_variables(catvars_import_path)
    y = np.array(labels_import)
    X_df = features_import

    # Perform splitting of dataframe into training and testset
    train_index = split[0]
    test_index = split[1]
    X_train_all_treat, X_test_all_treat = X_df.loc[train_index], X_df.loc[test_index]
    y_train_all_treat, y_test_all_treat = y[train_index], y[test_index]
    groups_train = groups_recoded[train_index]
    groups_test = groups_recoded[test_index]

    # Deal with missings (Remove variables with too many missings and
    # impute missings in remaining variables)
    imputer = MiceModeImputer_pipe()
    X_train_all_treat_imp = imputer.fit_transform(X_train_all_treat)
    X_test_all_treat_imp = imputer.transform(X_test_all_treat)
    feat_names_X_imp = imputer.new_feat_names

    # Exclude features using FeatureSelector across treatments
    selector = FeatureSelector()
    selector.fit(X_train_all_treat_imp)
    X_train_cleaned_all_treat = selector.transform(X_train_all_treat_imp)
    X_test_cleaned_all_treat = selector.transform(X_test_all_treat_imp)
    feature_names_clean = feat_names_X_imp[selector.is_feat_excluded == 0]
    feat_names_excluded = feat_names_X_imp[selector.is_feat_excluded == 1]
    
    # Z-Scale dimensional columns
    scaler = ZScalerDimVars()
    X_train_scaled = scaler.fit_transform(X_train_cleaned_all_treat)
    X_test_scaled = scaler.transform(X_test_cleaned_all_treat)
    
    # NEW: Add interaction features for both datasets
    X_train_scaled_int, feat_names_X_int = add_interaction_features(X=X_train_scaled,
                                                                      feat_names_X=feature_names_clean,
                                                                      groups=groups_train)
    X_test_scaled_int, feat_names_X_int = add_interaction_features(X=X_test_scaled,
                                                                      feat_names_X=feature_names_clean,
                                                                      groups=groups_test)
    

    # NEW: Select features based on stepwise regression based on training datasets
    is_feature_selected, feat_names_sel = apply_stepwise_regression_feat_selection(X_basis=X_train_scaled_int,
                                                                                   feat_names_X=feat_names_X_int,
                                                                                   y=y_train_all_treat,
                                                                                   p_thresholds=[
                                                                                       0.2, 0.1, 0.05],
                                                                                   feat_to_keep=["groups"])
    X_train_scaled_selected_factual = copy.deepcopy(
        X_train_scaled_int[:, is_feature_selected == True])
    X_test_scaled_selected_factual = copy.deepcopy(
        X_test_scaled_int[:, is_feature_selected == True])
    
    # NEW: Create counterfactual test-set
    X_test_scaled_selected_counterfactual = create_counterfactual_dataset(
        feat_names_sel, X_test_scaled_selected_factual)
    
    # NEW: Fit one linear classifier
    clf = LinearRegression(fit_intercept=True, copy_X=True,
                           n_jobs=None, positive=False)
    clf = clf.fit(X_train_scaled_selected_factual, y_train_all_treat)


    feature_weights = clf.coef_

    # Make predictions on the test-set for the factual treatment and save more information for later
    y_true_pred_df_one_fold = pd.DataFrame({
        "y_pred_factual" : np.ravel(clf.predict(X_test_scaled_selected_factual)),
        "y_pred_counterfactual" : np.ravel(clf.predict(X_test_scaled_selected_counterfactual)),
       "y_true" : np.ravel(y_test_all_treat),
        "group" : X_test_scaled_selected_factual[:, 0] # group is always the first column
        })
    
    y_true_pred_df_one_fold ["PAI"] = (y_true_pred_df_one_fold ['y_pred_factual'] -
                           y_true_pred_df_one_fold ['y_pred_counterfactual'])
   
    modelperformance_metrics = calc_modelperformance_metrics(y_true_pred_df_one_fold)

    # Save relevant information for each iteration in a dictionary
    results_single_iter = {
        "y_true_PAI": y_true_pred_df_one_fold[["y_true", "PAI", "group"]],
        "modelperformance_metrics": modelperformance_metrics,
        "sel_features_names": feat_names_sel,
        "sel_features_coef":feature_weights,
        "n_feat": clf.n_features_in_,
        "excluded_feat": feat_names_excluded,
        "all_features": feat_names_X_imp
    }

    return results_single_iter

# %% Run main script
if __name__ == '__main__':

    start_time = time.time()
    print('\nThe scikit-learn version is {}.'.format(sklearn.__version__))

    args, PATHS = set_options_and_paths()

    # Perform splitting stratified by treatment group
    splits = generate_treatstratified_splits(args.PATH_INPUT_DATA,
                                             n_folds=args.NUMBER_FOLDS,
                                             n_repeats=args.NUMBER_REPETITIONS)

    # Run procedure per iterations
    procedure_per_iter_spec = partial(procedure_per_iter,
                                      PATH_RESULTS=PATHS["RESULT"],
                                      PATH_INPUT_DATA=args.PATH_INPUT_DATA,
                                      args=args)
    outcomes = []

    # Multiprocessing (on cluster or local computer)
    pool = Pool(16)
    outcomes[:] = pool.map(procedure_per_iter_spec, splits)
    pool.close()
    pool.join()
    # outcomes[:] = map(procedure_per_iter_spec,splits)  #  no multiprocessing

    # Save outcomes
    with open(os.path.join(PATHS["RESULT"], 'outcomes.pkl'), 'wb') as file:
        pickle.dump(outcomes, file)
    # with open(os.path.join(PATHS["RESULT"], 'outcomes.pkl'), 'rb') as file:
       # outcomes = pickle.load(file)

    # Summarize results across folds or repetitions of k-fold cross-validation
    modelperformance_metrics_across_folds = get_modelperformance_metrics_across_folds(
        outcomes, key_modelperformance_metrics="modelperformance_metrics")
    modelperformance_metrics_summarized = summarize_modelperformance_metrics_across_folds(
        outcomes, key_modelperformance_metrics="modelperformance_metrics")
    PAI_metrics_across_reps = calc_PAI_metrics_across_reps_notpertreat(
        outcomes, key_PAI_df="y_true_PAI", n_folds=args.NUMBER_FOLDS,
        plot_path=PATHS["RESULT_PLOTS"])
    PAI_metrics_summarized = summarize_PAI_metrics_across_reps(
        PAI_metrics_across_reps)
    #feat_sum_treat_A = summarize_features(outcomes=outcomes,
     #                                     key_feat_names="sel_features_names_treat_A",
      #                                    key_feat_weights="sel_features_coef_treat_A")

    # Save summaries as csv
    modelperformance_metrics_across_folds.to_csv(os.path.join(
        PATHS["RESULT"], "modelperformance_across_folds.txt"), sep="\t", na_rep="NA")
    modelperformance_metrics_summarized.T.to_csv(os.path.join(
        PATHS["RESULT"], "modelperformance_summary.txt"), sep="\t")
    for subgroup in PAI_metrics_across_reps:
        PAI_metrics_across_reps[subgroup].to_csv(os.path.join(
            PATHS["RESULT"], ("PAI_across_repetitions_" + subgroup + ".txt")), sep="\t", na_rep="NA")
    for subgroup in PAI_metrics_summarized:
        PAI_metrics_summarized[subgroup].to_csv(os.path.join(
            PATHS["RESULT"], ("PAI_summary_" + subgroup + ".txt")), sep="\t", na_rep="NA")
    #feat_sum_treat_A.to_csv(os.path.join(
    #    PATHS["RESULT"], "features_sum_treat_A.txt"), sep="\t", na_rep="NA")     
    

    # HTML Summary
    try:
        PAI_to_HTML(PATHS["RESULT"], plots_directory=PATHS["RESULT_PLOTS"],
                    number_folds=args.NUMBER_FOLDS, number_repetit=args.NUMBER_REPETITIONS)
        print("HTML output successfully created and saved to HTML_output folder")
    except:
        print("Failed to create HTML output")

    elapsed_time = time.time() - start_time
    print('\nThe time for running was {}.'.format(elapsed_time))
    print('Results were saved at {}.'.format(PATHS["RESULT"]))
