# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:14:05 2021

@author: hilbertk
"""

# %% Import packages and set options

import sys
sys.path.append("C:\\Users\\meinkcha.PSYCHOLOGIE\\Documents\\GitHub\\PAI_Advanced_Approach\\library") # Set path to library

from Imputing import MiceModeImputer_pipe
from Preprocessing import FeatureSelector
from Scaling import ZScalerDimVars
from Evaluating_PAI import ev_PAI
from Evaluating_feat_importance import summarize_features
from Evaluating_modelperformance import calc_modelperformance_metrics, get_modelperformance_metrics_across_folds, summarize_modelperformance_metrics_across_folds
from Organizing import create_folder_to_save_results

import argparse
from collections import Counter
from sklearn.experimental import enable_iterative_imputer
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from pandas import read_csv
import time
import sklearn
import os
import pandas as pd
import numpy as np
import copy



# %%
start_time = time.time()

def set_paths(DATA):
    """ Set paths for input data"""
    OPTIONS_OVERALL = {}
    OPTIONS_OVERALL['name_features'] = 'features.txt'
    OPTIONS_OVERALL['name_labels'] = 'labels.txt'
    OPTIONS_OVERALL['groups_id'] = 'groups_id.txt'
    OPTIONS_OVERALL['groups'] = 'groups.txt'
    
    PATH_RESULTS_BASE = "Y:\\PsyThera\\Projekte_Meinke\\PAI_Personalized_Advantage_Index\\Results_PAI_Test"

    if DATA == "Protect":
        PATH_INPUT_BASE = "Y:\\PsyThera\\Projekte_Meinke\\PAI_Personalized_Advantage_Index\\4_Analysis_CM\\Datenaufbereitung_Protect"
    elif DATA == "PANIK":
        PATH_INPUT_BASE = "Y:\\PsyThera\\Projekte_Meinke\\PAI_Personalized_Advantage_Index\\4_Analysis_CM\\Datenaufbereitung_Panik"

    model_name = DATA + "_traditional_approach"
    PATH_RESULTS = os.path.join(PATH_RESULTS_BASE, model_name)
    PATH_INPUT_DATA = os.path.join(PATH_INPUT_BASE)

    return PATH_RESULTS, PATH_INPUT_DATA, OPTIONS_OVERALL

# %% Important functions


def add_interaction_features(X, feat_names_X, groups):
    """ Calculate interaction terms and add them as features
    X: feature set: numpy-array
    feat_names: feature names of numpy-array
    groups: vector with -0.5 and 0.5
    """
    X_only_int = (X * groups)  # Calculate interaction terms
    X_no_int = np.concatenate((groups, X), axis=1)
    X_int = np.concatenate((X_no_int, X_only_int), axis=1)
    feat_names_X = feat_names_X.tolist()
    feat_names_int = [item + ".groups" for item in feat_names_X]
    feat_names_X_int = ["groups"] + feat_names_X + feat_names_int

    return X_int, feat_names_X_int


def apply_stepwise_regression_feat_selection(X_basis, feat_names_X, y, p_thresholds, feat_to_keep):
    """
    X: features set (numpy-array)
    y: labels
    p_thresholds = list of p_thresholds 
    feat_to_keep = list of feature names to keep

    returns: names of the features that have been selected
    """

    # Apply stepwise regression
    feature_indices_Xbasis = np.arange(X_basis.shape[1])
    is_feature_selected = np.empty((X_basis.shape[1]), dtype=bool)

    def select_features_low(X, feature_indices_in_Xbasis, p_threshold):
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        features_pvalues = est2.pvalues[1:]
        for i in np.arange(len(features_pvalues)):
            if features_pvalues[i] < p_threshold:
                is_feature_selected[feature_indices_in_Xbasis[i]] = True
            if features_pvalues[i] >= p_threshold:
                is_feature_selected[feature_indices_in_Xbasis[i]] = False
        X_selected = X_basis[:, is_feature_selected == True]
        return is_feature_selected, X_selected

    for i in np.arange(len(p_thresholds)):
        p_threshold = p_thresholds[i]
        if i == 0:  # Round 1; p < 0.2
            feature_indices_in_Xbasis = feature_indices_Xbasis
            is_feature_selected, X_selected = select_features_low(X=X_basis,
                                                                  feature_indices_in_Xbasis=feature_indices_in_Xbasis,
                                                                  p_threshold=p_threshold)
        if i > 0:
            feature_indices_in_Xbasis = feature_indices_Xbasis[is_feature_selected == True]
            is_feature_selected, X_selected = select_features_low(X=X_selected,
                                                                  feature_indices_in_Xbasis=feature_indices_in_Xbasis,
                                                                  p_threshold=p_threshold)
    feat_names_X = np.array(feat_names_X)
    feat_names_sel = feat_names_X[is_feature_selected]

    # Force to keep group and interaction features
    # Get indices of features to keep
    feat_to_keep_idx = np.where(np.isin(feat_names_X, feat_to_keep))
    is_feature_selected[feat_to_keep_idx] = True

    # Keep Main feature if interaction feature was selected
    # Get selected interaction features
    feat_names_sel_int = [
        item for item in feat_names_sel if item.endswith(".groups")]
    for feat_name in feat_names_sel_int:
        feat_name_main = feat_name.split(".")[0]
        feat_idx_main = np.where(feat_names_X == feat_name_main)[0]
        is_feature_selected[feat_idx_main] = True

    # Get names of selected features
    # feat_to_remove = ["bl_ACQ_total"]
    # feat_to_remove_idx = np.where(np.isin(feat_names_X, feat_to_remove))
    # is_feature_selected[feat_to_remove_idx] = False
    feat_names_X = feat_names_X.tolist()
    feat_names_sel = [item for item, select in zip(
        feat_names_X, is_feature_selected) if select]

    return is_feature_selected, feat_names_sel


def create_counterfactual_dataset(feat_names_sel, X_imputed_scaled_selected_factual):
    """ This function creates a dataset that replaces all the groups columns with the opposite
    """
    # Get names of interaction features
    feat_names_sel_only_int = [
        item for item in feat_names_sel if item.endswith("groups")]
    feat_names_sel_only_int = np.array(feat_names_sel_only_int)
    # Get indices of interaction features
    feat_names_sel = np.array(feat_names_sel)
    is_int_feat = np.isin(feat_names_sel, feat_names_sel_only_int)
    feature_indices_sel = np.arange(X_imputed_scaled_selected_factual.shape[1])
    feature_indices_only_int = feature_indices_sel[is_int_feat]
    # Replace interaction features with the result when multplying them with -1
    X_imputed_scaled_selected_counterfactual = np.copy(
        X_imputed_scaled_selected_factual)
    interaction_columns = X_imputed_scaled_selected_counterfactual[:,
                                                                   feature_indices_only_int]
    X_imputed_scaled_selected_counterfactual[:,
                                             feature_indices_only_int] = -interaction_columns
    return X_imputed_scaled_selected_counterfactual

# %% Procedure per iteration


def preprocess_data_outside_cv(PATH_RESULTS, PATH_INPUT_DATA, OPTIONS_OVERALL):

    # Import features and labels
    features_import_path = os.path.join(
        PATH_INPUT_DATA, OPTIONS_OVERALL['name_features'])
    labels_import_path = os.path.join(
        PATH_INPUT_DATA, OPTIONS_OVERALL['name_labels'])
    groups_import_path = os.path.join(
        PATH_INPUT_DATA, OPTIONS_OVERALL['groups'])
    features_import = features_import = read_csv(
        features_import_path, sep="\t", header=0)
    labels_import = read_csv(labels_import_path, sep="\t", header=0)
    groups_import = read_csv(groups_import_path, sep="\t", header=0)

    # Convert DataFrame to a NumPy array
    groups = groups_import.to_numpy()
    # Recode groups to 0.5 and -0.5
    # Flatten the array to get a 1D array
    counts_groups = Counter(groups.flatten())
    group_labels = sorted(counts_groups, key=counts_groups.get, reverse=True)
    groups_recoded = np.where(
        groups == group_labels[0], -
        0.5, np.where(groups == group_labels[1], 0.5, groups)
    )
    groups_recoded = groups_recoded.astype(float)

    X_df = features_import
    feat_names_X = features_import.columns
    y = np.array(labels_import)

    # Deal with missings (Remove variables with too many missings and
    # impute missings in remaining variables)
    imputer = MiceModeImputer_pipe()
    X_imp = imputer.fit_transform(X_df)
    feat_names_X_imp_enc = imputer.new_feat_names

    # cols = ["hh_hhalone","hh_hhparents","hh_hhother","hh_hhpartner","Living_situation_metropolitan","Living_situation_smalltown","Living_situation_rural"]
    # feat_to_remove_idx = np.where(np.isin(feat_names_X_imp_enc, cols))
    # X_imp = np.delete(X_imp, feat_to_remove_idx, axis=1)
    # feat_names_X_imp_enc = np.delete(feat_names_X_imp_enc, feat_to_remove_idx)

    # Remove features with very high correlations and low variance
    selector = FeatureSelector()
    selector.fit(X_imp)
    X_imp_clean = selector.transform(X_imp)
    feat_names_X_cleaned = feat_names_X_imp_enc[selector.is_feat_excluded == 0]

    background_info = pd.DataFrame(selector.background_info)
    background_info.columns = ["reason", "high cor/sim with"]
    background_info["feature"] = feat_names_X_imp_enc

    def replace_with_array_entry(idx):
        return feat_names_X_imp_enc[int(idx)] if not np.isnan(idx) else np.nan
    # Use apply to replace values in the column
    background_info['high cor/sim with'] = background_info['high cor/sim with'].apply(
        replace_with_array_entry)
    #background_info["high cor/sim with"] = background_info["high cor/sim with"].map(lambda idx: feat_names_X_imp_enc[idx])
    background_info.to_csv(os.path.join(PATH_RESULTS, "background.txt"),
                           sep="\t",
                           index=False)

    # Scale features
    scaler = ZScalerDimVars()
    X_imputed_scaled = scaler.fit_transform(X_imp_clean)

    # Add interaction features and groups as column
    X_imputed_scaled_int, feat_names_X_int = add_interaction_features(X=X_imputed_scaled,
                                                                      feat_names_X=feat_names_X_cleaned,
                                                                      groups=groups_recoded)

    # Select features based on stepwise regression
    is_feature_selected, feat_names_sel = apply_stepwise_regression_feat_selection(X_basis=X_imputed_scaled_int,
                                                                                   feat_names_X=feat_names_X_int,
                                                                                   y=y,
                                                                                   p_thresholds=[
                                                                                       0.2, 0.1, 0.05],
                                                                                   feat_to_keep=["groups"])

    X_imputed_scaled_selected_factual = copy.deepcopy(
        X_imputed_scaled_int[:, is_feature_selected == True])

    # Create counterfactual dataframe
    X_imputed_scaled_selected_counterfactual = create_counterfactual_dataset(
        feat_names_sel, X_imputed_scaled_selected_factual)

    # Save factual and counterfactual dataset
    np.save('X_imputed_scaled_selected_factual.npy',
            X_imputed_scaled_selected_factual)
    np.save('X_imputed_scaled_selected_counterfactual.npy',
            X_imputed_scaled_selected_counterfactual)

    return feat_names_sel


# %%
#split = splits[0]
def loocv_iteration(split):
    # Load data
    X_imputed_scaled_selected_factual = np.load(
        'X_imputed_scaled_selected_factual.npy')
    X_imputed_scaled_selected_counterfactual = np.load(
        'X_imputed_scaled_selected_counterfactual.npy')

    # Perform splitting
    train_index = split[0]
    test_index = split[1]
    X_train_factual = X_imputed_scaled_selected_factual[train_index]
    X_test_factual = X_imputed_scaled_selected_factual[test_index]
    X_test_counterfactual = X_imputed_scaled_selected_counterfactual[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit model
    clf = LinearRegression(fit_intercept=True, copy_X=True,
                           n_jobs=None, positive=False)
    clf = clf.fit(X_train_factual, y_train)

    y_prediction = {
        "y_pred_factual": clf.predict(X_test_factual)[0][0],
        "y_true": y_test[0][0],
        "y_pred_counterfactual": clf.predict(X_test_counterfactual)[0][0],
        "group": X_test_factual[0][0]
    }
    y_prediction = pd.DataFrame([y_prediction])

    y_prediction["PAI"] = (y_prediction['y_pred_factual'] -
                           y_prediction['y_pred_counterfactual'])

    feature_importances = clf.coef_

    actual_treatment = X_test_factual[0, 0]  # -0,5 ist TAU, 0,5 ist IPI

    results_single_iter = {"y_prediction": y_prediction,
                           "feat_importances": feature_importances,
                           "actual_treatment": actual_treatment}

    return results_single_iter


# %% Functions to summarize results
key_y_prediction = "y_prediction"


def summarize_results_across_iterations(outcomes, key_y_prediction):
    # Combine y_predictions across iterations in one data frame
    y_prediction = pd.concat([inner_dict[key_y_prediction]
                             for inner_dict in outcomes], ignore_index=True)

    ev_metrics_all = calc_modelperformance_metrics(y_prediction)
    ev_PAI_results = ev_PAI(y_prediction, plot_path=PATH_RESULTS, suffix="all")
    # Evaluate PAI for those with the highest PAI
    median = np.median(abs(y_prediction["PAI"]))
    is_50_percent = abs(y_prediction["PAI"]) > median
    ev_PAI_results_50_perc = ev_PAI(
        y_prediction[is_50_percent], plot_path=PATH_RESULTS, suffix="50_perc")
    ev_PAI_results_treat_A = ev_PAI(
        y_prediction[y_prediction["group"] == -0.5], plot_path=PATH_RESULTS, suffix="treat_A")
    ev_PAI_results_treat_B = ev_PAI(
        y_prediction[y_prediction["group"] == 0.5], plot_path=PATH_RESULTS, suffix="treat_B")

    # TODO: Brauchen wir das?
    #is_treatment_A = groups[:,0] == 0.5
    #is_treatment_B = groups[:,0] == -0.5
    #ev_metrics_t_A = calc_modelperformance_metrics(y_prediction[is_treatment_A])
    #ev_metrics_t_B = calc_modelperformance_metrics(y_prediction[is_treatment_B])
    #ev_metrics_50_perc = calc_modelperformance_metrics(y_prediction[is_50_percent])

    # Combine dictionaries
    # ev_metrics = {}
    # prefixes = ["all","50_perc","option_A","option_B"]

    # for prefix, d in zip(prefixes, [ev_metrics_all, ev_metrics_50_perc,
    #                                 ev_metrics_t_A, ev_metrics_t_B]):
    #     for key, value in d.items():
    #         new_key = f'{prefix}_{key}'  # Add the prefix
    #         ev_metrics[new_key] = value

    # Save it as dataframes
    results_loocv = {"ev_metrics": pd.DataFrame([ev_metrics_all]),
                     "ev_PAI_results": pd.DataFrame([ev_PAI_results]),
                     "ev_PAI_results_50_perc": pd.DataFrame([ev_PAI_results_50_perc]),
                     "ev_PAI_results_treat_A": pd.DataFrame([ev_PAI_results_treat_A]),
                     "ev_PAI_results_treat_B": pd.DataFrame([ev_PAI_results_treat_B]),
                     }

    return results_loocv


def summarize_feat_importances(outcomes, key, sel_feat_names):
    # Concatenate numpy arrays of feature importances across iterations (numpy-array)
    feat_imp_array = np.concatenate(
        [inner_dict[key] for inner_dict in outcomes])
    feat_imp_mean_df = pd.DataFrame({"feature": sel_feat_names,
                                    "mean coefficient": np.mean(feat_imp_array, axis=0)
                                     })
    feat_imp_mean_df.sort_values(
        by="mean coefficient", key=abs, ascending=False, inplace=True)

    return feat_imp_mean_df


# %% Main

if __name__ == '__main__':

    print('The scikit-learn version is {}.'.format(sklearn.__version__))

    # TEST argpars
    PATH_RESULTS, PATH_INPUT_DATA, OPTIONS_OVERALL = set_paths(
        DATA="PANIK")

    # parser = argparse.ArgumentParser(description='Your script description')
    # parser.add_argument('--DATA', type=str, help='Data argument')
    # parser.add_argument('--INPUT_DATA_NAME', type=str, help='Input data argument')
    # args = parser.parse_args()

    #PATH_RESULTS, PATH_INPUT_DATA, OPTIONS_OVERALL = set_paths(DATA = args.DATA)

    create_folder_to_save_results(PATH_RESULTS)

    # Preprocess data outside of CV
    feat_names_sel = preprocess_data_outside_cv(
        PATH_RESULTS, PATH_INPUT_DATA, OPTIONS_OVERALL)

    # Perform Splitting
    labels_import_path = os.path.join(
        PATH_INPUT_DATA, OPTIONS_OVERALL['name_labels'])
    labels_import = read_csv(labels_import_path, sep="\t", header=0)
    y = np.array(labels_import)
    loo = LeaveOneOut()
    splits = list(loo.split(np.zeros(len(y)), y))

    # Parallelize runs of leave-one-out cross validation
    #pool = Pool(8)
    # runs_list = []
    outcomes = []
    #outcomes[:] = pool.map(just_do_it,runs_list)
    outcomes[:] = map(loocv_iteration, splits)
    # pool.close()
    # pool.join()

    # Summarize information
    feat_imp_df = summarize_feat_importances(
        outcomes, key="feat_importances", sel_feat_names=feat_names_sel)

    results_across_iter_df = summarize_results_across_iterations(
        outcomes, key_y_prediction="y_prediction")

    # Save information
    results_across_iter_df["ev_metrics"].to_csv(os.path.join(PATH_RESULTS, "model_eval_summary.txt"),
                                                sep="\t",
                                                index=False)
    results_across_iter_df["ev_PAI_results"].to_csv(os.path.join(PATH_RESULTS, "PAI_eval_summary.txt"),
                                                    sep="\t",
                                                    index=False)
    results_across_iter_df["ev_PAI_results_50_perc"].to_csv(os.path.join(PATH_RESULTS, "PAI_eval_50_perc.txt"),
                                                            sep="\t",
                                                            index=False)
    results_across_iter_df["ev_PAI_results_treat_A"].to_csv(os.path.join(PATH_RESULTS, "PAI_eval_treat_A.txt"),
                                                            sep="\t",
                                                            index=False)
    results_across_iter_df["ev_PAI_results_treat_B"].to_csv(os.path.join(PATH_RESULTS, "PAI_eval_treat_B.txt"),
                                                            sep="\t",
                                                            index=False)
    feat_imp_df.to_csv(os.path.join(PATH_RESULTS, "feat_importances.txt"),
                       sep="\t",
                       index=True)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
