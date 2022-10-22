import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from operator import itemgetter

from lib.dataframe_helper import fill_nan, vdrl_count

EMPTY_COUNT_COLS_NAMES = ['empty_count', 'empty_columns']
TARGET_COL = 'mc_cri_vdrl'
RANDOM_STATE = 28
IMBALANCED_SAMPLING_STRATEGY = 0.55
BALANCED_SAMPLING_STRATEGY = 'not minority'
COLUMNS_TO_ONE_HOT_ENCODING = [
    'CONS_ALCOHOL',
    'mc_get_fator_rh',
    'mc_get_fumo',
    'mc_get_gravidez_planejada',
    'mc_get_grupo_sanguineo',
    'mc_get_risco_gestacional',
    'mc_mul_chefe_familia',
    'mc_mul_est_civil',
    'mc_mul_nivel_inseguranca',
    'mc_mul_qtd_aborto',
    'mc_mul_qtd_filhos_vivos',
    'mc_mul_qtd_gest',
    'mc_mul_rec_inf_plan_fam',
    'mc_mul_tipo_const_casa',
    'mc_dae_escolaridade',
    'mc_dae_mrd_lgd_red_esg',
    'mc_dae_numero_res_domic',
    'mc_dae_possui_arv_frut',
    'mc_dae_possui_horta',
    'mc_dae_rfa',
    'mc_dae_sit_moradia',
    'mc_dae_trat_agua_uso'
]
ONE_HOT_ENCODING_COLUMNS_TO_DROP = [
    'CONS_ALCOHOL_2.0',
    'mc_get_fator_rh_2.0',
    'mc_get_fumo_2.0',
    'mc_get_gravidez_planejada_2.0',
    'mc_get_grupo_sanguineo_4.0',
    'mc_get_risco_gestacional_2.0',
    'mc_mul_chefe_familia_2.0',
    'mc_mul_est_civil_5.0',
    'mc_mul_nivel_inseguranca_2.0',
    'mc_mul_qtd_aborto_3.0',
    'mc_mul_qtd_filhos_vivos_4.0',
    'mc_mul_qtd_gest_4.0',
    'mc_mul_rec_inf_plan_fam_2.0',
    'mc_mul_tipo_const_casa_5.0',
    'mc_dae_escolaridade_9.0',
    'mc_dae_mrd_lgd_red_esg_2.0',
    'mc_dae_numero_res_domic_5.0',
    'mc_dae_possui_arv_frut_2.0',
    'mc_dae_possui_horta_2.0',
    'mc_dae_rfa_3.0',
    'mc_dae_sit_moradia_3.0',
    'mc_dae_trat_agua_uso_4.0'
]


def prepare_data_exp_ids(df, cols_to_keep=None):
    X, y = convert_to_numpy_array(df, cols_to_keep)

    X, y = apply_undersampling(X, y, IMBALANCED_SAMPLING_STRATEGY)

    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    feature_names = df.drop(TARGET_COL, axis=1).columns.to_list()

    return X_train, X_test, y_train, y_test, feature_names


def prepare_data_exp_bds(df, cols_to_keep=None):
    X, y = convert_to_numpy_array(df, cols_to_keep)

    X, y = apply_undersampling(X, y, BALANCED_SAMPLING_STRATEGY)

    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    feature_names = df.drop(TARGET_COL, axis=1).columns.to_list()

    return X_train, X_test, y_train, y_test, feature_names


def prepare_data_exp_iods(df, cols_to_keep=None):
    X, y = convert_to_numpy_array(df, cols_to_keep, True)

    X, y = apply_undersampling(X, y, IMBALANCED_SAMPLING_STRATEGY)

    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    feature_names = df.drop(TARGET_COL, axis=1).columns.to_list()

    return X_train, X_test, y_train, y_test, feature_names


def prepare_data_exp_bods(df, cols_to_keep=None):
    X, y = convert_to_numpy_array(df, cols_to_keep, True)

    X, y = apply_undersampling(X, y, BALANCED_SAMPLING_STRATEGY)

    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    feature_names = df.drop(TARGET_COL, axis=1).columns.to_list()

    return X_train, X_test, y_train, y_test, feature_names


def prepare_data_exp_iodds(df, cols_to_keep=None):
    X, y = convert_to_numpy_array(df, cols_to_keep, True, True)

    X, y = apply_undersampling(X, y, IMBALANCED_SAMPLING_STRATEGY)

    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    feature_names = df.drop(TARGET_COL, axis=1).columns.to_list()

    return X_train, X_test, y_train, y_test, feature_names


def prepare_data_exp_bodds(df, cols_to_keep=None):
    X, y = convert_to_numpy_array(df, cols_to_keep, True, True)

    X, y = apply_undersampling(X, y, BALANCED_SAMPLING_STRATEGY)

    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    feature_names = df.drop(TARGET_COL, axis=1).columns.to_list()

    return X_train, X_test, y_train, y_test, feature_names


def print_metrics(y_test, y_pred):
    """
    Function that prints the classification metrics.

    Parameters
    ----------
    y_test : np.array
        Test target.
    y_pred : np.array
        Predicted target.

    Returns
    -------
    dict
        Dictionary with the results of the classification.
    """

    # Calculate metrics
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    true_class_results, false_class_results, accuracy, macro_avg, weighted_avg = itemgetter(
        '0.0', '1.0', 'accuracy', 'macro avg', 'weighted avg')(clf_report)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"""Metrics:
  -> Accuracy: {accuracy}
  -> Macro avg: {macro_avg}
  -> Weighted avg: {weighted_avg}
  -> True class results: {true_class_results}
  -> False class results: {false_class_results}
  -> Confusion matrix: [{conf_matrix[0]}, {conf_matrix[1]}]""")


###############################################################################

def convert_to_numpy_array(df, cols_to_keep=None, apply_one_hot_encoding=False, drop_dummy_cols=False):
    print('Starting process to convert dataframe to numpy array...')

    df_copy = df.copy()

    df_copy = drop_empty_count_cols(df_copy)

    print(f'  -> Shape: {df_copy.shape}')

    df_copy = fill_nan(df_copy)

    if apply_one_hot_encoding:
        df_copy = run_one_hot_encoding(df_copy, drop_dummy_cols)

    if cols_to_keep is not None:
        df_copy = df_copy[cols_to_keep + [TARGET_COL]]

    X = np.array(df_copy.drop(TARGET_COL, axis=1))
    y = np.array(df_copy[TARGET_COL])

    return X, y


def drop_empty_count_cols(df):
    for col_name in EMPTY_COUNT_COLS_NAMES:
        if col_name in df.columns.to_list():
            print('  -> Removing empty count columns')
            df = df.drop(columns=[col_name], axis=1)

    return df


def run_one_hot_encoding(df_copy, drop_dummy_cols):
    print('  -> Applying one-hot encoding')

    df_copy = pd.get_dummies(df_copy, columns=COLUMNS_TO_ONE_HOT_ENCODING)
    print(f'  -> Shape after one-hot encoding: {df_copy.shape}')

    if drop_dummy_cols:
        df_copy = df_copy.drop(
            columns=ONE_HOT_ENCODING_COLUMNS_TO_DROP, axis=1)
        print(f'  -> Shape after drop dummy columns: {df_copy.shape}')

    return df_copy


def apply_undersampling(X, y, sampling_strategy):
    print('Applying undersampling...')
    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    X_undersampled, y_undersampled = undersampler.fit_resample(X, y)

    print(
        f'  -> Shape after undersampling: ({X_undersampled.shape[0]}, {X_undersampled.shape[1] + 1})')

    return X_undersampled, y_undersampled


def get_train_test_data(X, y):
    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    print(
        f'  -> Shape after splitting: train={X_train.shape} [0 = {train_counts[0]}, 1 = {train_counts[1]}] | test={X_test.shape} [0 = {test_counts[0]}, 1 = {test_counts[1]}]')

    return X_train, X_test, y_train, y_test
