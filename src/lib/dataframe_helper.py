import pandas as pd

col_empty_value = {
    'CONS_ALCOHOL': 2,
    'mc_get_fator_rh': 2,
    'mc_get_fumo': 2,
    'mc_get_gravidez_planejada': 2,
    'mc_get_grupo_sanguineo': 4,
    'mc_get_risco_gestacional': 2,
    'mc_get_vacina_anti_tetanica': 2,
    'mc_mul_chefe_familia': 2,
    'mc_mul_est_civil': 5,
    'mc_mul_nivel_inseguranca': 2,
    'mc_mul_qtd_aborto': 3,
    'mc_mul_qtd_filhos_vivos': 4,
    'mc_mul_qtd_gest': 4,
    'mc_mul_rec_inf_plan_fam': 2,
    'mc_mul_tipo_const_casa': 5,
    'mc_dae_escolaridade': 9,
    'mc_dae_mrd_lgd_red_esg': 2,
    'mc_dae_numero_res_domic': 5,
    'mc_dae_possui_arv_frut': 2,
    'mc_dae_possui_horta': 2,
    'mc_dae_rfa': 3,
    'mc_mul_renda_familiar': 3,
    'mc_dae_sit_moradia': 3,
    'mc_dae_trat_agua_uso': 4,
}


def remove_duplicated_rows(df, label):
    '''
    Remove duplicated rows.

    Parameters:
      df (DataFrame): DataFrame to remove duplated rows.
      df_label (str): Dataset label to be printed at logs.
    '''

    print(
        f'{label}: {df[df.duplicated(keep=False)].shape[0]} duplicated rows.')

    return df.drop_duplicates(keep='first')


def vdrl_count(df_to_count, label='mc_cri_vdrl'):
    '''
    Print the count of VDRL results. It's also returned the
    total of VDRL results.

    Parameters:
      df_to_count (DataFrame): DataFrame to count VDRL results.
      label (str): Label do print
    '''

    positive_count = df_to_count[df_to_count['mc_cri_vdrl'] == 0].shape[0]
    negative_count = df_to_count[df_to_count['mc_cri_vdrl'] == 1].shape[0]
    undefined_count = df_to_count[df_to_count['mc_cri_vdrl'].isna()].shape[0]

    print(f"""VDRL values ({label}):
        -> Positives: {positive_count}
        -> Negatives: {negative_count}
        -> Undefied (NULL): {undefined_count}
        -> Total: {df_to_count.shape[0]}""")


def inner_join(left_df, right_df, on_col=None, multi_on_col=None):
    '''
    Perform an INNER JOIN operation between two DataFrames.

    Parameters:
      left_df (DataFrame): DataFrame from the left side.
      right_df (DataFrame): DataFrame from the right side.
      on_col (str, default=None): Join column name.
      multi_on_col(tuple, default=None): Multiple join columns names.
    '''

    df_merged = None

    if (on_col is not None):
        df_merged = pd.merge(left_df, right_df, on=on_col, how='inner')
    elif (multi_on_col is not None):
        df_merged = pd.merge(
            left_df,
            right_df,
            left_on=multi_on_col[0],
            right_on=multi_on_col[1],
            how='inner'
        )

    return df_merged


def remove_rows_with_missing_values_negative_vdrl(df, col):
    df_to_return = df.drop(
        df[(df[col].isna()) & (df['mc_cri_vdrl'] == 1)].index)

    print(
        f"Count of removed rows '{col}': {df.shape[0] - df_to_return.shape[0]} rows")

    return df_to_return


def fill_nan(df):
    df_copy = df.copy()

    df_copy.loc[df_copy['CONS_ALCOHOL'].isna(
    ), 'CONS_ALCOHOL'] = col_empty_value['CONS_ALCOHOL']  # 2 - Outros

    df_copy.loc[df_copy['mc_get_fator_rh'].isna(
    ), 'mc_get_fator_rh'] = col_empty_value['mc_get_fator_rh']  # 2 - Outros

    df_copy.loc[df_copy['mc_get_fumo'].isna(
    ), 'mc_get_fumo'] = col_empty_value['mc_get_fumo']  # 2 - Outros

    df_copy.loc[df_copy['mc_get_gravidez_planejada'].isna(
    ), 'mc_get_gravidez_planejada'] = col_empty_value['mc_get_gravidez_planejada']  # 2 - Outros

    df_copy.loc[df_copy['mc_get_grupo_sanguineo'].isna(
    ), 'mc_get_grupo_sanguineo'] = col_empty_value['mc_get_grupo_sanguineo']  # 4 - Outros

    df_copy.loc[df_copy['mc_get_risco_gestacional'].isna(
    ), 'mc_get_risco_gestacional'] = col_empty_value['mc_get_risco_gestacional']  # 2 - Não informado

    df_copy.loc[df_copy['mc_get_vacina_anti_tetanica'].isna(
    ), 'mc_get_vacina_anti_tetanica'] = col_empty_value['mc_get_vacina_anti_tetanica']  # 2 - Não informado

    df_copy.loc[df_copy['mc_mul_chefe_familia'].isna(
    ), 'mc_mul_chefe_familia'] = col_empty_value['mc_mul_chefe_familia']  # 2 - Outros

    df_copy.loc[df_copy['mc_mul_est_civil'].isna(
    ), 'mc_mul_est_civil'] = col_empty_value['mc_mul_est_civil']  # 5 - Outros

    df_copy.loc[((df_copy['mc_mul_nivel_inseguranca'] >= 1) & (
        df_copy['mc_mul_nivel_inseguranca'] <= 6)), 'mc_mul_nivel_inseguranca'] = 1  # Insegurança
    df_copy.loc[df_copy['mc_mul_nivel_inseguranca'].isna(
    ), 'mc_mul_nivel_inseguranca'] = col_empty_value['mc_mul_nivel_inseguranca']  # 2 - Outros

    # Creating 2 new categories to mc_mul_qtd_aborto
    df_copy.loc[df_copy['mc_mul_qtd_aborto'] >= 2,
                'mc_mul_qtd_aborto'] = 2  # Mais que 1 aborto
    df_copy.loc[df_copy['mc_mul_qtd_aborto'].isna(
    ), 'mc_mul_qtd_aborto'] = col_empty_value['mc_mul_qtd_aborto']  # 3 - Outros

    # Creating 2 new categories to mc_mul_qtd_filhos_vivos
    df_copy.loc[df_copy['mc_mul_qtd_filhos_vivos'] >= 3,
                'mc_mul_qtd_filhos_vivos'] = 3  # Mais que 2 filhos vivos
    df_copy.loc[df_copy['mc_mul_qtd_filhos_vivos'].isna(
    ), 'mc_mul_qtd_filhos_vivos'] = col_empty_value['mc_mul_qtd_filhos_vivos']  # 4 - Outros

    # Creating 2 new categories to mc_mul_qtd_gest
    # Mais que 2 gestações
    df_copy.loc[df_copy['mc_mul_qtd_gest'] >= 3, 'mc_mul_qtd_gest'] = 3
    df_copy.loc[df_copy['mc_mul_qtd_gest'].isna(
    ), 'mc_mul_qtd_gest'] = col_empty_value['mc_mul_qtd_gest']  # 4 - Outros

    df_copy.loc[df_copy['mc_mul_rec_inf_plan_fam'].isna(
    ), 'mc_mul_rec_inf_plan_fam'] = col_empty_value['mc_mul_rec_inf_plan_fam']  # 2 - Outros

    df_copy.loc[df_copy['mc_mul_tipo_const_casa'].isna(
    ), 'mc_mul_tipo_const_casa'] = col_empty_value['mc_mul_tipo_const_casa']  # 5 - Outros

    df_copy.loc[df_copy['mc_dae_escolaridade'].isna(
    ), 'mc_dae_escolaridade'] = col_empty_value['mc_dae_escolaridade']  # 9 - Não informado

    df_copy.loc[df_copy['mc_dae_mrd_lgd_red_esg'].isna(
    ), 'mc_dae_mrd_lgd_red_esg'] = col_empty_value['mc_dae_mrd_lgd_red_esg']  # 2 - Não informado

    # Creating 2 new categories to mc_dae_numero_res_domic
    df_copy.loc[df_copy['mc_dae_numero_res_domic'] >= 4,
                'mc_dae_numero_res_domic'] = 4  # Mais que 3 residentes
    df_copy.loc[df_copy['mc_dae_numero_res_domic'].isna(
    ), 'mc_dae_numero_res_domic'] = col_empty_value['mc_dae_numero_res_domic']  # 5 - Outros

    df_copy.loc[df_copy['mc_dae_possui_arv_frut'].isna(
    ), 'mc_dae_possui_arv_frut'] = col_empty_value['mc_dae_possui_arv_frut']  # 2 - Não informado

    df_copy.loc[df_copy['mc_dae_possui_horta'].isna(
    ), 'mc_dae_possui_horta'] = col_empty_value['mc_dae_possui_horta']  # 2 - Não informado

    # Creating 2 new categories to mc_dae_rfa
    df_copy.loc[df_copy['mc_dae_rfa'] <= 500, 'mc_dae_rfa'] = 0
    df_copy.loc[((df_copy['mc_dae_rfa'] > 500) & (
        df_copy['mc_dae_rfa'] <= 1000)), 'mc_dae_rfa'] = 1
    df_copy.loc[df_copy['mc_dae_rfa'] > 1000, 'mc_dae_rfa'] = 2
    df_copy.loc[df_copy['mc_dae_rfa'].isna(
    ), 'mc_dae_rfa'] = col_empty_value['mc_dae_rfa']  # 3 - Outros

    df_copy.loc[df_copy['mc_dae_sit_moradia'].isna(
    ), 'mc_dae_sit_moradia'] = col_empty_value['mc_dae_sit_moradia']  # 3 - Não informado

    df_copy.loc[df_copy['mc_dae_trat_agua_uso'].isna(
    ), 'mc_dae_trat_agua_uso'] = col_empty_value['mc_dae_trat_agua_uso']  # 4 - Não informado

    df_copy.loc[df_copy['idade'].isna(), 'idade'] = round(
        df_copy['idade'].mean())

    return df_copy
