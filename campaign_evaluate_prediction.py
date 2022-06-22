# coding=utf-8
"""
Program: 
Author: WxZ
Create time: 2020-7-2
"""
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from datetime import timedelta
from tarot.pred.funcs.data_prepare import generate_x_y
from tarot.pred.mall_campagin.data_process import non_cam_feature, non_cam_value
from utils.data_porter import read_from_pkl
from tarot.pred.train_model import feature_process, ts_feature_process, innum_mean_feature_process
from tarot.pred.mall_campagin.config import CAM_PROCESSED_DATA_DIR, RESULT_DIR, MODEL_DIR
from tarot.pred.funcs.metrics import mape
from utils.logger import logger
from utils.data_porter import read_from_csv, save_to_csv
from pandas.plotting import register_matplotlib_converters
from tarot.pred.mall_campagin.campaign_evaluate_train import DROP_LST, CAT_LST


PRED_DAY_LST = [15]
MODEL_TS_FEATURE_PARAM_DICT = {
    15: {
        'series_feature_param': [182, 364],
        'series_feature_ratio_param': [(28, 182), (28, 364),
                                       (56, 182), (56, 364),
                                       (84, 182), (84, 364)]
    }
}


register_matplotlib_converters()


def deal_nan_feature(df):
    df['last_cam_name_period'] = df['last_cam_name_period'].fillna(df['last_cam_name_period'].max())
    # 满月庆之前的日子都认为距离上一次营销活动很久
    df['holiday_name'] = df['holiday_name'].fillna(0)
    df['holiday_type'] = df['holiday_type'].fillna(0)
    return df


def run_pred(data, pred_day_lst, model_type, test_split, ts_feature_param_dict,
             start, end, mall_id_lst, model_dir, STATE, y_name='innum',
             features=None, drop_lst=None, cat_lst=None, log_y=False, run=False):
    """
    预测未来客流

    Parameters
    ----------
    test_split :测试集分割时间
    data : pd.DataFrame
        预测数据
    pred_day_lst : list
        预测天数列表
    model_type : str
        模型类型, 'short_term' or 'long_term'
    ts_feature_param_dict :dict
        时序特征构建参数字典
    start
    end
    mall_id_lst
    model_dir : str
        模型存储路径
    y_name
    features
    drop_lst : list
        删除特征列表
    cat_lst
    log_y

    Returns
    -------

    """
    for pred_day in pred_day_lst:  # 对应的长短期中每个模型循环
        # 读取模型

        own_model_dir = os.path.join(model_dir, f"model_{test_split}")
        if run:
            booster = read_from_pkl(f'model_random_seed_{STATE}_{pred_day}d.pkl', own_model_dir)
        else:
            booster = read_from_pkl(f'model_{pred_day}d.pkl', own_model_dir)

    # 调用对应的长短期时序特征生成方式
    data_with_ts_feature = ts_feature_process(
        data, y_name=y_name, shift=pred_day,
        ts_feature_param_dict=ts_feature_param_dict[pred_day])
    data_with_ts_feature = innum_mean_feature_process(data_with_ts_feature)
    # 生成其余特征
    all_data, _ = feature_process(
        data_with_ts_feature, y_name=y_name, features=features,
        drop_lst=drop_lst, cat_lst=cat_lst)
    cols = (all_data.columns.drop(booster.feature_name()).tolist()
            + booster.feature_name())
    all_data = all_data.reindex(columns=cols)
    all_data = all_data[all_data['mall_id'].isin(mall_id_lst)].copy()
    features = booster.feature_name()

    # 根据模型构建类别特征
    cate_cols = all_data.dtypes[
        all_data.dtypes == 'category'].index.tolist()
    for col, cates in zip(cate_cols, booster.pandas_categorical):
        all_data[col] = all_data[col].astype(
            pd.CategoricalDtype(categories=cates, ordered=False))

    pred_data = all_data[(all_data['datetime'] >= start)
                         & (all_data['datetime'] <= end)].copy()

    pred_result, predict_test_y_ = model_test(booster, pred_data, features, log_y=log_y)

    total_mape = mape(pred_result['y'], pred_result['pred_y'])
    logger.info(f"TEST MAPE: {total_mape}!")
    # pred_result = pred_result.drop('y', axis=1)

    return pred_result, predict_test_y_


def delete_cam(df, cam_lst=non_cam_feature, cam_name_col='cam_no'):
    """
    删除活动数据
    :return:
    """
    for col in cam_lst:
        if col not in CAT_LST:
            df.loc[df[cam_name_col] != non_cam_value, col] = 0
        else:
            df.loc[df[cam_name_col] != non_cam_value, col] = '0'
    df[cam_name_col] = non_cam_value
    df['is_cam'] = '0'
    return df


def visualize_shap(predict_test_y):
    df = predict_test_y.copy()
    df['cam_total_SHAP'] = df['th_cam_day'] + df['rem_cam_days'] + df['cam_type']
    # Zoom in the picture
    # df = df[df['datetime'] < '2019-01-01']
    # df = df[(df['datetime'] < '2020-01-01') & (df['datetime'] > '2019-01-01')]


    shap_date_dict = dict(zip(df['datetime'].tolist(), df['cam_total_SHAP'].tolist()))

    df_ = df.copy()
    df_ = df_.groupby('cam_name').agg({'datetime': ['min', 'max'], 'cam_total_SHAP': 'sum'}).reset_index()
    df_ = df_.drop(index=df_.loc[(df_['cam_name'] == 'no_cam')].index)
    df_.columns = ['cam_name', 'cam_start', 'cam_end', 'cam_effect']
    df_['cam_duration'] = df_.apply(lambda x: (x['cam_end'] - x['cam_start']).days, axis=1)
    # df_['cam_announcement_SHAP_total'] = df_['cam_start'].apply(lambda x:
    #                                                             shap_date_dict[x - timedelta(days=3)] +
    #                                                             shap_date_dict[x - timedelta(days=2)] +
    #                                                             shap_date_dict[x - timedelta(days=1)]
    #                                                             )
    df_['cam_effect_with_ann'] = df_['cam_effect'].astype('int')
    df_['averge_daily_effect'] = df_.apply(lambda x: int(x['cam_effect_with_ann']) if x['cam_duration'] == 0
                                           else int(x['cam_effect_with_ann'] / x['cam_duration']), axis=1)

    cam_name_list = df_['cam_name'].tolist()
    begin_date_dict = dict(zip(cam_name_list, df_['cam_start'].tolist()))
    end_date_dict = dict(zip(cam_name_list, df_['cam_end'].tolist()))
    cam_duration_dict = dict(zip(cam_name_list, df_['cam_duration'].tolist()))
    cam_effect_with_ann = dict(zip(cam_name_list, df_['cam_effect_with_ann'].tolist()))
    cam_average_daily_effect = dict(zip(cam_name_list, df_['averge_daily_effect'].tolist()))

    fig, ax = plt.subplots()
    ax.plot(df['datetime'], df['cam_total_SHAP'], color='black', label='cam_SHAP')
    # ax.plot(df['datetime'], df['cam_type'], color='red', label='cam_type_SHAP')
    # ax.plot(df['datetime'], df['th_cam_day'], color='green', label='the_cam_day_SHAP')
    # ax.plot(df['datetime'], df['rem_cam_days'], color='blue', label='rem_cam_days_SHAP')

    for cam in cam_name_list:
        # ax.axvline(x=begin_date_dict[cam], color='blue', alpha=0.3, linestyle='--')
        df_begin_date_SHAP = df.loc[((df['cam_name'] == cam) & (df['datetime'] == begin_date_dict[cam])), 'cam_total_SHAP']
        ax.scatter(begin_date_dict[cam], df_begin_date_SHAP, s=250, color='blue', marker='^')
        ax.annotate(f'{cam}:\n{cam_effect_with_ann[cam]}\n{cam_duration_dict[cam]}Days\n{cam_average_daily_effect[cam]}per D',
                    (begin_date_dict[cam]-timedelta(days=2), df_begin_date_SHAP-50))

        # ax.axvline(x=end_date_dict[cam], color='red', alpha=0.3, linestyle='--')
        df_end_date_SHAP = df.loc[((df['cam_name'] == cam) & (df['datetime'] == end_date_dict[cam])), 'cam_total_SHAP']
        ax.scatter(end_date_dict[cam], df_end_date_SHAP, s=250, color='red', marker='v')
        if end_date_dict[cam] != begin_date_dict[cam]:
            ax.annotate(f'{cam}:\n{cam_effect_with_ann[cam]}\n{cam_duration_dict[cam]}Days\n{cam_average_daily_effect[cam]}per D',
                        (end_date_dict[cam]-timedelta(days=2), df_end_date_SHAP+25))

    weekend_dates = df.loc[df['is_weekend_feature'] == 1.0, 'datetime'].tolist()
    for weekend_date in weekend_dates:
        ax.axvline(weekend_date, ymin=-400, ymax=400, color='green', alpha=0.4, ls=':')

    # fig.set_size_inches([70, 30])
    fig.set_size_inches([140, 30])
    ax.legend()
    plt.savefig(os.path.join(RESULT_DIR, "cam_feature_SHAP.png"))
    logger.info(f"Already finished the cam_feature_SHAP.png!")


def analysis_cam_result(df, cam_name_col='cam_no'):
    cam_eval_list = []
    for cam_no, each_cam in df.groupby(cam_name_col):
        pred_y_daily_mean = each_cam['pred_y'].mean()
        pred_y_daily_mean_offday = each_cam[each_cam['is_work'] == 0]['pred_y'].mean()
        pred_y_daily_mean_workday = each_cam[each_cam['is_work'] == 1]['pred_y'].mean()

        pred_y_no_cam_daily_mean = each_cam['pred_y_no_cam'].mean()
        pred_y_no_cam_daily_mean_offday = each_cam[each_cam['is_work'] == 0]['pred_y_no_cam'].mean()
        pred_y_no_cam_daily_mean_workday = each_cam[each_cam['is_work'] == 1]['pred_y_no_cam'].mean()

        cam_effect = pred_y_daily_mean - pred_y_no_cam_daily_mean \
            if pred_y_daily_mean - pred_y_no_cam_daily_mean >= 0 else 0
        cam_effect_offday = pred_y_daily_mean_offday - pred_y_no_cam_daily_mean_offday \
            if pred_y_daily_mean_offday - pred_y_no_cam_daily_mean_offday >= 0 else 0
        cam_effect_workday = pred_y_daily_mean_workday - pred_y_no_cam_daily_mean_workday \
            if pred_y_daily_mean_workday - pred_y_no_cam_daily_mean_workday >= 0 else 0

        cam_eval_list.append([cam_no, each_cam['cam_category'].values[0], each_cam['hol_year'].values[0],
                              cam_effect, cam_effect_offday, cam_effect_workday,
                              pred_y_daily_mean, pred_y_daily_mean_offday, pred_y_daily_mean_workday,
                              pred_y_no_cam_daily_mean, pred_y_no_cam_daily_mean_offday,
                              pred_y_no_cam_daily_mean_workday])
    df = pd.DataFrame(cam_eval_list, columns=['cam_name', 'cam_category', 'hol_year',
                                              'cam_effect', 'cam_effect_offday', 'cam_effect_workday',
                                              'pred_y_daily_mean', 'pred_y_daily_mean_offday',
                                              'pred_y_daily_mean_workday', 'pred_y_no_cam_daily_mean',
                                              'pred_y_no_cam_daily_mean_offday',
                                              'pred_y_no_cam_daily_mean_workday'])
    return df


def model_test(booster, test_data, feature_lst, log_y):
    """
    模型测试并生成测试结果

    Parameters
    ----------
    booster : lightgbm.basic.Booster
        模型对象
    test_data : pd.DataFrame
        测试数据
    feature_lst : list
        特征名称列表
    log_y

    Returns
    -------
    pred_test_res : pd.DataFrame
        测试结果
    """
    mall_id = test_data['mall_id']
    test_data.drop('mall_id', axis=1)
    test_x, test_y = generate_x_y(test_data, feature_lst, log_y=log_y)

    predict_test_y = booster.predict(test_x)
    predict_test_y_ = booster.predict(test_x, pred_contrib=True)
    predict_test_y_ = pd.DataFrame(predict_test_y_, columns=(feature_lst + ['predict_test_y']))

    true_test_y = test_y
    if log_y:
        predict_test_y = np.expm1(predict_test_y)
        true_test_y = np.expm1(true_test_y)

    pred_res_all = test_data[['datetime', 'mall_id']].copy()
    pred_res_all['y'] = true_test_y
    pred_res_all['pred_y'] = predict_test_y
    pred_res_all['mall_id'] = mall_id
    return pred_res_all, predict_test_y_


def map_id(pic_df, map_df, mall_id, cam_col='cam_no'):
    begin_date_id = []
    cam_id = []

    map_df = map_df[map_df['mall_id']==mall_id]

    for cam_no, each_cam in map_df.groupby(cam_col):
        cam_id.append(cam_no)
        begin_date_id.append(each_cam['datetime'].values[0])
    cam_id = cam_id[:-1]  # drop(no_cam)
    begin_date_id = begin_date_id[:-1]

    map_dict = dict(zip(begin_date_id,cam_id))

    pic_df['cam_name'] = pic_df['datetime'].copy()
    pic_df['cam_name'] = pic_df['cam_name'].map(map_dict)
    return pic_df


def plt_result(df, pic_dir, boundary, mall_id, title='INNUM PRED', other_info='',
               is_savefig=True, is_show=True):

    plt.figure(figsize=(50, 30))

    datetime_series = df['datetime']
    no_cam_y = df['pred_y_no_cam']
    cam_y = df['pred_y']

    cam_datetime = df.loc[df['is_cam'] == '1', 'datetime'].tolist()

    cam_datetime_y = []
    cam_datetime_no_cam_y = []
    for datetime in cam_datetime:
        cam_datetime_y.append(df[df['datetime'] == datetime]['pred_y'].tolist()[0])
        cam_datetime_no_cam_y.append(df[df['datetime'] == datetime]['pred_y_no_cam'].tolist()[0])

    plt.plot(datetime_series, no_cam_y, 'b')
    plt.plot(datetime_series, cam_y, 'r')
    plt.scatter(cam_datetime, cam_datetime_no_cam_y, c='b', marker='o')
    plt.scatter(cam_datetime, cam_datetime_y, c='r', marker='o')

    df['cam_diff'] = cam_y - no_cam_y
    df.loc[df['cam_diff'] <= boundary, 'cam_name'] = 0
    eff_df = df[(df['cam_name'] != 0) & (df['cam_name'].notnull())]

    mall_data = read_from_csv('mall_data.csv', CAM_PROCESSED_DATA_DIR, dtype={'holiday_name': str, 'holiday_type': str},
                              parse_dates=['datetime'])
    category_dict = categorize_4(mall_data,mall_id_=mall_id)

    diff_pic(df=eff_df, dictionary=category_dict, cam_category='旺季主题活动', _fc_='black')
    diff_pic(df=eff_df, dictionary=category_dict, cam_category='旺季促销活动', _fc_='red')
    diff_pic(df=eff_df, dictionary=category_dict, cam_category='淡季主题活动', _fc_='yellow')
    diff_pic(df=eff_df, dictionary=category_dict, cam_category='淡季促销活动', _fc_='green')

    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.legend([],loc='best', title='cam_category')

    plt.title(title)
    if is_savefig:
        plt.savefig(os.path.join(pic_dir, "{}{}.png".format(title.lower(), other_info)))
    if is_show:
        plt.show()


def categorize_4(df, mall_id_, grouby_col='cam_category'):
    df = df[df['mall_id'] == mall_id_]

    df = df[(df['cam_category'] == '旺季主题活动')
            | (df['cam_category'] == '旺季促销活动')
            | (df['cam_category'] == '淡季主题活动')
            | (df['cam_category'] == '淡季促销活动')]

    all_cate = {}
    for cam_cate_, each_cate_ in df.groupby(grouby_col):
        each_cate_list = each_cate_[each_cate_['cam_no'].notnull()]['cam_no'].unique().tolist()
        all_cate[cam_cate_] = each_cate_list
    return all_cate


def diff_pic(df, dictionary, cam_category , _fc_):
    category = []
    for cam_id in dictionary[cam_category]:
        category.append(cam_id)
    eff_df = df[df['cam_name'].isin(category)]
    eff_cam_id = eff_df['cam_name'].tolist()
    eff_date = eff_df['datetime'].tolist()
    eff_no_cam = eff_df['pred_y_no_cam'].tolist()
    eff_zip = list(zip(eff_date, eff_no_cam))

    for cam_diff_obv in range(len(eff_date)):
        plt.annotate(eff_cam_id[cam_diff_obv], xy=eff_zip[cam_diff_obv],
                     xytext=(eff_zip[cam_diff_obv][0], eff_zip[cam_diff_obv][1] - 14000),
                     weight='bold', color='aqua',
                     arrowprops=dict(arrowstyle="->", connectionstyle='arc3', color='black'),
                     bbox=dict(boxstyle='round,pad=0.5', fc=_fc_, ec='b', lw=1, alpha=0.4))
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
        plt.legend([cam_category], loc='best', facecolor=_fc_, title='cam_category')


def days_ctrl(df, dur_ctrl):
    dur_ctrl_df = df[df['cam_days'] < dur_ctrl]
    return dur_ctrl_df


if __name__ == '__main__':

    mall_data = read_from_csv('mall_data.csv', CAM_PROCESSED_DATA_DIR, dtype={'holiday_name': str, 'holiday_type': str},
                              parse_dates=['datetime'])
    # mall_data = days_ctrl(mall_data, 15)
    # 对空值进行一些处理
    mall_data = deal_nan_feature(mall_data)
    # mall_data = mall_data[mall_data['is_weekend'] == 0.0]

    cam_data = mall_data[['mall_id', 'datetime', 'cam_name', 'is_work', 'hol_year', 'cam_category']]

    start_date = '2019-01-01'
    end_date = '2020-01-01'

    # mall_id_lst = ['S00122354', 'S00182996']
    mall_id_lst = ['S00029353', 'S00187310', 'S00012261',  'S00126754']

    mall_data_copy = mall_data[(mall_data['datetime'] >= start_date)
                               & (mall_data['datetime'] <= end_date)].copy()

    # print(mall_data_copy['mall_id'].unique())

    mall_data_copy_ = mall_data_copy[mall_data_copy['mall_id'].isin(mall_id_lst)]
    pred_cam_info = mall_data_copy_[['mall_id', 'cam_name', 'datetime', 'is_weekend']].reset_index(drop=True)
    pred_cam_info = pred_cam_info.rename(columns={'is_weekend': 'is_weekend_feature'})

    STATE = 0
    pred_result, predict_test_y_total = run_pred(data=mall_data, pred_day_lst=PRED_DAY_LST,
                                                 model_type='short_term', test_split='None',
                                                 ts_feature_param_dict=MODEL_TS_FEATURE_PARAM_DICT,
                                                 start=start_date, end=end_date, cat_lst=CAT_LST,
                                                 mall_id_lst=mall_id_lst, drop_lst=DROP_LST,
                                                 model_dir=MODEL_DIR, STATE=STATE)
    predict_test_y_total = predict_test_y_total.reset_index(drop=True)
    for iter_no in range(1, 5):
        STATE = iter_no
        pred_result, predict_test_y_ = run_pred(data=mall_data, pred_day_lst=PRED_DAY_LST,
                                                model_type='short_term', test_split='None',
                                                ts_feature_param_dict=MODEL_TS_FEATURE_PARAM_DICT,
                                                start=start_date, end=end_date, cat_lst=CAT_LST,
                                                mall_id_lst=mall_id_lst, drop_lst=DROP_LST,
                                                model_dir=MODEL_DIR, run=True, STATE=STATE)
        predict_test_y_ = predict_test_y_.reset_index(drop=True)
        predict_test_y_total = predict_test_y_total + predict_test_y_

    # average the predict_test_y
    predict_test_y_average = predict_test_y_total / 5
    predict_test_y_average.reset_index(drop=True)
    predict_test_y_average = pd.concat([pred_cam_info, predict_test_y_average], axis=1)

    save_to_csv(predict_test_y_average, 'average_SHAP.csv', RESULT_DIR)
    quit()
    logger.info('Already finished predict_test_y_average')

    corr_SHAP = predict_test_y_average.corr()
    save_to_csv(corr_SHAP, 'corr_SHAP.csv', RESULT_DIR)

    visualize_shap(predict_test_y=predict_test_y_average)
    quit()
    cam_datetime = mall_data[['mall_id', 'datetime']]
    mall_data_without_cam = delete_cam(mall_data)

    pred_result_all_no_cam, predict_test_y_no = run_pred(data=mall_data_without_cam, pred_day_lst=PRED_DAY_LST,
                                                         model_type='short_term', test_split='None',
                                                         ts_feature_param_dict=MODEL_TS_FEATURE_PARAM_DICT,
                                                         start=start_date, end=end_date, cat_lst=CAT_LST,
                                                         mall_id_lst=mall_id_lst, drop_lst=DROP_LST,
                                                         model_dir=MODEL_DIR, run=False, STATE='')

    result_all = pd.merge(pred_result, pred_result_all_no_cam.rename(
        columns={'pred_y': 'pred_y_no_cam'}), on=['datetime', 'y', 'mall_id'], how='left')

    mall_id = 'S00122682'
    result_all = result_all[result_all['mall_id'] == mall_id]
    result_all = pd.merge(result_all, cam_datetime, on=['mall_id', 'datetime'], how='left')
    # save_to_csv(result_all, 'result_visual.csv', RESULT_DIR)

    # map date_name to cam_name to better visualize
    mall_data_ = read_from_csv('mall_data.csv', CAM_PROCESSED_DATA_DIR, dtype={'holiday_name': str, 'holiday_type': str},
                               parse_dates=['datetime'])
    result_all = map_id(result_all, mall_data_, mall_id='S00012539')
    plt_result(result_all, boundary=1000, mall_id='S00012539', title=mall_id, pic_dir=RESULT_DIR,
               other_info=' ', is_savefig=True, is_show=False)

    result_all = pd.merge(result_all, cam_data, on=['mall_id', 'datetime'], how='left')
    cam_eval_df = analysis_cam_result(result_all, cam_name_col='cam_name')
    save_to_csv(cam_eval_df, 'cam_eval_df.csv', RESULT_DIR)



