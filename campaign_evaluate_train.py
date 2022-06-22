# coding=utf-8
"""
Program:
Author: Chengzw
Create time: 2020-06-08 15:48:04
"""
import pandas as pd
import numpy as np

from tarot.pred.mall_campagin.config import CAM_PROCESSED_DATA_DIR, RESULT_DIR, MODEL_DIR
from tarot.pred.train_model import run_train_test
from utils.data_porter import read_from_csv
DROP_LST = ['owner', 'scale_type', 'wind_day', 'wind_night', 'mall_name', 'cam_name',
            'business_district_id', 'year',  'anniversary', 'since_open',
            'online_shopping', 'issp', 'cam_category', 'seasonality', 'ispr', 'last_cam_name_period', 'next_cam_name_period']
CAT_LST = ['city_id', 'weather_type_day', 'weather_type_night',
           'holiday_name', 'week', 'is_weekend', 'is_work', 'holiday_type',
           'month', 'week_of_month', 'is_work_tmw', 'region', 'is_line',
           'province_id', 'sub_region', 'city_grade', 'city_group_id', 'cam_type']
PRED_DAY_LST = [15]
MODEL_TS_FEATURE_PARAM_DICT = {
    15: {
        'series_feature_param': [182, 364],
        'series_feature_ratio_param': [(28, 182), (28, 364),
                                       (56, 182), (56, 364),
                                       (84, 182), (84, 364)]
    }
}


def deal_nan_feature(df):
    df['last_cam_name_period'] = df['last_cam_name_period'].fillna(df['last_cam_name_period'].max())
    # 满月庆之前的日子都认为距离上一次营销活动很久
    df['holiday_name'] = df['holiday_name'].fillna(0)
    df['holiday_type'] = df['holiday_type'].fillna(0)
    return df


def over_sampling(df):
    cam_df = df[df['is_cam'] == '1']
    no_cam_df = df[df['is_cam'] == '0']
    len_cam_df = len(cam_df)
    len_no_cam_df = len(no_cam_df)
    lst = np.random.choice(range(len_cam_df), size=len_no_cam_df, replace=True)
    cam_df = cam_df.iloc[lst]
    df = pd.concat([no_cam_df, cam_df], axis=0)
    return df


def days_ctrl(df, dur_ctrl):
    dur_ctrl_df = df[df['cam_days'] < dur_ctrl]
    return dur_ctrl_df


if __name__ == '__main__':

    mall_data = read_from_csv('mall_data.csv', CAM_PROCESSED_DATA_DIR, dtype={'holiday_name': str, 'holiday_type': str},
                              parse_dates=['datetime'])
    # 对空值进行一些处理
    mall_data = deal_nan_feature(mall_data)
    train_data = mall_data[mall_data['datetime'] < '2019-10-01']

    # train_data = mall_data[mall_data['is_weekend'] == 0.0]

    #对数据进行过采样，使得有无活动数据达到平衡
    # train_data = over_sampling(train_data)

    # @WxZ 对训练集做特征分析
    # save_to_csv(train_data, 'cam_eval_train_df.csv', RESULT_DIR)

    # @WxZ: 仅保留活动时域在10天内的活动
    # train_data = days_ctrl(train_data, 10)

    for iter_no in range(5):
        STATE = iter_no
        model_args_dict = {
            'num_leaves': 64, 'n_estimators': 100000, 'random_state': STATE, 'objective': 'rmse'}
        val_split_date, test_split_date = '2019-08-01', None

        run_train_test(
            data=train_data, pred_day_lst=PRED_DAY_LST, model_args=model_args_dict,
            train_args=None, model_type='short_term',
            ts_feature_param_dict=MODEL_TS_FEATURE_PARAM_DICT,
            val_split=val_split_date, test_split=test_split_date, drop_lst=DROP_LST,
            cat_lst=CAT_LST, res_dir=RESULT_DIR, model_dir=MODEL_DIR, log_y=False, other_info=f'_random_seed_{STATE}')
