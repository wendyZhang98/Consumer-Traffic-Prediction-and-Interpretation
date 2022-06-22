# coding=utf-8
"""
Program:
Author: Chengzw
Create time: 2020-06-05

"""
import numpy as np
import pandas as pd
from functools import reduce


from utils.logger import logger
from utils.data_porter import read_from_excel, save_to_csv, read_from_csv
from tarot.pred.mall_campagin.config import CAM_PROCESSED_DATA_DIR, RAW_DATA_DIR, MALL_TRAFFIC_PRED_DATA_DIR


non_cam_feature = ('th_cam_day',  'rem_cam_days',  'cam_type', 'seasonality', 'cam_category')
non_cam_value = 'no_cam'


def check_duplicate_col(df, subset=None, cam_name_col='cam_no'):
    """
    用来检查某一列是否存在重复值
    :param df:
    :param subset: 确定唯一性的列列表
    :param cam_name_col: 针对哪一列进行重复值检查
    :return:
    """
    subset = df.columns.tolist() if subset is None else subset
    cam_num = df[cam_name_col].nunique()
    df_len = len(df.drop_duplicates(subset=subset)[cam_name_col].tolist())
    if df_len == cam_num:
        logger.info("There is no duplicated {}!".format(cam_name_col))
    else:
        logger.info("There are duplicated cam_names, please check the {}!".format(cam_name_col))
        raise ValueError


def expand_cam_date(df, start_end_cols=('start_date', 'end_date'), cam_name_col='cam_no'):
    """
    将标有开始结束日期的 dataframe 扩展成 每一天的dataframe，同时保持其他列不变
    例如： 万圣节活动 ‘2020-01-01’（开始时间） ‘2020-01-03’（结束时间）
    转变为： 2020-01-01 万圣节活动
            2020-01-02 万圣节活动
            2020-01-03 万圣节活动
    :param df:
    :param start_end_cols: 记录开始结束日期的列名列表
    :return:
    """
    cols = df.columns.tolist()
    df = df.set_index(list(set(cols) - set(start_end_cols)))
    df.columns.name = 'start_end'
    df = df.stack().reset_index(name='datetime').sort_values('datetime')
    df = df.set_index('datetime').groupby(cam_name_col, as_index=False).apply(lambda x: x.asfreq('d').ffill())\
        .reset_index(-1).reset_index(drop=True).sort_values('datetime')
    return df.drop(['start_end'], axis=1).reset_index(drop=True)


def add_cam_category(df):
    df['cam_category'] = '0'
    df.loc[df['issp'] == '折扣', 'cam_type'] = 'sale'
    df.loc[(df['issp'] != '折扣') & (df['ispr'] == '主题'), 'cam_type'] = 'theme'
    df.loc[(df['seasonality'] == '旺季') & (df['cam_type'] == 'sale'), 'cam_category'] = '旺季促销活动'
    df.loc[(df['seasonality'] == '淡季') & (df['cam_type'] == 'sale'), 'cam_category'] = '淡季促销活动'
    df.loc[(df['seasonality'] == '旺季') & (df['cam_type'] == 'theme'), 'cam_category'] = '旺季主题活动'
    df.loc[(df['seasonality'] == '淡季') & (df['cam_type'] == 'theme'), 'cam_category'] = '淡季主题活动'
    df.loc[df['online_shopping'] == '网络购物节期间举办的活动', 'cam_type'] = 'sale'
    df.loc[df['anniversary'] == '店庆', 'cam_type'] = 'sale'
    df.loc[df['online_shopping'] == '网络购物节期间举办的活动', 'cam_category'] = '网络购物节期间举办的活动'
    df.loc[df['anniversary'] == '店庆', 'cam_category'] = '店庆'
    return df


# TODO 如何处理同一天两个活动尚未完美解决
def handle_multiple_cam_one_day(df, col='cam_days'):
    """
    解决同一天多个活动问题，有的时候上一个活动没结束，下一个活动已经发生
    这种时候当天活动认为活动时间最多的那个，一般活动时间短的，活动力度大
    :return:
    """
    # 删去活动时间长的的活动
    df = df.sort_values(col, ascending=False).drop_duplicates(subset=['datetime'], keep='last').sort_values('datetime')
    return df


def gen_related_cam_date_info(df, cam_name_col='cam_name'):
    """
    计算活动的第几天（th_cam_day）以及剩余几天（rem_cam_days）,计算活动共有几天(cam_days)
    """
    cam_days_tmp = df.set_index('datetime', drop=False).groupby(cam_name_col). \
        apply(lambda x: (x['datetime'].max() - x['datetime'].min()).days+1).reset_index(name='cam_days')
    th_cam_day_temp = df.set_index('datetime', drop=False).groupby(cam_name_col).\
        apply(lambda x: (x['datetime'] - x['datetime'].min()).dt.days + 1).reset_index(name='th_cam_day')
    rem_cam_day_temp = df.set_index('datetime', drop=False).groupby(cam_name_col). \
        apply(lambda x: (x['datetime'].max() - x['datetime']).dt.days + 1).reset_index(name='rem_cam_days')
    df = reduce(lambda x, y: pd.merge(x, y, on=['datetime'] + cam_name_col, how='left'),
                [df, th_cam_day_temp, rem_cam_day_temp])
    df = pd.merge(df, cam_days_tmp, on=cam_name_col, how='left')
    return df


def fill_date(df, fill_name='datetime', level_col='cam_name', cal_last=True, cal_next=True):
    """
    填充上一次的活动日期，计算距离上一次节日的时间长度
    :param fill_name: 日期数据所在列
    :param level_value: 节日等级筛选值 计算距离相应节日等级的时间长短
    """
    def cal_last_next_date(df):

        df = df[['datetime', level_col]]
        tmp_data_list = df.values

        pre_cam_name = 'no_cam'
        pre_datetime = np.nan
        cam_date = {}
        for datetime, cam_name in tmp_data_list:
            if cam_name != pre_cam_name and pre_cam_name != 'no_cam':
                cam_date[(datetime, cam_name)] = pre_datetime
            elif pre_cam_name == 'no_cam':
                if (pre_datetime, pre_cam_name) in cam_date:
                    cam_date[(datetime, cam_name)] = cam_date[(pre_datetime, pre_cam_name)]
                else:
                    cam_date[(datetime, cam_name)] = np.nan
            else:
                cam_date[(datetime, cam_name)] = cam_date[(pre_datetime, pre_cam_name)]
            pre_cam_name = cam_name
            pre_datetime = datetime
        cam_date_list = []
        for key, value in cam_date.items():
            cam_date_list.append([key[0], key[1], value])
        return cam_date_list

    if cal_last:
        last_data = df.sort_values(fill_name)
        last_cam_date_list = cal_last_next_date(last_data)
        last_cam_date = pd.DataFrame(last_cam_date_list, columns=['datetime', 'cam_name', 'last_' + level_col + '_ffill'])
        df = pd.merge(df, last_cam_date, on=['datetime', level_col], how='left')
        if pd.notna(df['last_' + level_col + '_ffill']).sum() != 0:
            df['last_' + level_col + '_period'] = (df[fill_name] - df['last_' + level_col + '_ffill']).dt.days
        else:
            df['last_' + level_col + '_period'] = pd.NA
    if cal_next:
        next_data = df.sort_values(fill_name, ascending=False)
        next_cam_date_list = cal_last_next_date(next_data)
        next_cam_date = pd.DataFrame(next_cam_date_list, columns=['datetime', 'cam_name', 'next_' + level_col + '_bfill'])
        df = pd.merge(df, next_cam_date, on=['datetime', level_col], how='left')
        if pd.notna(df['next_' + level_col + '_bfill']).sum() != 0:
            df['next_' + level_col + '_period'] = (df['next_' + level_col + '_bfill'] - df[fill_name]).dt.days
        else:
            df['next_' + level_col + '_period'] = pd.NA
    return df.drop(columns=['last_' + level_col + '_ffill', 'next_' + level_col + '_bfill'])


if __name__ == '__main__':
    #24家商场营销活动信息
    mall_cam = read_from_excel('435家-采集活动结果分类-20200722.xlsx', RAW_DATA_DIR,
                               sheet_name='采集活动结果分类', headers=1,
                               usecols=['BeginTime', 'EndTime', 'ActivityName', 'issp', 'ispr', 'anniversary',
                                        'online_shopping', 'mall_id', '旺季/淡季活动分类', 'mall_name'],
                               parse_dates=['BeginTime', 'EndTime'])
    mall_cam = mall_cam.rename(columns={'BeginTime': 'start_date', 'EndTime': 'end_date', 'ActivityName': 'cam_name',
                                        '旺季/淡季活动分类': 'seasonality'})
    mall_cam = mall_cam[~((pd.isna(mall_cam['issp'])) & (pd.isna(mall_cam['ispr'])))]
    mall_cam = mall_cam[mall_cam['start_date'] < '2020-01-01']
    mall_cam = mall_cam[mall_cam['end_date'] < '2020-01-01']
    mall_id_list = mall_cam['mall_id'].unique().tolist()
    mall_cam = expand_cam_date(mall_cam,  cam_name_col=['cam_name', 'mall_id'])
    mall_cam = add_cam_category(mall_cam)
    mall_cam = gen_related_cam_date_info(mall_cam, cam_name_col=['cam_name', 'mall_id'])
    # 将mall_cam特征加入mall_traffic_pred的特征
    mall_cam = mall_cam.drop(['issp', 'ispr', 'mall_name'], axis=1)
    mall_innum = read_from_csv('mall_innum_cleaned_processed.csv', MALL_TRAFFIC_PRED_DATA_DIR, parse_dates=['datetime'])
    mall_innum = mall_innum[mall_innum['datetime'] >= '2017-11-01']
    mall_innum = mall_innum[mall_innum['datetime'] <= '2020-01-20']
    mall_innum_id_lst = mall_innum['mall_id'].unique().tolist()

    mall_id_list = list(set(mall_id_list).intersection(set(mall_innum_id_lst)))
    mall_innum = mall_innum[mall_innum['mall_id'].isin(mall_id_list)]
    mall_cam = pd.merge(mall_innum, mall_cam, on=['mall_id', 'datetime'], how='left')
    mall_cam['cam_name'] = mall_cam['cam_name'].fillna('no_cam')
    tmp_mall_cam_list = []
    for mall_id in mall_id_list:
        tmp_mall_cam = mall_cam[mall_cam['mall_id'] == mall_id]
        # 处理一天对应多种活动的情况
        tmp_mall_cam = handle_multiple_cam_one_day(tmp_mall_cam, col='cam_days')
        tmp_mall_cam = fill_date(tmp_mall_cam, fill_name='datetime', level_col='cam_name', cal_last=True, cal_next=True)
        tmp_mall_cam_list.append(tmp_mall_cam)
    mall_data = pd.concat(tmp_mall_cam_list, axis=0, ignore_index=True)
    mall_data['cam_type'] = mall_data['cam_type'].fillna('0')
    mall_data['cam_days'] = mall_data['cam_days'].fillna(0)
    mall_data['th_cam_day'] = mall_data['th_cam_day'].fillna(0)
    mall_data['rem_cam_days'] = mall_data['rem_cam_days'].fillna(0)
    save_to_csv(mall_data, 'mall_data.csv', CAM_PROCESSED_DATA_DIR)
