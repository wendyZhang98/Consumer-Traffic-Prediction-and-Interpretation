import os
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt

from tarot.pred.mall_campagin.campaign_evaluate_prediction import visualize_shap
from utils.logger import logger
from utils.data_porter import read_from_csv
from tarot.pred.mall_campagin.config import CAM_PROCESSED_DATA_DIR, RESULT_DIR


def visualize_shap(predict_test_y):
    df = predict_test_y.copy()
    df['cam_total_SHAP'] = df['th_cam_day'] + df['rem_cam_days'] + df['cam_type']

    shap_date_dict = dict(zip(df['datetime'].tolist(), df['cam_total_SHAP'].tolist()))

    df_ = df.copy()
    df_ = df_.groupby('cam_name').agg({'datetime': ['min', 'max'], 'cam_total_SHAP': 'sum'}).reset_index()
    df_ = df_.drop(index=df_.loc[(df_['cam_name'] == 'no_cam')].index)
    df_.columns = ['cam_name', 'cam_start', 'cam_end', 'cam_effect']
    # df_['cam_duration'] = df_.apply(lambda x: (x['cam_end'] - x['cam_start']).days, axis=1)

    # df_['cam_effect_with_ann'] = df_['cam_effect'].astype('int')
    # df_['averge_daily_effect'] = df_.apply(lambda x: int(x['cam_effect_with_ann']) if x['cam_duration'] == 0
    #                                        else int(x['cam_effect_with_ann'] / x['cam_duration']), axis=1)

    cam_name_list = df_['cam_name'].tolist()
    begin_date_dict = dict(zip(cam_name_list, df_['cam_start'].tolist()))
    end_date_dict = dict(zip(cam_name_list, df_['cam_end'].tolist()))
    # cam_duration_dict = dict(zip(cam_name_list, df_['cam_duration'].tolist()))
    # cam_effect_with_ann = dict(zip(cam_name_list, df_['cam_effect_with_ann'].tolist()))
    # cam_average_daily_effect = dict(zip(cam_name_list, df_['averge_daily_effect'].tolist()))

    ax[0].plot(df['datetime'], df['cam_total_SHAP'], color='black', label='cam_SHAP')
    # ax.plot(df['datetime'], df['cam_type'], color='red', label='cam_type_SHAP')
    # ax.plot(df['datetime'], df['th_cam_day'], color='green', label='the_cam_day_SHAP')
    # ax.plot(df['datetime'], df['rem_cam_days'], color='blue', label='rem_cam_days_SHAP')

    for cam in cam_name_list:
        # ax.axvline(x=begin_date_dict[cam], color='blue', alpha=0.3, linestyle='--')
        df_begin_date_SHAP = df.loc[((df['cam_name'] == cam) & (df['datetime'] == begin_date_dict[cam])), 'cam_total_SHAP']
        ax[0].scatter(begin_date_dict[cam], df_begin_date_SHAP, s=250, color='blue', marker='^')
        ax[1].scatter(begin_date_dict[cam], df_begin_date_SHAP, s=250, color='blue', marker='^')
        # ax[0].annotate(f'{cam}:\n{cam_effect_with_ann[cam]}\n{cam_duration_dict[cam]}Days\n{cam_average_daily_effect[cam]}per D',
        #             (begin_date_dict[cam]-timedelta(days=2), df_begin_date_SHAP-50))

        # ax.axvline(x=end_date_dict[cam], color='red', alpha=0.3, linestyle='--')
        df_end_date_SHAP = df.loc[((df['cam_name'] == cam) & (df['datetime'] == end_date_dict[cam])), 'cam_total_SHAP']
        ax[0].scatter(end_date_dict[cam], df_end_date_SHAP, s=250, color='red', marker='v')
        ax[1].scatter(end_date_dict[cam], df_end_date_SHAP, s=250, color='red', marker='v')
        # if end_date_dict[cam] != begin_date_dict[cam]:
        #     ax[0].annotate(f'{cam}:\n{cam_effect_with_ann[cam]}\n{cam_duration_dict[cam]}Days\n{cam_average_daily_effect[cam]}per D',
        #                 (end_date_dict[cam]-timedelta(days=2), df_end_date_SHAP+25))

    weekend_dates = df.loc[df['is_weekend_feature'] == 1.0, 'datetime'].tolist()
    for weekend_date in weekend_dates:
        ax[0].axvline(weekend_date, ymin=-400, ymax=400, color='green', alpha=0.4, ls=':')

    # fig.set_size_inches([70, 30])
    fig.set_size_inches([140, 30])
    ax[0].legend()
    plt.savefig(os.path.join(RESULT_DIR, "cam_feature_SHAP.png"))
    logger.info(f"Already finished the cam_feature_SHAP.png!")


cam_test_data = read_from_csv('cam_test_data.csv', CAM_PROCESSED_DATA_DIR, parse_dates=['datetime'])
mall_data = read_from_csv('mall_data.csv', CAM_PROCESSED_DATA_DIR, dtype={'holiday_name': str, 'holiday_type': str},
                          parse_dates=['datetime'])
shap_visual = read_from_csv('average_SHAP.csv', RESULT_DIR, parse_dates=['datetime'])


mall_id_lst = ['S00124585', 'S00137372', 'S00187169', 'S00012513', 'S00187048',
               'S00012549', 'S00011983', 'S00012651', 'S00187310', 'S00011323',
               'S00026110', 'S00034173', 'S00210552', 'S00146925', 'S00227303',
               'S00027476', 'S00012421', 'S00123717', 'S00011754']

mall_data_ = mall_data[(mall_data['datetime'] < '2020-01-01') & (mall_data['datetime'] > '2018-12-31')]
cam_test_data_ = cam_test_data[(cam_test_data['datetime'] < '2020-01-01') & (cam_test_data['datetime'] > '2018-12-31')]
shap_visual_ = shap_visual[(shap_visual['datetime'] < '2020-01-01') & (shap_visual['datetime'] > '2018-12-31')]

for pred_id in mall_id_lst:
    fig, ax = plt.subplots(2, 1, sharey=True)
    mall_data = mall_data_[mall_data_['mall_id']==pred_id]
    cam_test_data = cam_test_data_[cam_test_data_['mall_id']==pred_id]
    shap_visual = shap_visual_[shap_visual_['mall_id']==pred_id]

    visualize_shap(predict_test_y=shap_visual)

    cam_date_lst = mall_data.loc[mall_data['th_cam_day']==1.0, 'datetime'].tolist()
    end_data_lst = mall_data.loc[mall_data['rem_cam_days']==1.0, 'datetime'].tolist()
    wkd_lst = mall_data.loc[mall_data['is_weekend']==1.0, 'datetime'].tolist()

    # Make Visualization based on one mall
    ax[1] = sns.lineplot(x=cam_test_data['datetime'], y=cam_test_data['cam_effect'], color='red', label='cam_effect')

    for begin_date in cam_date_lst:
        ax[1].axvline(x=begin_date, alpha=0.3, color='blue')
    for end_data in end_data_lst:
        ax[1].axvline(x=end_data, alpha=0.3, color='red')
    for wkd_date in wkd_lst:
        ax[1].axvline(x=wkd_date, alpha=0.3, color='green', linestyle='--')
    plt.title(pred_id)
    plt.savefig(f'C:/Users/impor/mall_Pic/{pred_id}.png')
    # plt.show()
    plt.close()

