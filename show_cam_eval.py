# coding=utf-8
"""
Program:
Author: Guanguan
Create time: 2020-04-23 14:28:02
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

from utils.logger import logger
from utils.data_porter import read_from_csv, save_to_csv
from tarot.pred.mall_campagin.config import CAM_PROCESSED_DATA_DIR, RESULT_DIR, PIC_DATA_DIR

from tarot.pred.mall_campagin.campaign_evaluate_train import cam_id_dict

color_dict = {'red': {'diff': (254 / 256, 217 / 256, 118 / 256),
                      'daily': (126 / 256, 0, 0 / 256),
                      'offday': (253 / 256, 141 / 256, 60 / 256),
                      'workday': (189 / 256, 0 / 256, 38 / 256)},
              'brown': {'diff': (245 / 256, 230 / 256, 202 / 256),
                        'daily': (76 / 256, 63 / 256, 55 / 256),
                        'offday': (197 / 256, 179 / 256, 153 / 256),
                        'workday': (132 / 256, 119 / 256, 103 / 256)},
              'pink': {'diff': (220 / 256, 211 / 256, 230 / 256),
                       'daily': (128 / 256, 15 / 256, 124 / 256),
                       'offday': (221 / 256, 52 / 256, 151 / 256),
                       'workday': (140 / 256, 107 / 256, 177 / 256)},
              'green': {'diff': (220 / 256, 237 / 256, 200 / 256),
                        'daily': (76 / 256, 154 / 256, 42 / 256),
                        'offday': (164 / 256, 222 / 256, 2 / 256),
                        'workday': (30 / 256, 86 / 256, 49 / 256)},
              'blue': {'diff': (198 / 256, 219 / 256, 239 / 256),
                       'daily': (16 / 256, 109 / 256, 156 / 256),
                       'offday': (8 / 256, 186 / 256, 255 / 256),
                       'workday': (90 / 256, 146 / 256, 173 / 256)}}

cam_cat_color = {'旺季促销活动': 'red', '旺季主题活动': 'brown',
                 '淡季促销活动': 'pink', '淡季主题活动': 'green',
                 '网络购物节期间举办的活动': 'blue'}


def show_cam_increased_innum(df, figsize=(32, 45), pic_dir=PIC_DATA_DIR, is_show=True, is_savefig=True):
    df = df.sort_values('cam_effect', ascending=True)
    df['cam_diff'] = df.apply(lambda x: x.pred_y_daily_mean if x.cam_effect > 0 else x.cam_effect, axis=1)
    df['cam_offday_diff'] = df.apply(lambda x: x.pred_y_daily_mean_offday if x.cam_effect_offday > 0 else x.cam_effect_offday, axis=1)
    df['cam_workday_diff'] = df.apply(lambda x: x.pred_y_daily_mean_workday if x.cam_effect_workday > 0 else x.cam_effect_workday, axis=1)

    fig = plt.figure(figsize=figsize)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500  # 分辨率
    plt.subplots_adjust(hspace=0.2, wspace=0.2, left=0.05, bottom=0.05, right=0.9, top=0.90, )

    fig = plt.subplot(321, facecolor=(1, 1, 1))
    tmp = df[df['cam_category'] == '旺季促销活动']
    a1 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['cam_diff'], color=color_dict[cam_cat_color['旺季促销活动']]['diff'])
    a2 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['pred_y_no_budget_daily_mean'], color=color_dict[cam_cat_color['旺季促销活动']]['daily'])
    a3 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['cam_offday_diff'],
                  color=color_dict[cam_cat_color['旺季促销活动']]['diff'])
    a4 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['pred_y_no_budget_daily_mean_offday'],
                  color=color_dict[cam_cat_color['旺季促销活动']]['offday'])
    a5 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['cam_workday_diff'],
                  color=color_dict[cam_cat_color['旺季促销活动']]['diff'])
    a6 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['pred_y_no_budget_daily_mean_workday'],
                  color=color_dict[cam_cat_color['旺季促销活动']]['workday'])
    plt.xlim(-8000, 82000)
    plt.xticks([-8000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
    plt.xlabel('客流（人）')
    plt.yticks(np.arange(0, len(tmp)) * 4, tmp['cam_name'], fontsize=10)
    plt.legend([a2, a4, a6, a1], ['平均日客流', '休息日平均日客流', '工作日平均日客流', '活动客流增量'], loc='lower right', frameon=False)
    plt.title("旺季促销活动", fontsize=12)

    plt.subplot(3, 2, 2)
    tmp = df[df['cam_category'] == '旺季主题活动']
    b1 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['cam_diff'],
                  color=color_dict[cam_cat_color['旺季主题活动']]['diff'])
    b2 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['pred_y_no_budget_daily_mean'],
                  color=color_dict[cam_cat_color['旺季主题活动']]['daily'])
    b3 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['cam_offday_diff'],
                  color=color_dict[cam_cat_color['旺季主题活动']]['diff'])
    b4 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['pred_y_no_budget_daily_mean_offday'],
                  color=color_dict[cam_cat_color['旺季主题活动']]['offday'])
    b5 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['cam_workday_diff'],
                  color=color_dict[cam_cat_color['旺季主题活动']]['diff'])
    b6 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['pred_y_no_budget_daily_mean_workday'],
                  color=color_dict[cam_cat_color['旺季主题活动']]['workday'])
    plt.xlim(-8000, 82000)
    plt.xticks([-8000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
    plt.yticks(np.arange(0, len(tmp)) * 4, tmp['cam_name'], fontsize=10)
    plt.legend([b2, b4, b6, b1], ['平均日客流', '休息日平均日客流', '工作日平均日客流', '活动客流增量'], loc='lower right', frameon=False)
    plt.title("旺季主题活动", fontsize=16)
    plt.xlabel('客流（人）')

    plt.subplot(3, 2, 3)
    tmp = df[df['cam_category'] == '淡季促销活动']
    c1 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['cam_diff'],
                  color=color_dict[cam_cat_color['淡季促销活动']]['diff'])
    c2 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['pred_y_no_budget_daily_mean'],
                  color=color_dict[cam_cat_color['淡季促销活动']]['daily'])
    c3 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['cam_offday_diff'],
                  color=color_dict[cam_cat_color['淡季促销活动']]['diff'])
    c4 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['pred_y_no_budget_daily_mean_offday'],
                  color=color_dict[cam_cat_color['淡季促销活动']]['offday'])
    c5 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['cam_workday_diff'],
                  color=color_dict[cam_cat_color['淡季促销活动']]['diff'])
    c6 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['pred_y_no_budget_daily_mean_workday'],
                  color=color_dict[cam_cat_color['淡季促销活动']]['workday'])
    plt.xlim(-8000, 82000)
    plt.xticks([-8000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
    plt.yticks(np.arange(0, len(tmp)) * 4, tmp['cam_name'], fontsize=10)
    plt.legend([c2, c4, c6, c1], ['平均日客流', '休息日平均日客流', '工作日平均日客流', '活动客流增量'], loc='lower right', frameon=False)
    plt.title("淡季促销活动", fontsize=16)
    plt.xlabel('客流（人）')

    plt.subplot(3, 2, 4)
    tmp = df[df['cam_category'] == '淡季主题活动']
    d1 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['cam_diff'],
                  color=color_dict[cam_cat_color['淡季主题活动']]['diff'])
    d2 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['pred_y_no_budget_daily_mean'],
                  color=color_dict[cam_cat_color['淡季主题活动']]['daily'])
    d3 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['cam_offday_diff'],
                  color=color_dict[cam_cat_color['淡季主题活动']]['diff'])
    d4 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['pred_y_no_budget_daily_mean_offday'],
                  color=color_dict[cam_cat_color['淡季主题活动']]['offday'])
    d5 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['cam_workday_diff'],
                  color=color_dict[cam_cat_color['淡季主题活动']]['diff'])
    d6 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['pred_y_no_budget_daily_mean_workday'],
                  color=color_dict[cam_cat_color['淡季主题活动']]['workday'])
    plt.xlim(-8000, 82000)
    plt.xticks([-8000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
    plt.yticks(np.arange(0, len(tmp)) * 4, tmp['cam_name'], fontsize=10)
    plt.legend([d2, d4, d6, d1], ['平均日客流', '休息日平均日客流', '工作日平均日客流', '活动客流增量'], loc='lower right', frameon=False)
    plt.title("淡季主题活动", fontsize=16)
    plt.xlabel('客流（人）')

    plt.subplot(3, 2, 5)
    tmp = df[df['cam_category'] == '网络购物节期间举办的活动']
    e1 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['cam_diff'],
                  color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['diff'])
    e2 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['pred_y_no_budget_daily_mean'],
                  color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['daily'])
    e3 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['cam_offday_diff'],
                  color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['diff'])
    e4 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['pred_y_no_budget_daily_mean_offday'],
                  color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['offday'])
    e5 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['cam_workday_diff'],
                  color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['diff'])
    e6 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['pred_y_no_budget_daily_mean_workday'],
                  color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['workday'])
    plt.xlim(-8000, 82000)
    plt.xticks([-8000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
    plt.yticks(np.arange(0, len(tmp)) * 4, tmp['cam_name'], fontsize=10)
    plt.legend([e2, e4, e6, e1], ['平均日客流', '休息日平均日客流', '工作日平均日客流', '活动客流增量'], loc='lower right', frameon=False)
    plt.title("网络购物节期间举办的活动", fontsize=16)
    plt.xlabel('客流（人）')
    plt.suptitle("营销活动带来的客流增长绝对值", fontsize=25)
    if is_savefig:
        plt.savefig(os.path.join(pic_dir, "campaign_innum_with_hol.png"))
    if is_show:
        plt.show()
    logger.info("Finish plotting pic {}".format('营销活动带来的客流增长绝对值'))


def show_cam_increased_innum_single_plot(df, figsize=(12, 12), pic_dir=PIC_DATA_DIR, is_show=True, is_savefig=True):
    """
    单独绘制每张客流增量图
    与 show_cam_increased_innum 内容一致 只不过画图的表现方式不一致 本函数为每个类型单独画图
    """
    df = df.sort_values('cam_effect', ascending=True)
    df['cam_diff'] = df.apply(lambda x: x.pred_y_daily_mean if x.cam_effect > 0 else x.cam_effect, axis=1)
    df['cam_offday_diff'] = df.apply(lambda x: x.pred_y_daily_mean_offday if x.cam_effect_offday > 0 else x.cam_effect_offday, axis=1)
    df['cam_workday_diff'] = df.apply(lambda x: x.pred_y_daily_mean_workday if x.cam_effect_workday > 0 else x.cam_effect_workday, axis=1)

    for cam_type in ['旺季促销活动', '旺季主题活动', '淡季促销活动', '淡季主题活动', '网络购物节期间举办的活动']:
        fig = plt.figure(figsize=figsize)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.subplots_adjust(left=0.20, right=0.95, )
        tmp = df[df['cam_category'] == cam_type]
        if tmp is None:
            continue
        a1 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['cam_diff'], color=color_dict[cam_cat_color[cam_type]]['diff'])
        a2 = plt.barh(y=np.arange(0, len(tmp)) * 4, width=tmp['pred_y_no_cam_daily_mean'], color=color_dict[cam_cat_color[cam_type]]['daily'])
        a3 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['cam_offday_diff'],
                      color=color_dict[cam_cat_color[cam_type]]['diff'])
        a4 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 1, width=tmp['pred_y_no_cam_daily_mean_offday'],
                      color=color_dict[cam_cat_color[cam_type]]['offday'])
        a5 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['cam_workday_diff'],
                      color=color_dict[cam_cat_color[cam_type]]['diff'])
        a6 = plt.barh(y=np.arange(0, len(tmp)) * 4 + 2, width=tmp['pred_y_no_cam_daily_mean_workday'],
                      color=color_dict[cam_cat_color[cam_type]]['workday'])
        plt.xlim(-8000, 82000)
        plt.xticks([-8000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000])
        plt.xlabel('客流（人）')
        plt.yticks(np.arange(0, len(tmp)) * 4, tmp['cam_name'], fontsize=10)
        plt.legend([a2, a4, a6, a1], ['平均日客流', '休息日平均日客流', '工作日平均日客流', '活动客流增量'], loc='lower right', frameon=False)
        plt.title(cam_type, fontsize=12)
        if is_savefig:
            plt.savefig(os.path.join(pic_dir, "campaign_innum_with_hol_{}.png".format(cam_type)))
        if is_show:
            plt.show()
        plt.close()
    logger.info("Finish plotting pic {}".format('营销活动带来的客流增长绝对值'))


def show_each_cat_cam(df, bad_cam_dict, cam_category='旺季促销活动', figsize=(8, 6),
                      is_savefig=True, is_show=True, pic_dir=PIC_DATA_DIR):
    fig = plt.figure(figsize=figsize)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['font.sans-serif'] = ['SimHei']
    title_list = ['平均每日', '休息日', '工作日', '平均每天活动费用']
    col_dict = {'平均每日': ('daily', 'cam_effect'), '休息日': ('offday', 'cam_effect_offday'),
                '工作日': ('workday', 'cam_effect_workday')}
    col_list = [col_name[1] for key, col_name in col_dict.items()]

    tmp = df[df['cam_category'] == cam_category]
    tmp = tmp.sort_values('avg_budget_used').reset_index(drop=True)
    tmp[col_list] = tmp[col_list].where(tmp[col_list] > 0, 0)
    ix = 1
    fig, ax = plt.subplots(1, 4, sharey="all")
    fig.subplots_adjust(left=0.2, right=0.9, wspace=0.1)
    ax_id = 0
    for i in title_list[:3]:
        tmp_ax = ax[ax_id]
        tmp_ax.barh(y=np.arange(0, len(tmp)), width=tmp[col_dict[i][1]], color=color_dict[cam_cat_color[cam_category]][col_dict[i][0]],
                    alpha=0.6, edgecolor=color_dict[cam_cat_color[cam_category]][col_dict[i][0]])
        tmp_ax.set_xlabel("客流增量（人）", fontsize=7)
        tmp_ax.set_xlim([-100, 25000])
        tmp_ax.set_xticks([0, 5000, 10000, 15000, 20000])
        tmp_ax.set_xticklabels([0, 5000, 10000, 15000, 20000], fontdict={'fontsize': 5})
        tmp_ax.set_yticks(np.arange(0, len(tmp)))
        tmp_ax.set_yticklabels(tmp['cam_name'], fontdict={'fontsize': 6})
        tmp_ax.set_title(i, fontsize=8)
        for j in np.arange(0, len(tmp)):
            tmp_ax.text(x=tmp[col_dict[i][1]][j] + 1500, y=j - 0.4, s=int(tmp[col_dict[i][1]][j]),
                        fontsize=6)
        [k.set_color("red") for k in tmp_ax.get_yticklabels() if k.get_text() in bad_cam_dict[cam_category]]
        ax_id = ax_id + 1
    tmp_ax = ax[3]
    tmp_ax.scatter(x=tmp['avg_budget_used'], y=np.arange(0, len(tmp)), color='k', s=5)
    for j in np.arange(0, len(tmp)):
        tmp_ax.text(x=tmp['avg_budget_used'][j] + 10000, y=j - 0.4, s=int(tmp['avg_budget_used'][j]),
                    fontsize=6)
    tmp_ax.set_xlabel("平均日费用（元）", fontsize=7)
    tmp_ax.set_xticks([0, 100000, 200000])
    tmp_ax.set_xticklabels([0, 100000, 200000], fontdict={'fontsize': 5})
    tmp_ax.set_title(title_list[3], fontsize=8)
    tmp_ax.set_xlim([0, 300000])
    if is_savefig:
        plt.savefig(os.path.join(pic_dir, "campaign_{}_barh.png".format(cam_category)))
    if is_show:
        plt.show()
    logger.info("Finish plotting pic {}".format("campaign_{}_barh.png".format(cam_category)))


def cal_cam_cat_effect(df):
    cam_effect_lst = []
    for cam_category, cam_category_tmpdf in df.groupby('cam_category'):
        cam_effect = (cam_category_tmpdf['cam_effect'] / cam_category_tmpdf['pred_y_no_cam_daily_mean']).mean()
        cam_effect_off = (cam_category_tmpdf['cam_effect_offday'] / cam_category_tmpdf['pred_y_no_cam_daily_mean_offday']).mean()
        cam_effect_work = (cam_category_tmpdf['cam_effect_workday'] / cam_category_tmpdf['pred_y_no_cam_daily_mean_workday']).mean()
        cam_effect_lst.append([cam_category, cam_effect, cam_effect_off, cam_effect_work])
    return pd.DataFrame(cam_effect_lst, columns=['cam_category', 'cam_effect', 'cam_effect_off', 'cam_effect_work'])


def show_cam_increased_ratio(df, is_show=True, is_savefig=True, pic_dir=PIC_DATA_DIR):
    fig = plt.figure(figsize=(10, 6))
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    cam_effect_df = cal_cam_cat_effect(df)
    tmp = cam_effect_df[cam_effect_df['cam_category'] == '旺季促销活动']
    ax1 = plt.bar([0], tmp['cam_effect'], color=color_dict[cam_cat_color['旺季促销活动']]['daily'])
    ax2 = plt.bar([1], tmp['cam_effect_off'], color=color_dict[cam_cat_color['旺季促销活动']]['offday'])
    ax3 = plt.bar([2], tmp['cam_effect_work'], color=color_dict[cam_cat_color['旺季促销活动']]['workday'])
    for a, b in zip([0, 1, 2],
                    [tmp['cam_effect'].values[0], tmp['cam_effect_off'].values[0], tmp['cam_effect_work'].values[0]]):
        plt.text(a, b + 0.005, '{:.2f}%'.format(float(b) * 100), ha='center', va='bottom', fontsize=10)

    tmp = cam_effect_df[cam_effect_df['cam_category'] == '旺季主题活动']
    bx1 = plt.bar([4], tmp['cam_effect'], color=color_dict[cam_cat_color['旺季主题活动']]['daily'])
    bx2 = plt.bar([5], tmp['cam_effect_off'], color=color_dict[cam_cat_color['旺季主题活动']]['offday'])
    bx3 = plt.bar([6], tmp['cam_effect_work'], color=color_dict[cam_cat_color['旺季主题活动']]['workday'])
    for a, b in zip([4, 5, 6],
                    [tmp['cam_effect'].values[0], tmp['cam_effect_off'].values[0], tmp['cam_effect_work'].values[0]]):
        plt.text(a, b + 0.005, '{:.2f}%'.format(float(b) * 100), ha='center', va='bottom', fontsize=10)

    tmp = cam_effect_df[cam_effect_df['cam_category'] == '淡季促销活动']
    cx1 = plt.bar([8], tmp['cam_effect'], color=color_dict[cam_cat_color['淡季促销活动']]['daily'])
    cx2 = plt.bar([9], tmp['cam_effect_off'], color=color_dict[cam_cat_color['淡季促销活动']]['offday'])
    cx3 = plt.bar([10], tmp['cam_effect_work'], color=color_dict[cam_cat_color['淡季促销活动']]['workday'])
    for a, b in zip([8, 9, 10],
                    [tmp['cam_effect'].values[0], tmp['cam_effect_off'].values[0], tmp['cam_effect_work'].values[0]]):
        plt.text(a, b + 0.005, '{:.2f}%'.format(float(b) * 100), ha='center', va='bottom', fontsize=10)

    tmp = cam_effect_df[cam_effect_df['cam_category'] == '淡季主题活动']
    dx1 = plt.bar([12], tmp['cam_effect'], color=color_dict[cam_cat_color['淡季主题活动']]['daily'])
    dx2 = plt.bar([13], tmp['cam_effect_off'], color=color_dict[cam_cat_color['淡季主题活动']]['offday'])
    dx3 = plt.bar([14], tmp['cam_effect_work'], color=color_dict[cam_cat_color['淡季主题活动']]['workday'])
    for a, b in zip([12, 13, 14],
                    [tmp['cam_effect'].values[0], tmp['cam_effect_off'].values[0], tmp['cam_effect_work'].values[0]]):
        plt.text(a, b + 0.005, '{:.2f}%'.format(float(b) * 100), ha='center', va='bottom', fontsize=10)

    tmp = cam_effect_df[cam_effect_df['cam_category'] == '网络购物节期间举办的活动']
    dx1 = plt.bar([16], tmp['cam_effect'], color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['daily'])
    dx2 = plt.bar([17], tmp['cam_effect_off'], color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['offday'])
    dx3 = plt.bar([18], tmp['cam_effect_work'], color=color_dict[cam_cat_color['网络购物节期间举办的活动']]['workday'])
    for a, b in zip([16, 17, 18],
                    [tmp['cam_effect'].values[0], tmp['cam_effect_off'].values[0], tmp['cam_effect_work'].values[0]]):
        plt.text(a, b + 0.005, '{:.2f}%'.format(float(b) * 100), ha='center', va='bottom', fontsize=10)

    plt.xticks([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], ['平均每日', '休息日\n\n旺季促销活动', '工作日',
                                                                      '平均每日', '休息日\n\n旺季主题活动', '工作日',
                                                                      '平均每日', '休息日\n\n淡季促销活动', '工作日',
                                                                      '平均每日', '休息日\n\n淡季主题活动 ', '工作日',
                                                                      '平均每日', '休息日\n\n网络购物节期间举办的活动 ', '工作日'], fontsize=8)

    plt.ylim(0.0, 0.15)
    plt.yticks([0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15], ['0', '2.5%', '5%', '7.5%', '10%', '12.5%', '15%'])
    plt.title('营销活动带来的客流增长百分比', fontsize=20)
    if is_savefig:
        plt.savefig(os.path.join(pic_dir, "campaign_category_effect.png"))
    if is_show:
        plt.show()
    logger.info("Finish plotting pic {}".format('campaign_category_effect'))


if __name__ == '__main__':
    cam_eval_df = read_from_csv('cam_eval_df.csv', RESULT_DIR)

    show_cam_increased_innum_single_plot(cam_eval_df, is_show=False, is_savefig=True)

    show_cam_increased_ratio(cam_eval_df, is_show=False, is_savefig=True)
