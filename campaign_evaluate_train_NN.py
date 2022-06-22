import copy
import os
import random

from utils.logger import logger

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from tarot.pred.mall_campagin.config import CAM_PROCESSED_DATA_DIR, RESULT_DIR, MODEL_DIR, RAW_DATA_DIR
from tarot.pred.mall_campagin.campaign_evaluate_train import CAT_LST, DROP_LST, PRED_DAY_LST
from tarot.pred.mall_campagin.campaign_evaluate_train import deal_nan_feature, MODEL_TS_FEATURE_PARAM_DICT
from tarot.pred.train_model import ts_feature_process, feature_process, innum_mean_feature_process

from utils.data_porter import read_from_csv, save_to_pkl, save_to_csv, read_from_pkl
from tarot.pred.train_model import lgb_reg_train

from sklearn.metrics import mean_squared_error as mse
from tarot.pred.funcs.metrics import mape


pred_day_lst = [15]

def over_sampling(df):
    cam_df = df[df['cam_type'] != '0']
    no_cam_df = df[df['cam_type'] == '0']
    len_cam_df = len(cam_df)
    len_no_cam_df = len(no_cam_df)
    lst = np.random.choice(range(len_cam_df), size=len_no_cam_df, replace=True)
    cam_df = cam_df.iloc[lst]
    df = pd.concat([no_cam_df, cam_df], axis=0)
    return df


def one_hot(df, column):
    new_df = df.drop(columns=[column])
    values = set(df[column].values)
    for value in values:
        new_df[column + '_%s' % value] = df[column].apply(lambda x: 1 if x == value else 0)

    return new_df


def mall_model(shapes):
    x1_shape, x2_shape = shapes
    x1_input = Input(x1_shape, name='x1')
    x1 = Dense(128, activation=tf.nn.relu)(x1_input)
    x1 = Dense(1, name='non_cam_prediction')(x1)
    x2_input = Input(x2_shape, name='x2')
    x2 = Dense(8, activation=tf.nn.relu)(x2_input)
    x2 = Dense(4)(x2)
    x2 = Dense(1, name='cam_prediction')(x2)
    y_hat = Add()([x1, x2])
    nn_model = Model(inputs=[x1_input, x2_input], outputs=y_hat, name='mall_model')

    return nn_model


def train_and_test(pred_days, x_train, y_train, x_val, y_val, x_test, y_test,
                   model_args, train_args, res_dir, model_dir, other_info,
                   innum_, log_y=False):
    # 训练模型
    logger.debug("Start train model")
    model = lgb_reg_train(x_train, y_train, x_val, y_val,
                          model_args=model_args, train_args=train_args)
    # 保存模型
    own_model_dir = os.path.join(model_dir, "model_after_NN_train")
    save_to_pkl(
        model.booster_, f'model{other_info}_{pred_days}d.pkl', own_model_dir)
    # 生成测试结果
    logger.debug("Test model!")
    predict_test_y = model.booster_.predict(x_test)
    true_test_y = y_test
    if log_y:
        predict_test_y = np.expm1(predict_test_y)
        true_test_y = np.expm1(true_test_y)
    pred_res_all = pd.DataFrame()
    pred_res_all['y'] = true_test_y
    pred_res_all['pred_y'] = predict_test_y
    # 保存测试结果
    own_result_dir = os.path.join(res_dir, "model_after_NN_test")
    save_to_csv(pred_res_all, f'pred_res{other_info}_{pred_days}_d.csv', own_result_dir)
    total_rmse = mse(pred_res_all['y'], pred_res_all['pred_y']) ** 0.5

    # define the mape(represent score of the whole model compromising of NN model & lgb model)
    residual = np.array(pred_res_all['y']).astype(np.float64)
    pred_y = np.array(pred_res_all['pred_y']).astype(np.float64)
    innum_ = np.array(innum_).astype(np.float64)
    total_mape = np.nanmean(np.abs((residual - pred_y) / np.clip(np.abs(innum_), 1e-7, None)))
    logger.info(f"Total RMSE: {total_rmse}!")
    logger.info(f"Total MAPE: {total_mape}!")


def model_test(test_x, feature_lst, run, model_dir, STATE):
    for pred_days in pred_day_lst:  # 对应的长短期中每个模型循环
        # 读取模型
        own_model_dir = os.path.join(model_dir, "model_after_NN_train")
        if run:
            booster = read_from_pkl(f'model_random_seed_{STATE}_{pred_days}d.pkl', own_model_dir)
        else:
            booster = read_from_pkl(f'model_{pred_days}d.pkl', own_model_dir)

    # test_shap = booster.predict(test_x, pred_contrib=True)
    test = booster.predict(test_x)
    test_trans = np.transpose(test)
    test_df = pd.DataFrame(test_trans, columns=['predict_test_y']).reset_index(drop=True)
    # test_shap_df = pd.DataFrame(test_df, columns=(feature_lst+['predict_test_y']))
    # interact_df = test_df['predict_test_y']

    return test_df


if __name__ == '__main__':
    mall_data = read_from_csv('mall_data.csv', CAM_PROCESSED_DATA_DIR, dtype={'holiday_name': str, 'holiday_type': str},
                              parse_dates=['datetime'])



    # 对空值进行一些处理
    mall_data = deal_nan_feature(mall_data)
    mall_data.loc[:, CAT_LST] = mall_data.loc[:, CAT_LST].fillna('NULL')
    for pred_day in PRED_DAY_LST:  # 循环每个需要训练的模型
        logger.debug(f"Start training for {pred_day} day shift!")
        logger.debug("Process feature data!")
        # 调用对应的长期或短期时序特征生成函数
        data_with_ts_feature = ts_feature_process(
            mall_data, y_name='innum', shift=pred_day,
            ts_feature_param_dict=MODEL_TS_FEATURE_PARAM_DICT[pred_day])
        data_with_ts_feature = innum_mean_feature_process(data_with_ts_feature)
        mall_data, feature_lst = feature_process(
            data_with_ts_feature, y_name='innum', features=None,
            drop_lst=DROP_LST, cat_lst=CAT_LST)
    mall_data.drop(columns=['summer_innum_mean', 'spring_festival_innum_mean', 'spring_festival_after_innum_mean',
                            'week_innum_mean', 'weekend_innum_mean'])

    # Only focus on the time range from 2018-01-01 to 2019-12-31
    # Specifically, exclude the mall that opens after 2018-01-01
    mall_data = mall_data[(mall_data['datetime']>'2017-12-31')&(mall_data['datetime']<'2020-01-01')]
    mall_date_min = mall_data.groupby(['mall_id']).apply(lambda x:x['datetime'].min())
    mall_date_min.name = 'start_date'
    mall_date_min = mall_date_min.reset_index()
    mall_id_lst = mall_date_min.loc[mall_date_min['start_date']<='2018-01-01', 'mall_id'].tolist()
    mall_data = mall_data[mall_data['mall_id'].isin(mall_id_lst)]
    mall_id_unique = mall_data['mall_id'].unique().tolist() # 143

    # Fetch mall_id、datetime of the test data
    test_data_time = mall_data.loc[
        (mall_data['datetime'] > '2019-09-30') & (mall_data['datetime'] < '2020-01-01'), 'datetime']
    test_data_time = pd.DataFrame(test_data_time, columns=['datetime']).reset_index(drop=True)
    test_data_id = mall_data.loc[
        (mall_data['datetime'] > '2019-09-30') & (mall_data['datetime'] < '2020-01-01'), 'mall_id']
    test_data_id = pd.DataFrame(test_data_id, columns=['mall_id']).reset_index(drop=True)

    mall_id_concat = pd.DataFrame(mall_data['mall_id'])
    mall_id_concat = mall_id_concat.reset_index(drop=True)

    left_features = list(set(feature_lst + ['innum', 'datetime']).difference(set(['mall_id'])))
    mall_data = mall_data[left_features]
    mall_data = mall_data.reset_index(drop=True)

    # impute num features with mean
    NUM_LST = list(set(left_features).difference(set(CAT_LST+['datetime', 'innum'])))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(mall_data[NUM_LST])
    mall_data[NUM_LST] = pd.DataFrame(imp_mean.transform(mall_data[NUM_LST]), columns=NUM_LST)

    for feature in CAT_LST:
        mall_data.loc[:, feature] = mall_data.loc[:, feature].astype('category')
    for feature in NUM_LST:
        mall_data.loc[:, feature] = mall_data.loc[:, feature].astype('float')
    for col in CAT_LST:
        mall_data = one_hot(mall_data, col)
    mall_data = mall_data.reset_index(drop=True)

    # Split the mall_data into 2 parts, 1 contains cam_info, another sets all cam_info to zero
    normalizer = StandardScaler()
    normalizer.fit(mall_data[NUM_LST])
    mall_data.loc[:, NUM_LST] = pd.DataFrame(normalizer.transform(mall_data[NUM_LST]), columns=NUM_LST)

    cat_one_hot = mall_data.columns
    cat_one_hot = list(set(cat_one_hot).difference(set(NUM_LST+['datetime', 'innum'])))
    for feature in cat_one_hot:
        mall_data.loc[:, feature] = mall_data.loc[:, feature].astype('category')
    mall_data = mall_data.reset_index(drop=True)

    # Split train、valid、test based on mall_data for model fitting
    mall_data = pd.concat([mall_id_concat, mall_data], axis=1)

    train_data = mall_data[mall_data['datetime'] < '2019-09-01'].drop(['datetime'], axis=1)
    valid_data = mall_data[(mall_data['datetime'] > '2019-08-31') & (mall_data['datetime'] < '2019-10-01')].drop(
        ['datetime'], axis=1)
    test_data = mall_data[(mall_data['datetime'] > '2019-09-30') & (mall_data['datetime'] < '2020-01-01')].drop(
        ['datetime'], axis=1)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Replace non_cam data's features with 0 & '0'
    cam_feature_cat_0 = ['cam_type_theme', 'cam_type_sale']
    cam_feature_cat_1 = ['cam_type_0']
    cam_feature_num = ['th_cam_day', 'rem_cam_days']
    test_data_base = test_data.copy()
    test_data_base.loc[test_data_base['th_cam_day'] != 0.0, cam_feature_cat_0] = 0
    test_data_base.loc[test_data_base['th_cam_day'] != 0.0, cam_feature_cat_1] = 1
    test_data_base.loc[test_data_base['th_cam_day'] != 0.0, cam_feature_num] = 0

    test_data_base = test_data_base.reset_index(drop=True)

    # Use mall_data to fit the NN model
    cam_feature = ['cam_type_theme', 'cam_type_sale', 'cam_type_0', 'th_cam_day', 'rem_cam_days']
    no_cam_feature = list(set(mall_data.columns.tolist()) - set(cam_feature + ['datetime', 'innum', 'mall_id']))
    features = cam_feature + no_cam_feature

    # Keep important features when averaging the models' predictions
    features_imprt = read_from_csv('feature_importance.csv', RAW_DATA_DIR)
    features_imprt = features_imprt[features_imprt['importance'] > 1000]
    features_kept = []
    for feature_name in features_imprt['feature_name'].tolist():
        if feature_name in NUM_LST:
            features_kept.append(feature_name) # 25
    features_random = list(set(features) - set(features_kept + cam_feature))
    random_feature_num = 35
    feature_1 = random.sample(features_random, random_feature_num)
    feature_2 = random.sample(features_random, random_feature_num)
    feature_3 = random.sample(features_random, random_feature_num)
    feature_4 = random.sample(features_random, random_feature_num)
    feature_5 = random.sample(features_random, random_feature_num)
    feature_6 = random.sample(features_random, random_feature_num)
    feature_7 = random.sample(features_random, random_feature_num)
    feature_8 = random.sample(features_random, random_feature_num)
    feature_9 = random.sample(features_random, random_feature_num)
    feature_10 = random.sample(features_random, random_feature_num)
    feature_11 = random.sample(features_random, random_feature_num)
    feature_12 = random.sample(features_random, random_feature_num)
    feature_13 = random.sample(features_random, random_feature_num)
    feature_14 = random.sample(features_random, random_feature_num)
    feature_15 = random.sample(features_random, random_feature_num)
    feature_16 = random.sample(features_random, random_feature_num)
    feature_17 = random.sample(features_random, random_feature_num)
    feature_18 = random.sample(features_random, random_feature_num)
    feature_19 = random.sample(features_random, random_feature_num)
    feature_20 = random.sample(features_random, random_feature_num)

    train_y_all = pd.DataFrame()
    valid_y_all = pd.DataFrame()
    test_y_all = pd.DataFrame()
    test_y_base_all = pd.DataFrame()

    predict_test_y = pd.DataFrame()
    predict_test_y_base = pd.DataFrame()
    interaction_diff = pd.DataFrame()


    train_data_cp = train_data.copy()
    valid_data_cp = valid_data.copy()
    test_data_cp = test_data.copy()
    test_data_base_cp = test_data_base.copy()

    feature_20_lst = [feature_1, feature_2, feature_3, feature_4, feature_5,
                            feature_6, feature_7, feature_8, feature_9, feature_10,
                            feature_11, feature_12, feature_13, feature_14, feature_15,
                            feature_16, feature_17, feature_18, feature_19, feature_20]

    for random_features in feature_20_lst:
        iteration_times = feature_20_lst.index(random_features) + 1

        # mall_id_set = np.random.choice(mall_id_unique, 50)
        mall_id_set = mall_id_unique

        train_data = train_data_cp[train_data_cp['mall_id'].isin(mall_id_set)].drop(columns=['mall_id'])
        valid_data = valid_data_cp[valid_data_cp['mall_id'].isin(mall_id_set)].drop(columns=['mall_id'])
        test_data = test_data_cp[test_data_cp['mall_id'].isin(mall_id_set)].drop(columns=['mall_id'])
        test_data_base = test_data_base_cp[test_data_base_cp['mall_id'].isin(mall_id_set)].drop(columns=['mall_id'])

        train_data = train_data.reset_index(drop=True)
        valid_data = valid_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        test_data_base = test_data_base.reset_index(drop=True)

        input_1 = features_kept + random_features
        input_shape_1 = len(input_1)
        input_2 = cam_feature
        input_shape_2 = len(input_2)
        input_all = input_1 + input_2

        input_shapes = [(input_shape_1,), (input_shape_2,)]
        model = mall_model(input_shapes)

        opt = Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam')
        model.compile(optimizer=opt, loss="mse",
                      metrics=['mean_absolute_percentage_error'])
        # lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                                     factor=0.1,
        #                                                     patience=3,
        #                                                     verbose=1,
        #                                                     mode='min',
        #                                                     cooldown=0,
        #                                                     min_lr=1e-10)
        # tf_callback = tf.keras.callbacks.TensorBoard(
        #     log_dir=r'C:\code\tarot_avengers\tarot\pred\mall_campagin\logs',
        #     histogram_freq=1, batch_size=32,
        #     write_graph=True, write_grads=False, write_images=True,
        #     update_freq=500
        # )

        # , callbacks=[tf_callback, lr_reduction]
        model.fit(x=[train_data[input_1].astype('float64'), train_data[input_2].astype('float64')],
                  y=train_data['innum'].astype('float64'),
                  epochs=3, verbose=1, steps_per_epoch=256,
                  validation_data=([valid_data[input_1].astype('float64'), valid_data[input_2].astype('float64')],
                                   valid_data['innum'].astype('float64')),
                  validation_steps=128)

        # Construct model that can output cam_layer of the NN model
        cam_layer_model = Model(inputs=model.input,
                                outputs=model.get_layer('cam_prediction').output)

        # Make predictions separately based on cam
        cam_prediction = cam_layer_model.predict(x=[test_data[input_1].astype('float64'),
                                                    test_data[input_2].astype('float64')])
        cam_prediction_base = cam_layer_model.predict(x=[test_data_base[input_1].astype('float64'),
                                                         test_data_base[input_2].astype('float64')])

        # Modify the cam_prediction_df’s format
        cam_prediction_lst = []
        for i in cam_prediction:
            cam_prediction_lst.append(i[0])
        cam_prediction_df = pd.DataFrame(cam_prediction_lst, columns=['cam_prediction']).reset_index(drop=True)

        cam_prediction_base_lst = []
        for i in cam_prediction_base:
            cam_prediction_base_lst.append(i[0])
        cam_prediction_base_df = pd.DataFrame(cam_prediction_base_lst, columns=['cam_prediction_base']).reset_index(drop=True)

        # Construct model that can output non_cam_layer of the NN model
        non_cam_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer('non_cam_prediction').output)
        non_cam_prediction = non_cam_layer_model.predict(x=[test_data[input_1].astype('float64'), test_data[input_2].astype('float64')])
        non_cam_prediction_lst = []
        for i in non_cam_prediction:
            non_cam_prediction_lst.append(i[0])
        non_cam_prediction_df = pd.DataFrame(non_cam_prediction_lst, columns=['non_cam_prediction']).reset_index(drop=True)

        # Caculate predictions of the whole NN model, to fetch the residual for later training
        train_y_hat = pd.DataFrame(model.predict(x=[train_data[input_1].astype('float64'),
                                                    train_data[input_2].astype('float64')]))
        valid_y_hat = pd.DataFrame(model.predict(x=[valid_data[input_1].astype('float64'),
                                                    valid_data[input_2].astype('float64')]))
        test_y_hat = pd.DataFrame(model.predict(x=[test_data[input_1].astype('float64'),
                                                   test_data[input_2].astype('float64')]))
        test_y_base_hat = pd.DataFrame(model.predict(x=[test_data_base[input_1].astype('float64'),
                                                        test_data_base[input_2].astype('float64')]))

        # Fetch the residual
        train_tmp = pd.DataFrame(pd.concat([train_data['innum'].reset_index(drop=True), train_y_hat], axis=1))
        train_tmp.columns = ['y', 'y_hat']
        train_y_all[f'y_residual_{iteration_times}'] = train_tmp.apply(lambda x: x['y'] - x['y_hat'], axis=1)

        valid_tmp = pd.DataFrame(pd.concat([valid_data['innum'].reset_index(drop=True), valid_y_hat], axis=1))
        valid_tmp.columns = ['y', 'y_hat']
        valid_y_all[f'y_residual_{iteration_times}'] = valid_tmp.apply(lambda x: x['y'] - x['y_hat'], axis=1)

        test_tmp = pd.DataFrame(pd.concat([test_data['innum'].reset_index(drop=True), test_y_hat], axis=1))
        test_tmp.columns = ['y', 'y_hat']
        test_y_all[f'y_residual_{iteration_times}'] = test_tmp.apply(lambda x: x['y'] - x['y_hat'], axis=1)

        test_tmp_base = pd.DataFrame(pd.concat([test_data_base['innum'].reset_index(drop=True), test_y_base_hat], axis=1))
        test_tmp_base.columns = ['y', 'y_hat']
        test_y_base_all[f'y_residual_{iteration_times}'] = test_tmp.apply(lambda x: x['y'] - x['y_hat'], axis=1)

        # Output the performance of the whole NN model
        history = model.evaluate(x=[train_data[input_1].astype('float64'),
                                    train_data[input_2].astype('float64')],
                                 y=train_data['innum'].astype('float64'))
        print("Loss = " + str(history[0]))

        # train_y = pd.DataFrame()
        # train_y['y_residual'] = train_y_all.apply(lambda x: (x['y_residual_1']+x['y_residual_2']+x['y_residual_3']
        #                                                     +x['y_residual_4']+x['y_residual_5'])/5, axis=1)
        # valid_y = pd.DataFrame()
        # valid_y['y_residual'] = valid_y_all.apply(lambda x: (x['y_residual_1']+x['y_residual_2']+x['y_residual_3']
        #                                                     +x['y_residual_4']+x['y_residual_5'])/5, axis=1)
        # test_y = pd.DataFrame()
        # test_y['y_residual'] = test_y_all.apply(lambda x: (x['y_residual_1']+x['y_residual_2']+x['y_residual_3']
        #                                                     +x['y_residual_4']+x['y_residual_5'])/5, axis=1)

        # Delete feature 'innum' for later training
        innum_ = test_data['innum']
        train_data = train_data.drop(['innum'], axis=1).reset_index(drop=True)
        valid_data = valid_data.drop(['innum'], axis=1).reset_index(drop=True)
        test_data = test_data.drop(['innum'], axis=1).reset_index(drop=True)
        test_data_base = test_data_base.drop(['innum'], axis=1).reset_index(drop=True)

        # Use residuals to fit lgb model
        train_args_dict = {
            'eval_metric': 'rmse'}
        model_args_dict = {
            'num_leaves': 64, 'n_estimators': 100000, 'random_state': iteration_times, 'objective': 'rmse'}

        train_and_test(pred_days=PRED_DAY_LST[0],
                       x_train=train_data[input_all], y_train=train_y_all[f'y_residual_{iteration_times}'],
                       x_val=valid_data[input_all], y_val=valid_y_all[f'y_residual_{iteration_times}'],
                       x_test=test_data[input_all], y_test=test_y_all[f'y_residual_{iteration_times}'],
                       model_args=model_args_dict, train_args=train_args_dict, res_dir=RESULT_DIR, model_dir=MODEL_DIR,
                       other_info=f'_random_seed_{iteration_times}', innum_=innum_, log_y=False)

        predict_test_y_ = model_test(test_data[input_all], input_all, run=True, model_dir=MODEL_DIR, STATE=iteration_times)
        predict_test_y_ = predict_test_y_.apply(lambda x: x[0], axis=1)
        predict_test_y[f'interaction_prediction_{iteration_times}'] = predict_test_y_

        predict_test_y_base_ = model_test(test_data_base[input_all], input_all, run=True, model_dir=MODEL_DIR, STATE=iteration_times)
        predict_test_y_base_ = predict_test_y_base_.apply(lambda x: x[0], axis=1)
        predict_test_y_base[f'interaction_prediction_{iteration_times}'] = predict_test_y_base_

        interaction_diff[f'interaction_diff_{iteration_times}'] = predict_test_y[f'interaction_prediction_{iteration_times}'] \
                                                                  - predict_test_y_base[f'interaction_prediction_{iteration_times}']


    interaction_diff['average_diff_for_5_times'] = interaction_diff.apply(lambda x: (x['interaction_diff_1']
                                                                                   +x['interaction_diff_2']
                                                                                   +x['interaction_diff_3']
                                                                                   +x['interaction_diff_4']
                                                                                   +x['interaction_diff_5'])/5,
                                                                          axis=1)
    # Have a glimpse at test_data_with_cam
    cam_df = pd.DataFrame()

    cam_df['datetime'] = test_data_time['datetime']
    cam_df['mall_id'] = test_data_id['mall_id']

    cam_df['cam_effect'] = interaction_diff['average_diff_for_5_times']

    # cam_df['cam_prediction'] = cam_prediction_df['cam_prediction']
    # cam_df['non_cam_prediction'] = non_cam_prediction_df['non_cam_prediction']

    # cam_df[f'interaction_prediction_{iteration_times}'] = predict_test_y
    # cam_df[f'interaction_prediction_base_{iteration_times}'] = predict_test_y_base

    # cam_df['diff'] = cam_df['interaction_prediction'] - cam_df['interaction_prediction_base'] + cam_df['cam_prediction']

    save_to_csv(cam_df, 'cam_test_data.csv', CAM_PROCESSED_DATA_DIR)
