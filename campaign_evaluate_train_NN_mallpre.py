import os
from utils.logger import logger

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tarot.pred.funcs.nn_preprocess_final_v0_1 import CampaignDataInputer

from tarot.pred.mall_campagin.config import CAM_PROCESSED_DATA_DIR, RESULT_DIR, MODEL_DIR, RAW_DATA_DIR
from tarot.pred.mall_campagin.campaign_evaluate_train import CAT_LST, DROP_LST, PRED_DAY_LST
from tarot.pred.mall_campagin.campaign_evaluate_train import deal_nan_feature, MODEL_TS_FEATURE_PARAM_DICT
from tarot.pred.train_model import ts_feature_process, feature_process
from tarot.pred.train_model import lgb_reg_train

from utils.data_porter import read_from_csv, save_to_pkl, save_to_csv, read_from_pkl
from sklearn.metrics import mean_squared_error as mse


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


def model_test(test_x, run, model_dir, state):
    global booster
    for pred_days in PRED_DAY_LST:  # 对应的长短期中每个模型循环
        # 读取模型
        own_model_dir = os.path.join(model_dir, "model_after_NN_train")
        if run:
            booster = read_from_pkl(f'model_random_seed_{state}_{pred_days}d.pkl', own_model_dir)
        else:
            booster = read_from_pkl(f'model_{pred_days}d.pkl', own_model_dir)

    # test_shap = booster.predict(test_x, pred_contrib=True)
    test = booster.predict(test_x)
    test_trans = np.transpose(test)
    test_df = pd.DataFrame(test_trans, columns=['predict_test_y']).reset_index(drop=True)
    # test_shap_df = pd.DataFrame(test_df, columns=(feature_lst+['predict_test_y']))
    # interact_df = test_df['predict_test_y']

    return test_df


def gen_x(data):
    return data[input_1].astype("float64"), data[input_2].astype("float64")


if __name__ == '__main__':
    mall_data = read_from_csv('mall_data.csv', CAM_PROCESSED_DATA_DIR, dtype={'holiday_name': str,
                                                                              'holiday_type': str},
                              parse_dates=['datetime'])
    mall_data.loc[:, CAT_LST] = mall_data.loc[:, CAT_LST].fillna('NULL').reset_index(drop=True)

    # 对空值进行一些处理
    mall_data = deal_nan_feature(mall_data)
    feature_lst = list()
    for pred_day in PRED_DAY_LST:  # 循环每个需要训练的模型
        logger.debug(f"Start training for {pred_day} day shift!")
        logger.debug("Process feature data!")
        # 调用对应的长期或短期时序特征生成函数
        data_with_ts_feature = ts_feature_process(
            mall_data, y_name='innum', shift=pred_day,
            ts_feature_param_dict=MODEL_TS_FEATURE_PARAM_DICT[pred_day])
        mall_data, feature_lst = feature_process(
            data_with_ts_feature, y_name='innum', features=None,
            drop_lst=DROP_LST, cat_lst=CAT_LST)

    if 'mall_id' in feature_lst:
        feature_lst.remove('mall_id')

    pred_id_lst = np.random.choice(pd.unique(mall_data['mall_id']), size=20, replace=False)
    mall_data = mall_data[feature_lst + ['innum', 'datetime', 'mall_id']]

    # Use mall_data to fit the NN model
    cam_feature = ['cam_type_theme', 'cam_type_sale', 'cam_type_0', 'th_cam_day', 'rem_cam_days']

    # Keep important features when averaging the models' predictions
    features_imprt = read_from_csv('feature_importance.csv', RAW_DATA_DIR)
    features_imprt = features_imprt[features_imprt['importance'] > 1000]

    data_inputer = CampaignDataInputer(mall_data, CAT_LST, feature_lst, cam_feature, features_imprt)
    data_inputer.prepare_campaign_data(
        train_start="2019-01-01",
        valid_split="2019-10-01",
        valid_end="2019-12-31",
        pred_id_lst=pred_id_lst
    )

    train_y_all = pd.DataFrame()
    valid_y_all = pd.DataFrame()
    test_y_all = pd.DataFrame()
    test_y_base_all = pd.DataFrame()

    predict_test_y = pd.DataFrame()
    predict_test_y_base = pd.DataFrame()
    interaction_diff = pd.DataFrame()

    for iters in range(1, 21):
        train, valid, test, test_base = data_inputer.gen_campaign_data()
        input_1, input_2 = data_inputer.input_1, data_inputer.input_2
        input_all = input_1 + input_2

        input_shapes = [(len(input_1),), (len(input_2),)]
        model = mall_model(input_shapes)

        opt = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
        model.compile(optimizer=opt, loss="mse",
                      metrics=['mean_absolute_percentage_error'])

        model.fit(x=gen_x(train),
                  y=train['innum'].astype("float64"),
                  epochs=3, verbose=1, steps_per_epoch=256,
                  validation_data=(gen_x(valid),
                                   valid['innum'].astype("float64")),
                  validation_steps=128)

        # Construct model that can output cam_layer of the NN model
        cam_layer_model = Model(inputs=model.input,
                                outputs=model.get_layer('cam_prediction').output)

        # Modify the cam_prediction_df’s format
        cam_prediction_df = pd.DataFrame([i[0] for i in cam_layer_model.predict(x=gen_x(test))],
                                         columns=['cam_prediction'])
        cam_prediction_base_df = pd.DataFrame([i[0] for i in cam_layer_model.predict(x=gen_x(test_base))],
                                              columns=['cam_prediction_base'])

        # Construct model that can output non_cam_layer of the NN model
        non_cam_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer('non_cam_prediction').output)
        non_cam_prediction_df = pd.DataFrame([i[0] for i in non_cam_layer_model.predict(x=gen_x(test))],
                                             columns=['non_cam_prediction'])

        # Fetch the resid-1, )
        train_y_all[f'y_residual_{iters}'] = train['innum'].values - model.predict(gen_x(train)).reshape(-1, )
        valid_y_all[f'y_residual_{iters}'] = valid['innum'].values - model.predict(gen_x(valid)).reshape(-1, )
        test_y_all[f'y_residual_{iters}'] = test['innum'].values - model.predict(gen_x(test)).reshape(-1, )
        test_y_base_all[f'y_residual_{iters}'] = test_base['innum'].values - model.predict(
            gen_x(test_base)).reshape(-1, )

        # Output the performance of the whole NN model
        history = model.evaluate(x=gen_x(train), y=train['innum'].astype("float64"))
        print("Loss = " + str(history[0]))

        # Use residuals to fit lgb model
        train_args_dict = {'eval_metric': 'rmse'}
        model_args_dict = {
            'num_leaves': 64,
            'n_estimators': 100000,
            'random_state': iters,
            'objective': 'rmse'
        }

        train_and_test(pred_days=PRED_DAY_LST[0],
                       x_train=train[input_all],
                       y_train=train_y_all[f'y_residual_{iters}'],
                       x_val=valid[input_all],
                       y_val=valid_y_all[f'y_residual_{iters}'],
                       x_test=test[input_all],
                       y_test=test_y_all[f'y_residual_{iters}'],
                       model_args=model_args_dict,
                       train_args=train_args_dict,
                       res_dir=RESULT_DIR,
                       model_dir=MODEL_DIR,
                       other_info=f'_random_seed_{iters}',
                       innum_=test['innum'],
                       log_y=False)

        predict_test_y[f'interaction_prediction_{iters}'] = model_test(
            test[input_all], run=True, model_dir=MODEL_DIR,
            state=iters).apply(lambda x: x[0], axis=1)

        predict_test_y_base[f'interaction_prediction_{iters}'] = model_test(
            test_base[input_all], run=True, model_dir=MODEL_DIR,
            state=iters).apply(lambda x: x[0], axis=1)

        interaction_diff[f'interaction_diff_{iters}'] = \
            predict_test_y[f'interaction_prediction_{iters}'] - predict_test_y_base[f'interaction_prediction_{iters}']

    interaction_diff['average_diff_for_20_times'] = interaction_diff.apply(
        lambda x: np.mean([x[f'interaction_diff_{i}'] for i in range(1, 21)]), axis=1)

    # Have a glimpse at test_data_with_cam
    cam_df = pd.DataFrame()

    cam_df['datetime'] = data_inputer.test_time
    cam_df['mall_id'] = data_inputer.test_id

    cam_df['cam_effect'] = interaction_diff['average_diff_for_20_times']

    # cam_df['cam_prediction'] = cam_prediction_df['cam_prediction']
    # cam_df['non_cam_prediction'] = non_cam_prediction_df['non_cam_prediction']

    # cam_df[f'interaction_prediction_{iters}'] = predict_test_y
    # cam_df[f'interaction_prediction_base_{iters}'] = predict_test_y_base

    # cam_df['diff'] = cam_df['interaction_prediction'] - \
    #                  cam_df['interaction_prediction_base'] + cam_df['cam_prediction']

    save_to_csv(cam_df, 'cam_test_data.csv', CAM_PROCESSED_DATA_DIR)
