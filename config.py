import os
from utils.config import DATA_DIR
from tarot.pred.mall_traffic_pred.config import PROCESSED_DATA_DIR

__pred_data_dir = os.path.join(DATA_DIR, 'pred')
__mall_campaign_data_dir = os.path.join(__pred_data_dir, 'mall_campaign')

MALL_TRAFFIC_PRED_DATA_DIR = PROCESSED_DATA_DIR
RAW_DATA_DIR = os.path.join(__mall_campaign_data_dir, 'raw_data')
CAM_PROCESSED_DATA_DIR = os.path.join(__mall_campaign_data_dir, 'processed_data')
RESULT_DIR = os.path.join(__mall_campaign_data_dir, 'result')
MODEL_DIR = os.path.join(__mall_campaign_data_dir, 'model')
PIC_DATA_DIR = os.path.join(RESULT_DIR, 'pic_data')
