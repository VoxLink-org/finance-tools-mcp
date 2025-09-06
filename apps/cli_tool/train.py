import pickle
from config.my_paths import DATA_DIR
from packages.predict_lib.train_down_v3 import main

import sys
period = sys.argv[1] if len(sys.argv) > 1 else "1y"
feature_importances, model, report = main(period=period, end_date_str="2025-04-10")
with open(DATA_DIR / 'xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
"""
牛市价差策略，

上涨胜率 x 权利金 = 上涨败率 * 预测股价 x 平均失败冲击变动 - 权利金 * 上涨败率


上涨胜率 x 权利金 + 权利金 * 上涨败率 = 

解出可以获利的权利金 = 上涨败率 * 预测股价 x 平均失败冲击变动 

"""

18