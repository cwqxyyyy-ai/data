import random
import sys
from pathlib import Path
# 获取当前脚本的绝对路径（假设脚本在 MLP 目录下）
current_script_path = Path(__file__).resolve()
# 项目根目录（即 "实验" 目录）
project_root = current_script_path.parent.parent
# 将根目录添加到 Python 模块搜索路径
sys.path.append(str(project_root))
from lambdaMARTs import lambdaMART
from GP.dealdatas import lambdaMARTDataDeal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")
def main():
    df = pd.read_excel('../data/landmarkDataSets.xlsx', sheet_name='数据集')
    data = np.array(df.drop(['地标编号', '名称'], axis=1))
    dealdata = lambdaMARTDataDeal(data)


    # 模型开始训练
    # start_time = time.time()
    predicted = []
    total_ndcg = []
    total_precision = []
    total_top1Right = []
    max_iter = 1
    for i in range(max_iter):  # 每次迭代是独立的
        rs = 678
        train_data, test_data = dealdata.group_split(2, 5, 255, rs)
        # train_data, test_data = dealdata.split(0.2, 200, 52)  # （测试集比例，查询id总数，随机种子）
        print('训练集/测试集总数：', len(train_data), '/', len(test_data))
        print('start Fold ' + str(i + 1))
        # sklearn_DTR / sklearn_RF / XGBoost/LightGBM/catboost/GP
        model = lambdaMART(
            train_data,
            number_of_trees=10,
            learning_rate=0.01,
            tree_type='sklearn_DTR',
            method=False
        )
        model.fit()
        # model.save('./model/lambdamart_model_%d' % (i + 1))
        trainRes = model.validate(train_data)[1]
        testRes = model.validate(test_data)[1]

        print('\n', f"                          train Accuracy evaluation")
        print('                           ---------------------')
        print(
            f"{'average_ndcg':<16}{'rightRate_succedSort':<24}{'rightRate_top1':<18}{'average_rightRate'}")
        print('-' * 80)
        print(
            f"   {trainRes['average_ndcg']:<16}   {trainRes['rightRate_succedSort']:<23}"
            f"{trainRes['rightRate_top1']:<19}{trainRes['average_rightRate']}")

        # predicted_scores = model.validate(test_data)
        # predicted.append(predicted_scores)
        print('\n', f"                          test Accuracy evaluation")
        print('                           ---------------------')
        print(
            f"{'average_ndcg':<16}{'rightRate_succedSort':<24}{'rightRate_top1':<18}{'average_rightRate'}")
        print('-' * 80)
        print(
            f"   {testRes['average_ndcg']:<16}   {testRes['rightRate_succedSort']:<23}"
            f"{testRes['rightRate_top1']:<19}{testRes['average_rightRate']}")

    # 模型结束训练
    # end_time = time.time()
    # print('训练总耗时：', end_time - start_time)


if __name__ == '__main__':
    main()
