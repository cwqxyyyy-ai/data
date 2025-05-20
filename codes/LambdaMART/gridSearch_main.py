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

    randomState = [66, 123, 253, 163, 300, 534, 678, 777, 865]
    learning_rate = [0.01]
    number_of_trees = [10]
    for lr in learning_rate:
        for nt in number_of_trees:
            for rs in randomState:
                for i in range(5):
                    train_data, test_data = dealdata.group_split(i, 5, 255, rs)
                    # sklearn_DTR / sklearn_RF / XGBoost/LightGBM/catboost/GP
                    model = lambdaMART(
                        train_data,
                        number_of_trees=nt,
                        learning_rate=lr,
                        tree_type='sklearn_DTR',
                        method=False
                    )
                    model.fit()
                    # model.save('./model/lambdamart_model_%d' % (i + 1))
                    trainRes = model.validate(train_data)[1]
                    testRes = model.validate(test_data)[1]
                    # predicted_scores = model.validate(test_data)
                    # predicted.append(predicted_scores)
                    if testRes['rightRate_top1'] > 0.4:
                        print('最佳参数：')
                        print(
                            {'learning_rate': lr,
                             'number_of_trees': nt,
                             'i': i,
                             'random_state': rs}
                        )

                        # print(top1)
                        print('\n', f"                           Accuracy evaluation")
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
