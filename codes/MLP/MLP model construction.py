import sys
from pathlib import Path

# 获取当前脚本的绝对路径（假设脚本在 MLP 目录下）
current_script_path = Path(__file__).resolve()

# 项目根目录（即 "实验" 目录）
project_root = current_script_path.parent.parent  # MLP → 实验

# 将根目录添加到 Python 模块搜索路径
sys.path.append(str(project_root))

# 现在可以正常导入 GP 模块
from GP.dealdatas import lambdaMARTDataDeal  # 根据实际文件名调整
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
import warnings
from GP.dealdatas import lambdaMARTDataDeal
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def id_indexes(training_data, id_position):
    query_indexes = {}  # 创建一个空字典用于存储查询ID和
    index = 0  # 初始化索引为0
    # 遍历训练数据中的每条记录
    for record in training_data:
        # 将记录的查询ID作为字典的键，如果键不存在则设为空列表
        query_indexes.setdefault(record[id_position], [])
        # 将当前索引添加到与查询ID相关联的索引列表中
        query_indexes[record[id_position]].append(index)
        index += 1  # 增加索引
    return query_indexes  # 返回查询ID和索引的字典

def precision(predicted):
    rightNum = np.sum([predicted[i - 1] >= predicted[i] for i in range(1, len(predicted))])
    rightRate = rightNum / (len(predicted) - 1)
    return rightRate

def top1Result(predicted):
    return all(predicted[0] >= predicted[i] for i in range(1, len(predicted)))

def mrr(predicted):
    return 1 / (np.argmax(predicted) + 1)

def evaluate_pred(y_true, y_pred, id_indexes_result_evaluate):
    # 预测评价
    average_rightRate = []
    top1 = []
    mrrs = []
    for key_id in id_indexes_result_evaluate:
        p_results = y_pred[id_indexes_result_evaluate[key_id]]
        predicted_sorted_indexes = np.argsort(p_results)[::-1]  # 获取预测值倒序索引号
        # print(p_results)

        t_results = y_true[id_indexes_result_evaluate[key_id]]
        #  计算每个查询文档下的ndcg
        t_results_sort = t_results[predicted_sorted_indexes]
        #  计算每个查询文档下的准确率
        average_rightRate.append(precision(t_results_sort))
        # 计算每个查询文档的TOP1正确率
        top1.append(top1Result(t_results_sort))

        mrrs.append(mrr(t_results_sort))

    # 获取ndcg
    average_mrr = round(np.nanmean(mrrs), 3)

    # 获取精确度
    rightRate_succedSort = round(average_rightRate.count(1.0) / len(average_rightRate), 3)
    # 获取top1RightRate
    rightRate_top1 = round(sum(top1) / len(top1), 3)
    average_rightRate = round(np.nanmean(average_rightRate), 3)
    return {
        'average_mrr': average_mrr,
        'rightRate_succedSort': rightRate_succedSort,
        'rightRate_top1': rightRate_top1,
        'average_rightRate': average_rightRate
    }

df = pd.read_excel('../data/landmarkDataSets.xlsx', sheet_name='数据集')
data = np.array(df.drop(['地标编号', '名称'], axis=1))
dealdata = lambdaMARTDataDeal(data)
rs = 15
train_data, test_data = dealdata.group_split(1, 5, 255, rs)
scaler = MinMaxScaler()
X = scaler.fit_transform(train_data[:, 2:])
y = train_data[:, 1]
X_test = scaler.transform(test_data[:, 2:])
y_test = test_data[:, 1]

id_indexes_result_train = id_indexes(train_data, 0)
id_indexes_result_test = id_indexes(test_data, 0)


# The model is constructed by the optimal parameter combination obtained by parameter optimization.
MLP = MLPRegressor(activation='tanh', hidden_layer_sizes=(3, 3), learning_rate_init=0.1, max_iter=200, random_state=rs, solver='lbfgs')
MLP_m = MLP.fit(X, y)
# Accuracy evaluation of training set
y_train_pred = MLP_m.predict(X)
results = evaluate_pred(y, y_train_pred, id_indexes_result_train)
print('Best individual 训练集精度评价(rightRate_top1/rightRate_succedSort)：', results['rightRate_top1'], results['rightRate_succedSort'], '\n')


# 预测评价
y_test_pred = MLP_m.predict(X_test)
results = evaluate_pred(y_test, y_test_pred, id_indexes_result_test)

print('\n', f"                        test Accuracy evaluation")
print('                           ---------------------')
print(f"{'average_mrr':<16}{'rightRate_succedSort':<24}{'rightRate_top1':<18}{'average_rightRate'}")
print('-' * 80)
print(
    f"   {results['average_mrr']:<16}   {results['rightRate_succedSort']:<23}{results['rightRate_top1']:<19}{results['average_rightRate']}")
print('\n')



