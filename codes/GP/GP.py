import functools
import random

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms, gp
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor

from skimage.metrics import mean_squared_error

from dealdatas import lambdaMARTDataDeal

from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from sklearn.preprocessing import MinMaxScaler

def get_pairs(scores):
    query_pair = []
    for query_scores in scores:
        temp = sorted(query_scores, reverse=True)
        pairs = []
        for i in range(len(temp)):
            for j in range(i + 1, len(temp)):
                # print(j)
                if temp[i] >= temp[j]:
                    pairs.append((i, j))
        query_pair.append(pairs)
    return query_pair


def dcg(scores):
    return np.sum([
        (np.power(2, scores[i]) - 1) / np.log2(1 + (i + 1))  # i加一个1，因为i从0开始，而计算dcg值的公式中的i是从1开始的
        for i in range(len(scores))
    ])


def ideal_dcg(scores):
    scores = sorted(scores)[::-1]  # 从大到小排序
    return dcg(scores)


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


def err(relevance_scores):
    erred = 0.0
    p = 1.0
    max_grade = np.max(relevance_scores)
    for i, relevance in enumerate(relevance_scores):
        r = (2 ** relevance - 1) / (2 ** max_grade)
        erred += p * r / (i + 1)
        p *= (1 - r)
    return erred


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


# 数据
df = pd.read_excel('../data/landmarkDataSets.xlsx', sheet_name='数据集')
data = np.array(df.drop(['地标编号', '名称'], axis=1))
dealdata = lambdaMARTDataDeal(data)

# 初始化并训练模型
# 自定义MSE适应度函数
def mse_(y_true, y_pred, w):
    mse = mean_squared_error(y_true, y_pred)
    return mse
mse = make_fitness(function=mse_, greater_is_better=False)
if __name__ == '__main__':
    rs = 333
    train_data, test_data = dealdata.group_split(0, 5, 255, rs)
    id_indexes_result_train = id_indexes(train_data, 0)
    id_indexes_result_test = id_indexes(test_data, 0)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(train_data[:, 2:])
    y = train_data[:, 1]
    X_test = scaler.transform(test_data[:, 2:])
    y_test = test_data[:, 1]

    # 初始化并训练模型
    p = 300  #
    cxpb = 0.91  # 交叉概率
    mutpb = 0.04  # 变异概率
    ngen = 30  # 进化代数
    max_depth = 10
    est = SymbolicRegressor(
        metric=mse,  #
        population_size=p,  #
        generations=ngen,  #
        stopping_criteria=0,  # 停止标准，当达到指定精度时停止迭代
        p_crossover=cxpb,  # 交叉概率，个体交叉的概率为75%
        p_subtree_mutation=mutpb,  # 子树变异概率，个体中子树发生变异的概率为10%
        # p_hoist_mutation=0.03,  # 提升变异概率，个体中子树提升的概率为5%
        # p_point_mutation=0.02,  # 点变异概率，个体中单点发生变异的概率为10%

        # p_point_replace=0.1,

        max_samples=1,  # 最大样本比例，训练时使用的样本比例为90%
        verbose=True,  # 输出详细程度，设为1时输出训练过程中的详细信息
        # parsimony_coefficient=0.0001,  # 简约系数，用于惩罚过于复杂的表达式，防止过拟合
        function_set=['add', 'sub', 'mul', 'div', 'abs', 'log', 'sqrt', 'sin'],
        # function_set=['add', 'sub', 'log', 'sqrt', 'sin'],
        init_depth=(3, max_depth),  # 初始化深度，生成初始种群时个体的最大深度范围为3到5
        # tournament_size=30,
        # init_method='grow',
        random_state=66,  # 随机种子，确保实验结果可重复
    )

    est.fit(X, y)
    best_individual = est._program
    y_train_pred = est.predict(X)
    print("Best individual (as formula):", est._program)

    results = evaluate_pred(y, y_train_pred, id_indexes_result_train)
    print('Best individual 训练集精度评价(rightRate_top1/rightRate_succedSort)：', results['rightRate_top1'], results['rightRate_succedSort'], '\n')


    # 预测评价
    y_test_pred = est.predict(X_test)
    results = evaluate_pred(y_test, y_test_pred, id_indexes_result_test)

    print('\n', f"                        test Accuracy evaluation")
    print('                           ---------------------')
    print(f"{'average_mrr':<16}{'rightRate_succedSort':<24}{'rightRate_top1':<18}{'average_rightRate'}")
    print('-' * 80)
    print(
        f"   {results['average_mrr']:<16}   {results['rightRate_succedSort']:<23}{results['rightRate_top1']:<19}{results['average_rightRate']}")
    print('\n')


