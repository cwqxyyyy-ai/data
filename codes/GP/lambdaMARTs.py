import numpy as np
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import TweedieRegressor, BayesianRidge
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn import neighbors
from catboost import Pool as PoolCat
from multiprocessing import Pool
import pickle

from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
def dcg(scores):
    return np.sum([
        (np.power(2, scores[i]) - 1) / np.log2(1 + (i + 1))  # i加一个1，因为i从0开始，而计算dcg值的公式中的i是从1开始的
        for i in range(len(scores))
    ])


def ideal_dcg(scores):
    scores = sorted(scores)[::-1]  # 从大到小排序
    return dcg(scores)


def single_dcg(scores, i, j):
    if i < j:
        return (np.power(2, scores[i]) - 1) / np.log2(1 + (i + 1)) \
            + (np.power(2, scores[j]) - 1) / np.log2(1 + (j + 1))
    else:
        return (np.power(2, scores[i]) - 1) / np.log2(1 + (j + 1)) \
            + (np.power(2, scores[j]) - 1) / np.log2(1 + (i + 1))
    # return (np.power(2, scores[i]) - 1) / np.log2(1 + (j + 1))


def precision(predicted):
    totalNum = 0
    rightNum = 0
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            totalNum += 1
            if predicted[i] >= predicted[j]:
                rightNum += 1
    rightRate = rightNum / totalNum
    return rightRate


def top1Result(predicted):
    return all(predicted[0] >= predicted[i] for i in range(1, len(predicted)))
def mrr(predicted):
    return 1/(np.argmax(predicted)+1)

def compute_lambda(args):
    true_scores, predicted_scores, good_ij_pairs, idcg, query_key, method = args
    num_docs = len(true_scores)
    sorted_indexes = np.argsort(predicted_scores)[::-1]  # 倒叙排序，获取排序后对应排序前的元素索引
    rev_indexes = np.argsort(sorted_indexes)  # 获取原始顺序的索引数组
    true_scores = true_scores[sorted_indexes]
    predicted_scores = predicted_scores[sorted_indexes]
    lambdas = np.zeros(num_docs)
    w = np.zeros(num_docs)

    k_true_value = sorted(true_scores, reverse=True)
    # print(predicted_scores)
    for i, j in good_ij_pairs:
        if method:  # 判断是否使用改进后的方法
            k = abs(predicted_scores[i] / predicted_scores[j])  # 为什么要以绝对值,应该用预测值的比值还是真实值的比值可以考虑一下？
            if np.isnan(k):
                k = 1
            # if k < 1:
            #     k = abs(predicted_scores[j] / predicted_scores[i])
            zij_ndcg = abs(
                (np.power(single_dcg(true_scores, i, j), k) - np.power(single_dcg(true_scores, j, i), k)) / idcg
            )
            # print(k)
        else:
            # zij_ndcg = abs((single_dcg(true_scores, i, j) - single_dcg(true_scores, i, i)
            #                 + single_dcg(true_scores, j, i) - single_dcg(true_scores, j, j)) / idcg)
            zij_ndcg = abs((single_dcg(true_scores, i, j) - single_dcg(true_scores, j, i)) / idcg)
        c = -1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
        lambda_ij = c * zij_ndcg  # i文档的梯度λi, 则j文档的梯度为-lambda_ij
        lambdas[i] += lambda_ij
        lambdas[j] -= lambda_ij

        c_complement = 1.0 - c
        w_val = c * zij_ndcg * c_complement
        w[i] += w_val
        w[j] += w_val
    return lambdas[rev_indexes], w[rev_indexes], query_key


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


class lambdaMART:
    def __init__(self, training_data=None, number_of_trees=5, learning_rate=0.1, tree_type='sklearn_DTR', method=False):

        # 检查树的类型是否合法
        if tree_type not in ['sklearn_DTR', 'original', 'sklearn_RF', 'XGBoost', 'LightGBM', 'catboost', 'GP']:
            raise ValueError('The "tree_type" must be one of the allowed tree types.')

        self.training_data = training_data  # 训练数据
        self.number_of_trees = number_of_trees  # 树的数量
        self.learning_rate = learning_rate  # 学习率
        self.trees = []  # 保存训练出来的树
        self.tree_type = tree_type  # 树的类型
        self.method = method  # 是否使用改进后的方法，False表示不使用，True表示使用，默认为False

    def fit(self):
        """
        根据训练数据拟合模型
        """
        if self.training_data is None:
            raise ValueError('Training data is not provided.')

        predicted_scores = np.zeros(len(self.training_data))  # 初始化预测值
        # 根据training_data中的数据，提取查询信息
        id_indexes_result = id_indexes(self.training_data, 0)
        query_keys = id_indexes_result.keys()
        true_scores = [self.training_data[id_indexes_result[key_id], 1] for key_id in query_keys]
        good_ij_pairs = get_pairs(true_scores)  # 获取好的配对
        idcg = [ideal_dcg(scores) for scores in true_scores]  # 计算理想的dcg值
        methods = [self.method for _ in true_scores]  # 获取是否改进数组，用于传入lambda中进行判断计算

        # 开始训练树
        for k in range(self.number_of_trees):
            print('Tree %d' % (k + 1))
            lambdas = np.zeros(len(predicted_scores))  # 初始化lambda数组
            w = np.zeros(len(predicted_scores))  # 初始化w数组
            pred_scores = [predicted_scores[id_indexes_result[key_id]] for key_id in query_keys]  # 预测分数

            # 使用多进程计算lambda和w
            with Pool() as pool:
                for lambda_val, w_val, query_key in pool.map(compute_lambda,
                                                             zip(true_scores, pred_scores, good_ij_pairs, idcg,
                                                                 query_keys,
                                                                 methods),
                                                             chunksize=1):
                    indexes = id_indexes_result[query_key]
                    lambdas[indexes] = lambda_val
                    w[indexes] = w_val  # 权重

            # 训练树，并更新预测分数
            if self.tree_type == 'sklearn_DTR':
                # Sklearn实现的树
                tree = DecisionTreeRegressor(max_depth=5)  # 和随机森林设置的参数一样
                tree.fit(self.training_data[:, 2:], lambdas)
                self.trees.append(tree)
                prediction = tree.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate
            elif self.tree_type == 'GP':
                def _ndcg_rightRate(y_true, y_pred, w):
                    if len(y_pred) != len(self.training_data[:, 1]):
                        # print('不对')
                        return 0
                    ndcgs_rightRates = []
                    for key_id in id_indexes_result:
                        pred = y_pred[id_indexes_result[key_id]]
                        predicted_sorted_indexes = np.argsort(pred)[::-1]  # 获取预测值倒序索引号

                        t_results = y_true[id_indexes_result[key_id]]
                        #  计算每个查询文档下的ndcg
                        t_results_sort = t_results[predicted_sorted_indexes]
                        dcg_val = dcg(t_results_sort)
                        idcg_val = ideal_dcg(t_results_sort)
                        ndcg = dcg_val / idcg_val
                        rightRate = precision(t_results_sort)
                        ndcgs_rightRates.append(ndcg * rightRate)
                    return np.nanmean(ndcgs_rightRates)
                    """Calculate the root mean squared error."""
                ndcg_rightRate = make_fitness(function=_ndcg_rightRate, greater_is_better=True)

                # 初始化并训练模型
                tree = SymbolicRegressor(
                    metric=ndcg_rightRate,  #
                    population_size=500,  #
                    generations=6,  #
                    stopping_criteria=1,  # 停止标准，当达到指定精度（0.98）时停止迭代

                    p_crossover=0.87,  # 交叉概率，个体交叉的概率为75%
                    p_subtree_mutation=0.07,  # 子树变异概率，个体中子树发生变异的概率为10%
                    # p_hoist_mutation=0.025,  # 提升变异概率，个体中子树提升的概率为5%
                    # p_point_mutation=0.025,  # 点变异概率，个体中单点发生变异的概率为10%

                    # p_point_replace=0.05,
                    # max_samples=0.9,  # 最大样本比例，训练时使用的样本比例为90%
                    verbose=0,  # 输出详细程度，设为1时输出训练过程中的详细信息
                    # parsimony_coefficient=0.01,  # 简约系数，用于惩罚过于复杂的表达式，防止过拟合
                    # function_set=['add', 'sub', 'mul', 'div', 'abs', 'log'],
                    function_set=['add', 'sub', 'mul', 'div', 'log'],
                    init_depth=(2, 6),  # 初始化深度，生成初始种群时个体的最大深度范围为3到5
                    # tournament_size=30,
                    # init_method='grow',
                    random_state=0,  # 随机种子，确保实验结果可重复
                )
                self.trees.append(tree)
                # 创建Yeo-Johnson变换器
                # pt = PowerTransformer(method='yeo-johnson')
                # # 应用Yeo-Johnson变换
                # lambdas = pt.fit_transform(lambdas.reshape(-1, 1)).flatten()
                tree.fit(self.training_data[:, 2:], lambdas)
                prediction = tree.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate
                # # 提取最佳个体表达式
                # print("Best program:", tree._program)
            elif self.tree_type == 'sklearn_RF':
                tree = RandomForestRegressor(n_estimators=6, max_features=4, max_depth=14, min_samples_split=6,
                                           random_state=75)
                tree.fit(self.training_data[:, 2:], lambdas)
                self.trees.append(tree)
                prediction = tree.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate
            elif self.tree_type == 'XGBoost':
                tree = XGBRegressor(
                    objective='reg:squarederror',  # 指定目标函数：平方误差回归
                    n_estimators=3,  # 树的数量（迭代次数）
                    learning_rate=0.01,  # 学习率
                    max_depth=5  # 树的最大深度
                )
                tree.fit(self.training_data[:, 2:], lambdas)
                self.trees.append(tree)
                prediction = tree.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate
            elif self.tree_type == 'LightGBM':
                # 训练数据集
                train_data = lgb.Dataset(self.training_data[:, 2:][23:], label=lambdas[23:])
                valid_data = lgb.Dataset(self.training_data[:, 2:][:23], label=lambdas[:23])
                # 测试数据集（验证集）
                # test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
                # 设置参数
                params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9
                }
                # 训练模型
                num_round = 10
                model = lgb.train(params, train_data, num_round, valid_sets=[valid_data])
                # 预测
                prediction = model.predict(self.training_data[:, 2:], num_iteration=model.best_iteration)
                predicted_scores += prediction * self.learning_rate
            elif self.tree_type == 'catboost':
                # 创建CatBoost数据池,
                train_pool = PoolCat(data=self.training_data[:, 2:][23:], label=lambdas[23:])
                test_pool = PoolCat(data=self.training_data[:, 2:][:23], label=lambdas[:23])
                # 设置参数
                params = {
                    'loss_function': 'RMSE',
                    'iterations': 10,
                    'learning_rate': 0.01,
                    'depth': 6,
                    'verbose': False
                }

                # 训练模型
                model = CatBoostRegressor(**params)
                model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=10, verbose_eval=False)

                # 预测
                prediction = model.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate
    def validate(self, data):
        if data is None:
            raise ValueError('Validation data is not provided.')

        id_indexes_result = id_indexes(data, 0)
        average_ndcg = []
        average_rightRate = []
        top1 = []
        predicted_scores = np.zeros(len(data))  # 初始化预测分数
        for key_id in id_indexes_result:
            p_results = np.zeros(len(id_indexes_result[key_id]))
            for tree in self.trees:
                p_results += self.learning_rate * tree.predict(data[id_indexes_result[key_id], 2:])
            predicted_scores[id_indexes_result[key_id]] = p_results  # 预测结果
            predicted_sorted_indexes = np.argsort(p_results)[::-1]  # 获取预测值倒序索引号

            t_results = data[id_indexes_result[key_id], 1]
            #  计算每个查询文档下的ndcg
            t_results_sort = t_results[predicted_sorted_indexes]
            dcg_val = dcg(t_results_sort)
            idcg_val = ideal_dcg(t_results_sort)
            ndcg_val = (dcg_val / idcg_val)
            average_ndcg.append(ndcg_val)
            #  计算每个查询文档下的准确率
            average_rightRate.append(precision(t_results_sort))
            # 计算每个查询文档的TOP1正确率
            top1.append(top1Result(t_results_sort))

        # 获取ndcg
        average_ndcg = round(np.nanmean(average_ndcg), 3)

        # 获取精确度
        rightRate_succedSort = round(average_rightRate.count(1.0)/len(average_rightRate),3)
        average_rightRate = round(np.nanmean(average_rightRate), 3)

        # 获取top1RightRate
        rightRate_top1 = round(sum(top1) / len(top1), 3)
        # print(top1)
        print('\n', "                           Accuracy evaluation")
        print('                           ---------------------')
        print(f"{'average_ndcg':<16}{'rightRate_succedSort':<24}{'rightRate_top1':<18}{'average_rightRate'}")
        print('-' * 80)
        print(f"   {average_ndcg:<16}   {rightRate_succedSort:<23}{rightRate_top1:<19}{average_rightRate}")

        return predicted_scores

    def save(self, fname):
        """
        Saves the model into a ".lmart" file with the name given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to save
        """
        with open('%s.lmart' % fname, "wb") as f:
            pickle.dump(self, f, protocol=2)

    def load(self, fname):
        """
        Loads the model from the ".lmart" file given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to load
        """
        with open(fname, "rb") as f:
            model = pickle.load(f)
            self.training_data = model.training_data
            self.number_of_trees = model.number_of_trees
            self.tree_type = model.tree_type
            self.learning_rate = model.learning_rate
            self.trees = model.trees