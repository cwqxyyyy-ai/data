import functools
import random

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms, gp
import operator
import math
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
    # totalNum = 0
    # rightNum = 0
    # for i in range(len(predicted)):
    #     for j in range(i + 1, len(predicted)):
    #         totalNum += 1
    #         if predicted[i] >= predicted[j]:
    #             rightNum += 1
    # rightRate = rightNum / totalNum
    # return rightRate


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


def evaluate_pred(individual, id_indexes_result_evaluate, X_evaluate, y_evaluate):
    func = gp.compile(expr=individual, pset=pset)
    y_pred = np.array([func(*x) for x in X_evaluate])  # 使用编译后的函数进行预测
    # 预测评价
    average_rightRate = []
    top1 = []
    mrrs = []
    for key_id in id_indexes_result_evaluate:
        p_results = y_pred[id_indexes_result_evaluate[key_id]]
        predicted_sorted_indexes = np.argsort(p_results)[::-1]  # 获取预测值倒序索引号
        # print(p_results)

        t_results = y_evaluate[id_indexes_result_evaluate[key_id]]
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
if __name__ == '__main__':
    population_size = [500]
    cxpb = [0.95]
    mutpb = [0.15]
    mus = [160, 200]  #  父代个体数
    lambda_s = [360, 400]  # 子代个体数
    ngen = [30]  # 进化代数
    randomState = [13, 34, 46, 68, 99, 117, 147, 246, 289, 376, 333, 476, 573, 666]
    max_depth = [12]
    verbose = False
    searchNums = len(population_size) * len(cxpb) * len(mutpb) * len(ngen) * len(randomState) * len(max_depth) * len(mus) * len(lambda_s)
    print("搜索总次数：", searchNums)
    searchNum = 1
    for rs in randomState:
        for p in population_size:
            for md in max_depth:
                for mu in mus:
                    for lambda_ in lambda_s:
                        for n in ngen:
                            for c in cxpb:
                                for m in mutpb:
                                    if c + m <= 99999 and mu < lambda_:
                                        top1s = []
                                        rightRate_succedSorts = []
                                        for i in range(1):
                                            train_data, test_data = dealdata.group_split(i, 5, 255, rs)
                                            scaler = MinMaxScaler()
                                            X = scaler.fit_transform(train_data[:, 2:])
                                            y = train_data[:, 1]
                                            X_test = scaler.transform(test_data[:, 2:])
                                            y_test = test_data[:, 1]

                                            id_indexes_result_train = id_indexes(train_data, 0)
                                            id_indexes_result_test = id_indexes(test_data, 0)

                                            def log(x):
                                                return math.log(x) if x > 0 else 0  # 或者其他适当的值


                                            def sqrt(x):
                                                return math.sqrt(x) if x > 0 else 0


                                            def div(x, y):
                                                return x / y if y != 0 else 1  # 或者其他合适的值


                                            def power_two(x):
                                                return np.power(x, 2)  # 或者其他合适的值


                                            pset = gp.PrimitiveSet("MAIN", arity=X.shape[1])
                                            pset.addPrimitive(operator.add, 2)
                                            pset.addPrimitive(operator.sub, 2)
                                            pset.addPrimitive(operator.mul, 2)
                                            pset.addPrimitive(div, 2)

                                            # pset.addPrimitive(power_two, 1)
                                            pset.addPrimitive(log, 1)
                                            pset.addPrimitive(sqrt, 1)
                                            pset.addPrimitive(math.sin, 1)

                                            if not hasattr(creator, 'FitnessSingle'):
                                                creator.create("FitnessSingle", base.Fitness, weights=(1.0,))
                                            if not hasattr(creator, 'Individual'):
                                                creator.create("Individual", gp.PrimitiveTree,
                                                               fitness=creator.FitnessSingle)

                                            # 初始化工具箱
                                            # 定义最大树深度
                                            toolbox = base.Toolbox()
                                            toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=md)
                                            toolbox.register("individual", tools.initIterate, creator.Individual,
                                                             toolbox.expr)
                                            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

                                            # 注册遗传操作
                                            toolbox.register("mate", gp.cxOnePoint)
                                            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
                                            # 注册修剪操作到交叉和变异中
                                            toolbox.decorate("mate",
                                                             gp.staticLimit(key=operator.attrgetter("height"),
                                                                            max_value=md))
                                            toolbox.decorate("mutate",
                                                             gp.staticLimit(key=operator.attrgetter("height"),
                                                                            max_value=md))

                                            toolbox.register("select_roulette", tools.selRoulette)


                                            # id_indexes_result_v = id_indexes(v_data, 0)
                                            # 自定义函数解析表达式并计算系数
                                            def parse_and_fit(individual):
                                                """提取满足条件的子表达式"""
                                                subexpressions = []
                                                limit_len = 0
                                                for i, node in enumerate(individual):
                                                    if node.name in ['add', 'sub']:
                                                        if limit_len > 12:
                                                            break
                                                        limit_len += 1
                                                        # 提取子表达式
                                                        # 左
                                                        left_slice = individual.searchSubtree(i + 1)
                                                        left_subexpr = gp.PrimitiveTree(individual[left_slice])
                                                        # 右
                                                        right_slice = individual.searchSubtree(
                                                            i + len(left_subexpr) + 1)

                                                        right_subexpr = gp.PrimitiveTree(individual[right_slice])
                                                        subexpressions.append((left_subexpr, right_subexpr))

                                                """对子表达式进行回归并返回权重"""
                                                if len(subexpressions) == 0:
                                                    return individual
                                                else:
                                                    new_X = []
                                                    for left, right in subexpressions:
                                                        left_func = gp.compile(expr=left, pset=pset)
                                                        right_func = gp.compile(expr=right, pset=pset)
                                                        new_X.append([[left_func(*x), right_func(*x)] for x in X])

                                                    new_X = np.hstack(np.array(new_X)).tolist()
                                                    # 创建回归模型
                                                    ransac = RANSACRegressor(random_state=666)
                                                    ransac.fit(new_X, y)
                                                    coefficients = ransac.estimator_.coef_  # 获取系数
                                                    intercept = ransac.estimator_.intercept_  # 获取截距

                                                    # TheilSen = TheilSenRegressor(random_state=rs)
                                                    # TheilSen.fit(new_X, y)
                                                    # coefficients = TheilSen.coef_  # 获取系数
                                                    # intercept = TheilSen.intercept_  # 获取截距

                                                    # Linear = LinearRegression(fit_intercept=True)
                                                    # Linear.fit(new_X, y)
                                                    # coefficients = Linear.coef_  # 获取系数
                                                    # intercept = Linear.intercept_  # 获取截距

                                                    """根据回归结果生成新表达式"""
                                                    # 初始化一个列表来保存加权后的子表达式
                                                    weighted_subexprs = []
                                                    for j, subexpression in enumerate(subexpressions):
                                                        left_subexpr, right_subexpr = subexpression

                                                        # 创建加权左子表达式
                                                        new_left_subexpr = [
                                                            gp.Primitive("mul", [pset.ret, pset.ret], pset.ret),
                                                            gp.Terminal(round(coefficients[j * 2], 3), pset.ret,
                                                                        pset.ret)
                                                        ]
                                                        new_left_subexpr.extend(left_subexpr)

                                                        # 创建加权右子表达式
                                                        new_right_subexpr = [
                                                            gp.Primitive("mul", [pset.ret, pset.ret], pset.ret),
                                                            gp.Terminal(round(coefficients[j * 2 + 1], 3), pset.ret,
                                                                        pset.ret)
                                                        ]
                                                        new_right_subexpr.extend(right_subexpr)

                                                        # 将两个加权的子表达式加在一起，得到一个加法表达式
                                                        combined_expr = [
                                                            gp.Primitive("add", [pset.ret, pset.ret], pset.ret)
                                                        ]
                                                        combined_expr.extend(new_left_subexpr)
                                                        combined_expr.extend(new_right_subexpr)

                                                        # 将组合后的表达式添加到列表中
                                                        weighted_subexprs.append(combined_expr)

                                                    # 如果有多个子表达式，确保它们依次连接起来
                                                    if len(weighted_subexprs) > 1:
                                                        # 从第一个开始合并所有的子表达式
                                                        final_expr = weighted_subexprs[0]
                                                        for expr in weighted_subexprs[1:]:
                                                            new_add_expr = [
                                                                gp.Primitive("add", [pset.ret, pset.ret], pset.ret)
                                                            ]
                                                            new_add_expr.extend(final_expr)
                                                            new_add_expr.extend(expr)
                                                            final_expr = new_add_expr
                                                    else:
                                                        # 如果只有一个子表达式，直接使用它
                                                        final_expr = weighted_subexprs[0]

                                                    # 创建一个新的个体，将所有部分相加
                                                    new_individual = gp.PrimitiveTree(final_expr)

                                                    # 打印新个体
                                                    # print("新的个体表达式：", new_individual)
                                                    if intercept != 0:
                                                        individualed = gp.PrimitiveTree(
                                                            [gp.Primitive("add", [pset.ret, pset.ret], pset.ret),
                                                             gp.Terminal(round(intercept, 3), pset.ret, pset.ret),
                                                             *new_individual])
                                                    else:
                                                        individualed = new_individual

                                                    return individualed


                                            # 定义适应度函数
                                            def evaluate(individual, id_indexes_result_evaluate, X_evaluate,
                                                         y_evaluate):
                                                func = gp.compile(expr=individual, pset=pset)
                                                y_pred = np.array([func(*x) for x in X_evaluate])  # 使用编译后的函数进行预测
                                                mrrs = []
                                                rightRates = []
                                                for key_id in id_indexes_result_evaluate:
                                                    pred = y_pred[id_indexes_result_evaluate[key_id]]
                                                    predicted_sorted_indexes = np.argsort(pred)[::-1]  # 获取预测值倒序索引号

                                                    t_results = y_evaluate[id_indexes_result_evaluate[key_id]]
                                                    #  计算每个查询文档下的ndcg
                                                    t_results_sort = t_results[predicted_sorted_indexes]

                                                    mrrs.append(mrr(t_results_sort))
                                                    rightRates.append(precision(t_results_sort))

                                                mrr_mean = np.nanmean(mrrs)
                                                precision_mean = np.nanmean(rightRates)
                                                # 使用指数函数进行加权
                                                fitness = (np.exp(mrr_mean) - 1) * (
                                                            np.exp(precision_mean) - 1)  # 避免单个指标占主导地位
                                                return fitness,


                                            toolbox.register("evaluate", evaluate)

                                            from concurrent.futures import ThreadPoolExecutor


                                            def process_individual(args):
                                                """处理个体，计算其新表达式和适应度"""
                                                ind, id_indexes_result_train, X, y = args
                                                new_ind = parse_and_fit(ind)
                                                if new_ind:
                                                    new_tree = gp.PrimitiveTree(new_ind)
                                                    new_individual = creator.Individual(new_tree)
                                                    fitness = toolbox.evaluate(new_individual, id_indexes_result_train, X, y)
                                                    if fitness[0] > 0:
                                                        new_individual.fitness.values = fitness
                                                        return new_individual
                                                return None


                                            def parallel_evaluate(population, id_indexes_result_train, X, y, n_jobs=4):
                                                """并行处理种群中的所有个体"""
                                                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                                                    # 准备参数列表
                                                    args_list = [(ind, id_indexes_result_train, X, y) for ind in
                                                                 population]
                                                    # 并行执行
                                                    results = list(executor.map(process_individual, args_list))

                                                # 更新种群中的有效个体
                                                for i, new_ind in enumerate(results):
                                                    if new_ind:  # 如果新个体有效，则替换原种群中的个体
                                                        population[i] = new_ind

                                                return population


                                            def adaptive_rate(fitness, max_fitness, avg_fitness, base_rate, type):
                                                rate = 0
                                                epsilon = 1e-6  # 防止除零错误
                                                if (type == 'pc'):
                                                    if fitness > avg_fitness:
                                                        rate = base_rate * ((max_fitness - fitness) / (
                                                                    max_fitness - avg_fitness + epsilon))
                                                        # rate = base_rate * np.power((max_fitness - fitness) / (max_fitness - avg_fitness), 2)
                                                    else:
                                                        rate = base_rate
                                                if (type == 'pm'):
                                                    if fitness > avg_fitness:
                                                        rate = base_rate * ((max_fitness - fitness) / (
                                                                    max_fitness - avg_fitness + epsilon))
                                                        # rate = base_rate * np.power((max_fitness - fitness) / (max_fitness - avg_fitness), 2)
                                                    else:
                                                        rate = base_rate
                                                return max(min(rate, 1), 0)
                                            # 运行遗传算法

                                            def main():
                                                random.seed(666)
                                                population = toolbox.population(n=p)
                                                # 设置遗传算法参数
                                                all_best_individual = []
                                                # 运行遗传算法并计算个体适应度
                                                # new_population = []
                                                n_jobs = 14  # 并行线程数，可以根据机器核数调整
                                                population = parallel_evaluate(population, id_indexes_result_train, X,
                                                                               y, n_jobs=n_jobs)
                                                for gen in range(n):
                                                    elites = tools.selBest(population, 3)
                                                    # 常规选择生成下一代种群，排除精英个体
                                                    selected_population = toolbox.select_roulette(population, mu)
                                                    # 计算适应度，更新交叉和变异概率
                                                    fitness_vals = [ind.fitness.values[0] for ind in selected_population]
                                                    max_fitness = max(fitness_vals)
                                                    avg_fitness = sum(fitness_vals) / len(fitness_vals)
                                                    # print(max_fitness, avg_fitness)

                                                    # 生成子代
                                                    offspring = []
                                                    while len(offspring) <= lambda_:
                                                        parent1, parent2 = random.sample(selected_population, 2)
                                                        # 自适应调整交叉率和变异率
                                                        fitness1 = parent1.fitness.values[0]
                                                        fitness2 = parent2.fitness.values[0]
                                                        cxpb_ = adaptive_rate((fitness1 + fitness2) / 2, max_fitness,
                                                                              avg_fitness, c, 'pc')
                                                        mutpb_ = adaptive_rate((fitness1 + fitness2) / 2, max_fitness,
                                                                               avg_fitness, m, 'pm')

                                                        while cxpb_ + mutpb_ > 1:
                                                            scale = 1 / (cxpb_ + mutpb_)
                                                            cxpb_ *= scale
                                                            mutpb_ *= scale
                                                        offspring.extend(
                                                            algorithms.varOr([parent1, parent2], toolbox, 2, cxpb_,
                                                                             mutpb_))
                                                    # 更新筛选后的个体的最终适应度值
                                                    for i, offs in enumerate(offspring):
                                                        offspring[i].fitness.values = toolbox.evaluate(offs,
                                                                                                       id_indexes_result_train,
                                                                                                       X, y)

                                                    population[:] = elites + offspring

                                                    best_individual = tools.selBest(population, 1)[0]
                                                    # results_test = \
                                                    # toolbox.evaluate(best_individual, id_indexes_result_test, X_test,
                                                    #                  y_test)[0]
                                                    # results_train = \
                                                    # toolbox.evaluate(best_individual, id_indexes_result_train, X, y)[0]
                                                    # all_best_individual.append(
                                                    #     [best_individual, results_test])
                                                    results_train = evaluate_pred(best_individual, id_indexes_result_train,
                                                                            X,
                                                                            y)
                                                    results_test = evaluate_pred(best_individual, id_indexes_result_test,
                                                                                  X_test,
                                                                                  y_test)
                                                    if results_train['rightRate_top1'] >= 0.8 and results_test['rightRate_top1'] >= 0.8:
                                                        print('最佳参数：')
                                                        print(
                                                                f"population_size: {p}; cxpb: {c}; mutpb; {m}; ngen: {n}; random_state: {rs}; max_depth: {md}; mu: {mu}; lambda_: {lambda_}")
                                                        print(str(best_individual))
                                                        print("top1 train/test:", results_train['rightRate_top1'], results_test['rightRate_top1'])
                                                        print("RA train/test:", results_train['rightRate_succedSort'],
                                                              results_test['rightRate_succedSort'])
                                                    # all_best_individual.append(
                                                    #     [best_individual, results['rightRate_top1'],
                                                    #      results['rightRate_succedSort']])

                                                return all_best_individual


                                            all_best_individual = main()
                                        #     if len(all_best_individual) > 0:
                                        #         all_best_individual_top1 = [item[1] for item in all_best_individual]  #
                                        #         reversed_indices = np.argmax(all_best_individual_top1[::-1])
                                        #         maxIndex = len(all_best_individual_top1) - reversed_indices - 1
                                        #
                                        #         best_individual = all_best_individual[maxIndex][0]
                                        #         # 预测评价
                                        #         results = evaluate_pred(best_individual, id_indexes_result_test, X_test,
                                        #                                 y_test)
                                        #         top1s.append(results['rightRate_top1'])
                                        #         rightRate_succedSorts.append(results['rightRate_succedSort'])
                                        #
                                        # if np.nanmean(top1s) > 0.8 or max(top1s) > 0.8:
                                        #     print('最佳参数：')
                                        #     print(
                                        #         f"population_size: {p}; cxpb: {c}; mutpb; {m}; ngen: {n}; random_state: {rs}; max_depth: {md}; mu: {mu}; lambda_: {lambda_}")
                                        #     print('top1:', top1s, np.average(top1s))
                                        #     print('rightRate_succedSort:', rightRate_succedSorts,
                                        #           np.nanmean(rightRate_succedSorts),
                                        #           '\n')
                                        #     print('---------------------------------------')
                                        print(f"\r已完成{searchNum}次，还剩{searchNums - searchNum}次", end='')
                                        searchNum = searchNum + 1
