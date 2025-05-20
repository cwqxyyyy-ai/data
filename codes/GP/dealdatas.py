import random

import numpy as np
import math
from itertools import combinations

class lambdaMARTDataDeal:
    def __init__(self, data=None):
        self.data = data  # 原始数据
    def group_split(self, index, group_num, qid_num, random_sate): # 测试集索引，分割组数，数据查询总数，随机种子
        """将数据随机分为5组，并设置随机种子，保证每次训练数据一致"""
        # 生成整数区间 [0, 199] 的整数列表
        numbers = list(range(1, qid_num + 1))
        random.seed(random_sate)
        # # 将整数列表随机打乱
        random.shuffle(numbers)
        # 将打乱后的整数列表分成5组

        chunk_size = math.floor(qid_num/group_num)
        groups = [numbers[i:i + chunk_size] for i in range(0, qid_num, chunk_size)]
        groups_index = [group for group in groups]
        # 将 groups_index 除了 index 的部分合并为训练集
        # 将第 index 组作为测试集
        test_indices = groups_index[index]
        train_data = []
        test_data = []
        for row in self.data:
            if row[0] in test_indices:
                test_data.append(row)
            else:
                train_data.append(row)

        return np.array(train_data), np.array(test_data)
    def split(self, test_size, qidNum, random_state):
        if qidNum != 0:
            np.random.seed(random_state)  # 设置随机种子
            size = math.floor(qidNum * test_size)  # 测试集数目
            random_numbers = np.random.choice(np.arange(1, qidNum + 1), size=size, replace=False)
            train_data = []
            test_data = []
            for row in self.data:
                if row[0] in random_numbers:
                    test_data.append(row)
                else:
                    train_data.append(row)
            if qidNum == 2 and len(train_data) < len(test_data):
                return np.array(test_data), np.array(train_data)
            else:
                return np.array(train_data), np.array(test_data)
        else:
            np.random.seed(random_state)  # 设置随机种子
            size = math.floor(len(self.data) * test_size)  # 测试集数目
            random_numbers = np.random.choice(np.arange(0, len(self.data)), size=size, replace=False)
            # print(random_numbers)
            train_data = []
            test_data = []
            for i in range(len(self.data)):
                arr = self.data[i]
                if i in random_numbers:
                    test_data.append(np.insert(arr, 0, 2))
                else:
                    train_data.append(np.insert(arr, 0, 1))
            return np.array(train_data), np.array(test_data)

    def combination(self, combination_data, num, combination_type):
        # 用于存放组合结果的数组
        combin_result = []
        if combination_type == 'repeat':
            id_num = 0
            for combo in combinations(combination_data, num):
                id_num += 1
                for arr in combo:
                    new_arr = [id_num] + list(arr)  # 创建一个新的列表而不是直接修改原列表
                    combin_result.append(new_arr)
            return np.array(combin_result)
        elif combination_type == 'nonrepeat':
            split_num = math.ceil(len(combination_data) / num)
            id_num = 0
            for i in range(len(combination_data)):
                if i % split_num == 0:
                    id_num += 1
                new_arr = [id_num] + list(combination_data[i])  # 创建一个新的列表而不是直接修改原列表
                combin_result.append(new_arr)
            return np.array(combin_result)
        else:
            return "参数错误，应该为'repeat'或'nonrepeat'"
