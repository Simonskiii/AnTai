# itemCF,根据商品热度填充
import pandas as pd
import math
from operator import itemgetter
import time
import random
import os
item_id = []
item_cate = {}
dataset = {}
test_dataset = {}
sim_num = 100
rec_num = 30
sim_matrix = {}
item_popular = {}
item_count = pd.DataFrame()
result = {}
attr = pd.DataFrame()
items_cnts = []


def data_processing():
    global attr
    global item_count
    global dataset
    global test_dataset
    global item_id
    global items_cnts
    t1 = time.process_time()
    attr = pd.read_csv('..\\data\\Antai_AE_round1_item_attr_20190626.csv')
    attr = attr.sort_values(by=['cate_id', 'item_id'], ascending=(True, True)).reset_index(drop=True)
    attr = attr[['item_id', 'cate_id']]
    train = pd.read_csv('..\\data\\Antai_AE_round1_train_20190626.csv')
    train = train.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    test = pd.read_csv('..\\data\\Antai_AE_round1_test_20190626.csv')
    test = test.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    item_id = attr.loc[:, 'item_id'].values
    # 购买频率表
    temp = train.loc[train.buyer_country_id == 'yy']
    temp = temp.drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')
    item_count = temp.groupby(['item_id']).size().reset_index()
    item_count.columns = ['item_id', 'cnts']
    item_count = item_count.sort_values('cnts', ascending=False)
    items_cnts = item_count['item_id'].values.tolist()
    # 物品种类表
    for row in attr.itertuples(index=True, name='Pandas'):
        cate = getattr(row, "cate_id")
        item_cate.setdefault(cate, [])
        item_cate[cate].append(getattr(row, "item_id"))
    # 训练集用户购买物品字典
    for row in train.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        dataset.setdefault(admin, {})
        dataset[admin][getattr(row, "item_id")] = 1
    # 测试集用户购买物品字典
    for row in test.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        test_dataset.setdefault(admin, {})
        test_dataset[admin][getattr(row, "item_id")] = 1
    for user, items in dataset.items():
        for item in items:
            if item not in item_popular:
                item_popular[item] = 0
            item_popular[item] += 1
    for user, items in dataset.items():
        for i1 in items:
            for i2 in items:
                if i1 == i2:
                    continue
                sim_matrix.setdefault(i1, {})
                sim_matrix[i1].setdefault(i2, 0)
                sim_matrix[i1][i2] += 1
    print("物品物品矩阵已完毕")
    for i1 in list(sim_matrix.keys()):
        if sim_matrix[i1]:
            for i2 in list(sim_matrix[i1].keys()):
                if sim_matrix[i1][i2] <= 1:
                    del sim_matrix[i1][i2]
        else:
            del sim_matrix[i1]
            print('*********************')
    for u, related_movies in sim_matrix.items():
        for i, count in related_movies.items():
            if item_popular[u] == 0 or item_popular[i] == 0:
                sim_matrix[u][i] = 0
            else:
                sim_matrix[u][i] = count / math.sqrt(item_popular[u] * item_popular[i])
    for u, i in sim_matrix.items():
        sim_matrix.update({u: sorted(i.items(), key=itemgetter(1), reverse=True)})
    ll = []
    os.system('pause')
    t2 = time.process_time()
    print("物品相似度矩阵完毕")
    print(t2 - t1)


def recommend():
    for user in test_dataset:
        list1 = []
        try:
            t1 = time.process_time()
            rank = {}
            items = test_dataset[user]
            for item, rating in items.items():
                try:
                    for related_movie, w in sim_matrix[item][:sim_num]:
                        if related_movie in items:
                            continue
                        rank.setdefault(related_movie, 0)
                        rank[related_movie] += w * float(rating)
                except:
                    continue
            rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[:rec_num]
            list1 = [x[0] for x in rank]
            # 在加权时需要改
            if len(list1) < 30:
                for item in test_dataset[user]:
                    cata = attr.loc[attr['item_id'] == item]['cate_id'].values
                    for c in cata:
                        len_cata = len(item_cate[c])
                        for i in range(0, int(len_cata/2)):
                            ran1 = random.randint(0, len_cata-1)
                            while item_id[ran1] in list1:
                                ran1 = random.randint(0, len_cata-1)
                            list1.append(item_cate[c][ran1])
                            if len(list1) == 30:
                                break
                    if len(list1) == 30:
                        break
            if len(list1) < 30:
                items_ = items_cnts.copy()
                length = 30 - len(list1)
                for i in range(0, length):
                    it = items_.pop(0)
                    while it in list1:
                        it = items_.pop(0)
                    list1.append(it)
            result.setdefault(user, list1)
            t2 = time.process_time()
            print(t2-t1)
        except Exception as e:
            print(e,)
            for item in test_dataset[user]:
                cata = attr.loc[attr['item_id'] == item]['cate_id'].values
                for c in cata:
                    len_cata = len(item_cate[c])
                    for i in range(0, int(len_cata / 2)):
                        ran1 = random.randint(0, len_cata-1)
                        while item_id[ran1] in list1:
                            ran1 = random.randint(0, len_cata-1)
                        list1.append(item_cate[c][ran1])
                        if len(list1) == 30:
                            break
                if len(list1) == 30:
                    break
            if len(list1) < 30:
                items_ = items_cnts.copy()
                length = 30 - len(list1)
                for i in range(0, length):
                    it = items_.pop(0)
                    while it in list1:
                        it = items_.pop(0)
                    list1.append(it)
            result.setdefault(user, list1)
            print(user)


if __name__ == '__main__':
    t1 = time.process_time()
    data_processing()
    recommend()
    df = pd.DataFrame(result).T
    df.to_csv(r'F:\res1.csv')
    print(df)
    t2 = time.process_time()
    print(t2 - t1)
