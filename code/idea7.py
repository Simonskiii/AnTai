# 历史记录，用usercf补
import pandas as pd
import math
from operator import itemgetter
import time
import os
item_id = []  # 所有的item
item_cate = {}  # 每个种类对应的所有商品
dataset = {}  # 训练数据集
test_dataset = {}  # 测试数据集
item_content = {}
sim_num = 100  # 总共算100个
rec_num = 30  # 挑30个
sim_matrix = {}  # 相似度矩阵
item_user = {}  # key：item value：user[]
item_count = pd.DataFrame()  # 计算每个商品买了几次
items_cnts = []  # 根据商品购买次数对商品进行排序
result = {}  # 结果
attr = pd.DataFrame()  # 商品属性对应表
train = pd.DataFrame()


def data_processing():
    global attr
    global item_count
    global dataset
    global test_dataset
    global item_id
    global items_cnts
    global train
    global item_content
    t1 = time.process_time()
    attr = pd.read_csv('..\\data\\Antai_AE_round1_item_attr_20190626.csv')  # 商品属性对应
    attr = attr.sort_values(by=['cate_id', 'item_id'], ascending=(True, True)).reset_index(drop=True)
    train = pd.read_csv('..\\data\\Antai_AE_round1_train_20190626.csv')
    train = train.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    test = pd.read_csv('..\\data\\Antai_AE_round1_test_20190626.csv')
    test = test.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    train = pd.concat([train.loc[train['buyer_country_id'] == 'yy'], test])
    train = pd.merge(train, attr, how='left')
    test = pd.merge(test, attr, how='left')
    train = train.fillna(int(0))
    test = test.fillna(int(0))
    # 购买频率表
    temp = train.loc[train.buyer_country_id == 'yy']
    temp = temp.drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')
    item_count = temp.groupby(['item_id']).size().reset_index()
    item_count.columns = ['item_id', 'cnts']
    item_count = item_count.sort_values('cnts', ascending=False)
    items_cnts = item_count['item_id'].values.tolist()
    # 物品种类表
    for row in attr.itertuples(index=True, name='Pandas'):
        cate = getattr(row, "item_id")
        item_cate.setdefault(cate, [])
        item_cate[cate].append(getattr(row, "item_id"))
    # 训练集用户购买物品字典
    for row in train.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        item = getattr(row, "item_id")
        dataset.setdefault(admin, {})
        dataset[admin].setdefault(item, 0)
        dataset[admin][item] += 1
    for row in attr.itertuples(index=True, name='Pandas'):
        item = getattr(row, "item_id")
        cate = getattr(row, 'cate_id')
        price = getattr(row, 'item_price')
        store = getattr(row, "store_id")
        item_content.setdefault(item, [])
        item_content[item].append(cate)
        item_content[item].append(price)
        item_content[item].append(store)
    # 测试集用户购买物品字典
    for row in test.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        item = getattr(row, "item_id")
        test_dataset.setdefault(admin, {})
        test_dataset[admin].setdefault(item, 0)
        test_dataset[admin][item] += 1
    for user, items in dataset.items():
        for item in items:
            if item not in item_user:
                item_user[item] = set()
            item_user[item].add(user)
    print("物品-用户矩阵已完毕")
    for item, users in item_user.items():
        for u in users:
            for v in users:
                if u == v:
                    continue
                sim_matrix.setdefault(u, {})
                sim_matrix[u].setdefault(v, 0)
                sim_matrix[u][v] += 1
    print("用户用户矩阵已完毕")
    for u, related_users in sim_matrix.items():
        for i, count in related_users.items():
            sim_matrix[u][i] = count / math.sqrt(len(dataset[u]) * len(dataset[i]))
    for u, i in sim_matrix.items():
        sim_matrix.update({u:sorted(i.items(), key=itemgetter(1), reverse=True)})
    for u, i in test_dataset.items():
        test_dataset.update({u: sorted(i.items(), key=itemgetter(1), reverse=True)})
    t2 = time.process_time()

    print("用户相似度矩阵完毕")
    print(t2 - t1)


def recommend():
    for user in test_dataset:
        print(user)
        l = []
        stores = train.loc[train['buyer_admin_id'] == user]['store_id'].values.tolist()
        cates = train.loc[train['buyer_admin_id'] == user]['cate_id'].values.tolist()
        for i in range(0, 5):
            try:
                l.append(list(test_dataset[user])[i][0])
            except:
                break
        items_ = items_cnts.copy()
        if len(l) < 30:
            length = 5
            for i in range(0, length):
                it = items_.pop(0)
                while it in l:
                    it = items_.pop(0)
                l.append(it)

        if len(l) == 30:
            result.setdefault(user, l)
        else:
            rank = {}
            items = dataset[user]
            try:
                for v, wuv in sim_matrix[user][:sim_num]:
                        for item in dataset[v]:
                            try:
                                a = item_content[item][2]
                                c = item_content[item][0]
                                # c = float(c)
                                price = item_content[item][1]
                                # price = float(price)
                                if price < 10000:
                                    if item in items:
                                        continue
                                    rank.setdefault(item, 0)
                                    rank[item] += wuv
                                    if a in stores:
                                        rank[item] += 2.3
                                    if c in cates:
                                        rank[item] += 1.7

                            except Exception as e:
                                print(e)
                                continue
                rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[:rec_num]
                for x in rank:
                    if len(l) == 30:
                        break
                    l.append(x[0])
            except:
                length = 30 - len(l)
                for i in range(0, length):
                    it = items_.pop(0)
                    while it in l:
                        it = items_.pop(0)
                    l.append(it)
            if len(l) < 30:
                length = 30 - len(l)
                for i in range(0, length):
                    it = items_.pop(0)
                    while it in l:
                        it = items_.pop(0)
                    l.append(it)
            if len(l) > 30:
                length = len(l) - 30
                for i in range(0,length):
                    l.pop(len(l)-1)
            result.setdefault(user, l)
    print(result)


if __name__ == '__main__':
    t1 = time.process_time()
    data_processing()
    recommend()
    df = pd.DataFrame(result).T
    df.to_csv('userCF.csv',header=None)
    print(df)
    t2 = time.process_time()
    print(t2 - t1)

