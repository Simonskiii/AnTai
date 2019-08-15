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
sim_num = 10  # 总共算100个
rec_num = 30  # 挑30个
sim_matrix = {}  # 相似度矩阵
item_user = {}  # key：item value：user[]
item_count = pd.DataFrame()  # 计算每个商品买了几次
items_cnts = []  # 根据商品购买次数对商品进行排序
result = {}  # 结果
attr = pd.DataFrame()  # 商品属性对应表


def data_processing():
    global attr
    global item_count
    global dataset
    global test_dataset
    global item_id
    global items_cnts
    global r
    t1 = time.process_time()
    attr = pd.read_csv('..\\data\\Antai_AE_round1_item_attr_20190626.csv')  # 商品属性对应
    attr = attr.sort_values(by=['cate_id', 'item_id'], ascending=(True, True)).reset_index(drop=True)
    attr = attr[['item_id', 'cate_id']]
    train = pd.read_csv('..\\data\\Antai_AE_round1_train_20190626.csv')
    r = pd.read_csv('..\\data\\t_result.csv')
    train = train.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    test = pd.read_csv('..\\data\\t_test.csv')
    test = test.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    train = pd.concat([train.loc[train['buyer_country_id'] == 'yy'], test])
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
    # print("用户相似度矩阵完毕")
    print(t2 - t1)


def recommend():

    length = [6,7,8,9]
    with open('../2.txt', 'w', encoding='utf-8') as f:
        for leng in length:
            a = 0
            b = 0
            c1 = 0
            c2 = 0
            d = 0
            for user in test_dataset:
                l = []
                last = r.loc[r.buyer_admin_id == user]['item_id'].values[0]
                # for i in test_dataset[user]:
                #     try:
                #         l.append(i[0])
                #         if len(l) == 30:
                #             result.setdefault(user, l)
                #             break
                #     except:
                #         break
                for i in range(0, leng):
                    try:
                        l.append(list(test_dataset[user])[i][0])
                        if list(test_dataset[user])[i][0] == last:
                            a += 1
                    except:
                        break
                # print(1)
                # print(l)
                # os.system('pause')
                if len(l) == 30:
                    result.setdefault(user, l)
                else:
                    # rank = {}
                    # items = dataset[user]
                    try:
                        # for v, wuv in sim_matrix[user][:sim_num]:
                        #     try:
                        #         for item in dataset[v]:
                        #             if item in items:
                        #                 continue
                        #             rank.setdefault(item, 0)
                        #             if rank[item] < wuv:
                        #                 rank[item] = wuv
                        #     except Exception as e:
                        #         print(e)
                        #         os.system('pause')
                        #         continue
                        # rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[:rec_num]
                        # for x in rank:
                        #     if len(l) == 30:
                        #         break
                        #     l.append(x[0])
                        #     if x[0] == last:
                        #         b += 1
                        # print(2)
                        # print(l)
                        # os.system('pause')
                        if len(l) < 30:
                            c1 += 1
                            items_ = items_cnts.copy()
                            length = 30 - len(l)
                            for i in range(0, length):
                                it = items_.pop(0)
                                while it in l:
                                    it = items_.pop(0)
                                l.append(it)
                                if it == last:
                                    c2 += 1
                    # print(3)
                    # print(l)
                    # os.system('pause')
                    except Exception as e:
                        d += 1
                        items_ = items_cnts.copy()
                        length = 30 - len(l)
                        c1 += 1
                        for i in range(0, length):
                            it = items_.pop(0)
                            while it in l:
                                it = items_.pop(0)
                            l.append(it)
                            if it == last:
                                c2 += 1
                    result.setdefault(user, l)
            f.write(str(leng))
            f.write('\n')
            f.write('历史记录命中：')
            f.write(str(a))
            f.write('\n')
            f.write('userCF命中：')
            f.write(str(b))
            f.write('\n')
            f.write('热度用了几次：')
            f.write(str(c1))
            f.write('\n')
            f.write('热度命中：')
            f.write(str(c2))
            f.write('\n')
            f.write('总命中：')
            f.write(str(a+b+c2))
            f.write('\n')
            f.write('出错：')
            f.write(str(d))
            f.write('\n')
            f.write('********************')
            f.write('\n')


if __name__ == '__main__':
    t1 = time.process_time()
    data_processing()
    recommend()
    # df = pd.DataFrame(result).T
    # df.to_csv('res1.csv')
    # print(df)
    t2 = time.process_time()
    print(t2 - t1)
