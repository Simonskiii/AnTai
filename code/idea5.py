# item2vec
import pandas as pd
import math
from operator import itemgetter
from gensim.models import fasttext
from gensim.models import word2vec
import gensim
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from numba import jit
item_id = []  # 所有的item
item_cate = {}  # 每个种类对应的所有商品
dataset = {}  # 训练数据集
test_dataset = {}  # 测试数据集
sim_num = 5  # 总共算100个
rec_num = 30  # 挑30个
sim_matrix = {}  # 相似度矩阵
item_user = {}  # key：item value：user[]
item_count = pd.DataFrame()  # 计算每个商品买了几次
items_cnts = []  # 根据商品购买次数对商品进行排序
result = {}  # 结果
attr = pd.DataFrame()  # 商品属性对应表
lists = []
mod = {}
size = 1000
name = 'word2vec' + str(size) + '.model'
model = gensim.models.Word2Vec.load(name)


def data_processing():
    global attr
    global item_count
    global dataset
    global test_dataset
    global item_id
    global items_cnts
    t1 = time.process_time()
    train = pd.read_csv('..\\data\\Antai_AE_round1_train_20190626.csv')
    train = train.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    test = pd.read_csv('..\\data\\Antai_AE_round1_test_20190626.csv')
    test = test.fillna(int(0)).sort_values(by=['buyer_admin_id', 'irank'], ascending=(True, True)).reset_index(
        drop=True)
    item_id = list(test.groupby(['item_id']).count().index)
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
        item = str(getattr(row, "item_id"))
        dataset.setdefault(admin, {})
        dataset[admin].setdefault(item, 0)
        dataset[admin][item] += 1
    # 测试集用户购买物品字典
    for row in test.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        item = str(getattr(row, "item_id"))
        test_dataset.setdefault(admin, {})
        test_dataset[admin].setdefault(item, 0)
        test_dataset[admin][item] += 1
    for user, items in dataset.items():
        lists.append(list(items))
    t2 = time.process_time()
    print("数据处理完毕")
    print(t2 - t1)

@jit
def cal():
    global mod
    for item in item_id[:]:
        l = model.wv.most_similar(str(item), topn=sim_num)
        print('over')
        mod.setdefault(str(item), l)

def recommend():
    model = word2vec.Word2Vec(lists, sg=1, size=size, hs=0, iter=10, min_count=1, window=100)
    # model.save(name)
    # for i in lists[1]:
    #     print(model.wv.most_similar(i,topn=30))
    #     print('********************')
    # l = []  # 可视化
    # la = np.linalg
    # words = model.wv.index2word[:]
    # for word in words:
    #     l.append(model[word])
    # s, U, Vh = tf.linalg.svd(l, full_matrices=False)
    #
    # plt.axis([0, 0.08, -0.08, 0.1])  # 高维度的图
    # # # plt.axis([-1, 1, -1, 1])  #  二维图
    # plt.scatter(U[:, 0], U[:, 1], c='magenta', alpha=0.3)
    # plt.show()
    for user in test_dataset:
        list1 = []
        try:
            rank = {}
            items = test_dataset[user]
            for item, rating in items.items():
                try:
                    lis1 = mod[item]
                    for i in lis1:
                        rank.setdefault(i[0], 0)
                        if rank[i[0]] < i[1]:
                            rank[i[0]] = i[1]
                    # print(user)
                except Exception as e:
                    with open('../1.txt','w', encoding='utf-8') as f:
                        f.write('1e:')
                        f.write(e)
                        f.write("  ")
                        f.write(user)
                        f.write('\n')
                    continue
            rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[:rec_num]
            list1 = [x[0] for x in rank]
        except Exception as e:
            with open('../1.txt','w', encoding='utf-8') as f:
                f.write('2e:')
                f.write(e)
                f.write("  ")
                f.write(user)
                f.write('\n')
        if len(list1) < 30:
            items_ = items_cnts.copy()
            length = 30 - len(list1)
            for i in range(0, length):
                it = items_.pop(0)
                while it in list1:
                    it = items_.pop(0)
                list1.append(it)
        result.setdefault(user, list1)

if __name__ == '__main__':
    t1 = time.process_time()
    data_processing()
    cal()
    recommend()
    df = pd.DataFrame(result).T
    df.to_csv(r'F:\1132.csv')
    t2 = time.process_time()
    print(t2 - t1)
    os.system('shutdown -s -t 1')