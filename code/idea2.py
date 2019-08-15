# 历史纪录，cateCF，热度
import pandas as pd
from operator import itemgetter
import time
import os


item_id =[]   # items
item_cate = {}  # key：cate values：  of items
dataset = {}  # key:admin values: list of items the admin bought
test_dataset = {}  # key:admin values: list of items the admin bought
item_cate_cnt = {}
item_count = pd.DataFrame()
result = {}
attr = pd.DataFrame()
items_cnts = []
s_users = []
test = pd.DataFrame()


def get_preprocessing(df_):
    df = df_.copy()
    df['hour'] = df['create_order_time'].apply(lambda x: int(x[11:13]))
    df['day'] = df['create_order_time'].apply(lambda x: int(x[8:10]))
    df['month'] = df['create_order_time'].apply(lambda x: int(x[5:7]))
    df['year'] = df['create_order_time'].apply(lambda x: int(x[0:4]))
    df['date'] = (df['month'].values - 7) * 31 + df['day']
    return df




def data_processing():
    global attr
    global item_count
    global dataset
    global test_dataset
    global item_id
    global items_cnts
    global s_users
    global test
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
    train = pd.concat([train.loc[train['buyer_country_id'] == 'yy'], test])
    train = pd.merge(train, attr,how='left')
    test = pd.merge(test, attr,how='left')
    train = train.fillna(int(0))
    test = test.fillna(int(0))
    # print(train)
    # os.system('pause')
    item_id = attr.loc[:, 'item_id'].values
    # 购买频率表
    temp = train.loc[train.buyer_country_id == 'yy']
    temp = temp.drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')
    temp1 = pd.merge(temp, attr)
    item_count = temp.groupby(['item_id']).size().reset_index()
    item_count.columns = ['item_id', 'cnts']
    item_count = item_count.sort_values('cnts', ascending=False)
    item_count = pd.merge(item_count,attr)
    attr_count = temp1.groupby(['cate_id']).size().reset_index()
    attr_count.columns = ['cate_id', 'cnts']
    attr_count = attr_count.sort_values('cnts', ascending=False)
    items_cnts = item_count['item_id'].values.tolist()
    # 物品种类表
    for row in attr.itertuples(index=True, name='Pandas'):
        cate = getattr(row, "cate_id")
        item = getattr(row, "item_id")
        item_cate.setdefault(cate, [])
        item_cate[cate].append(item)
    # 训练集用户购买物品字典
    for row in train.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        cate = getattr(row, "cate_id")
        dataset.setdefault(admin, {})
        dataset[admin].setdefault(cate, 0)
        dataset[admin][cate] += 1
    for u, i in dataset.items():
        dataset.update({u: sorted(i.items(), key=itemgetter(1), reverse=True)})
    for row in item_count[:230167].itertuples(index=True, name='Pandas'):
        cate = getattr(row, "cate_id")
        item = getattr(row, "item_id")
        item_cate_cnt.setdefault(cate, [])
        item_cate_cnt[cate].append(item)
    # 测试集用户购买物品字典
    for row in test.itertuples(index=True, name='Pandas'):
        admin = getattr(row, "buyer_admin_id")
        cate = getattr(row, "cate_id")
        test_dataset.setdefault(admin, {})
        test_dataset[admin].setdefault(cate, 0)
        test_dataset[admin][cate] += 1
    for key in test_dataset:
        for k in test_dataset[key]:
            if k == 0:
                s_users.append(key)
                break
    for u, i in test_dataset.items():
        test_dataset.update({u: sorted(i.items(), key=itemgetter(1), reverse=True)})
    t2 = time.process_time()
    print("处理完毕")
    print(t2 - t1)


def recommend():
    for user in test_dataset:
        lis =[]
        cates = {}
        for i in range(0, 9):
            try:
                lis.append(int(list(test_dataset[user])[i][0]))
            except:
                break
        for i in test_dataset[user]:
            cates.setdefault(i[0],0)
            cates[i[0]] += 1
        cates = sorted(cates.items(), key=itemgetter(1), reverse=True)
        if user in s_users:
            itemss = test.loc[test['buyer_admin_id'] == user].loc[test['cate_id'] == 0]['item_id'].values
            itemss = set(itemss)
            for i in itemss:
                lis.append(i)
                if len(lis) == 30:
                    break
        cL = len(cates)
        length = 5
        if cL > 6:
            length = int(20 / cL)
        for key in cates:
            try:
                if len(lis) == 30:
                    break
                else:
                    items = item_cate_cnt[int(key[0])]
                    if len(items) >= 5:
                        for i in range(0, length):
                            if len(lis) == 30:
                                break
                            lis.append(items[i])
                    else:
                        for item in items:
                            if len(lis) == 30:
                                break
                            lis.append(item)
            except Exception as e:
                continue
        if len(lis) < 30:
            items_ = items_cnts.copy()
            length = 30 - len(lis)
            for i in range(0, length):
                it = items_.pop(0)
                while it in lis:
                    it = items_.pop(0)
                lis.append(it)
        result.setdefault(user, lis)



if __name__ == '__main__':
    t1 = time.process_time()
    data_processing()
    recommend()
    df = pd.DataFrame(result).T
    df.to_csv(r'F:\91.csv')

    # df = pd.DataFrame(result).T
    # df.to_csv(r'F:\res1.csv')
    # print(df)
    # t2 = time.process_time()
    # print(t2 - t1)
