import pandas as pd
from sklearn import model_selection
import config
import numpy as np

def create_folds(data,target_type,k_fold):
    '''
    :param data: a dataframe
    :param target_type: a string, 'c' for categorical target 'd' for continuous target
    if 'c' then data must have a target column

    增加了kfold列，将原始样本分为6份
    ''' 
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    # split and concat with space
    # 这里的分词简单了点
    print(df.head(5))
    data['news_keywords_split'] = data['news_keywords'].map(lambda x:' '.join(x.split(',')) if x >'' else '','ignore')
    data['news_title_split'] = data['news_title'].map(lambda x:' '.join([i for i in x]) if x >'' else '','ignore')

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=k_fold)

    if target_type == 'd':
        # calculate the number of bins by Sturge's rule
        # I take the floor of the value, you can also
        # just round it
        num_bins = int(np.floor(1 + np.log2(len(data))))
        # bin targets
        data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
        )



        # fill the new kfold column
        # note that, instead of targets, we use bins!
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
            data.loc[v_, 'kfold'] = f

        # drop the bins column
        data = data.drop("bins", axis=1)
    if target_type == 'c':
        # fill the new kfold column
        # note that, instead of targets, we use bins!
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.target.values)):
            data.loc[v_, 'kfold'] = f        
    # return dataframe with folds
    return data
if __name__ == "__main__":
    df = pd.read_csv(config.RAW_DATA,sep='_!_',names=['news_id','cate_code','cate_name','news_title','news_keywords'])
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    
    df['target']=le.fit_transform(df['cate_code'])
    print('原始数据的缺失情况')
    print([(i,df.isna()[i].sum()) for i in df.columns])
    # create folds
    # 由于readme中的样本划分是15%的test，因此划分为6份
    df = create_folds(df,'c',6)
    # 数据输出
    # df3.to_csv('E:\\data\\xxxx.csv',index=False,header= 0,sep='|', encoding="utf-8", quoting=csv.QUOTE_NONE,escapechar='|')

    df.to_csv(config.K_FOLD_DATA,index=False,sep='|', encoding="utf-8",escapechar='\\')
    print(f'{config.K_FOLD_DATA}输出完成')