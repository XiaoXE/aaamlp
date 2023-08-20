import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
from sklearn import metrics
import config

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]

    *[]:作用是：将列表解开成几个独立的参数，传入函数。类似的运算符还有两个星号(**)，是将字典解开成独立的元素作为形参。
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

def run(df,fold,model_name):
    '''
    :param df:a Dataframe excluding test data, with a column x_string(concat title with key words),a column target
    :param fold:a int which fold to be the validation data
    :param model_name:a string which model to use,i.e. lr or deep learning model
    '''
    valid_df = df[df['kfold']==fold].reset_index(drop=True).copy()
    train_df = df[df['kfold']!=fold].reset_index(drop=True).copy()
    print(f'{len(train_df)} train senetences')
    print(f'{len(valid_df)} validation senetences')

    print("Fitting tokenizer")
    # we use tf.keras for tokenization
    # you can use your own tokenizer and then you can 
    # get rid of tensorflow
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.x_string.values.tolist())
    # convert training data to sequences
    # for example : "bad movie" gets converted to
    # [24, 27] where 24 is the index for bad and 27 is the
    # index for movie
    xtrain = tokenizer.texts_to_sequences(train_df.x_string.values)
    # similarly convert validation data to
    # sequences
    xvalid = tokenizer.texts_to_sequences(valid_df.x_string.values)

    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, xtrain)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, xvalid)), dtype=int)))
    
    print('Min train sequence length: {}'.format(
        min(list(map(len, xtrain)))))
    print('Min test sequence length: {}'.format(
        min(list(map(len, xvalid)))))

    print('Max train sequence length: {}'.format(
        max(list(map(len, xtrain)))))
    print('Max test sequence length: {}'.format(
        max(list(map(len, xvalid)))))        
    

    if model_name == 'lr':
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfv = TfidfVectorizer(analyzer='char', token_pattern=None,ngram_range=(1,config.N_GRAM))
        tfv.fit(train_df.x_string.values)
        xtrain = tfv.transform(train_df.x_string)
        xvalid = tfv.transform(valid_df.x_string)

        # print(f'{train_df.x_string[:5]}')
        # print(f'{xtrain[:5]}')
        from sklearn import linear_model
        model = linear_model.LogisticRegression(n_jobs=-1,max_iter=200,solver='saga')
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.target)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xvalid)
        preds_prob = model.predict_proba(xvalid)
        # calculate accuracy
        accuracy = metrics.accuracy_score(valid_df.target, preds)
        auc = metrics.roc_auc_score(valid_df.target,preds_prob,average='macro',multi_class='ovr')
        # f1 = metrics.f1_score(valid_df.target, preds)

        print(f"Fold: {fold}")
        print(f"Accuracy = {accuracy}")
        print(f"AUC = {auc}")
        print("")
    if model_name == 'fasttext':

        ngram_range = config.N_GRAM
        max_features = config.MAX_FEA
        maxlen = config.MAX_LEN
        batch_size = config.BATCH_SIZE
        embedding_dims = 50
        epochs = 5 

        if config.N_GRAM > 1:
            print('Adding {}-gram features'.format(ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in xtrain:
                for i in range(2, ngram_range + 1):
                    set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting xtrain and xvalid with n-grams features
            xtrain = add_ngram(xtrain, token_indice, ngram_range)
            xvalid = add_ngram(xvalid, token_indice, ngram_range)
            print('Average train sequence length: {}'.format(
                np.mean(list(map(len, xtrain)), dtype=int)))
            print('Average test sequence length: {}'.format(
                np.mean(list(map(len, xvalid)), dtype=int)))

        print('Pad sequences (samples x time)')
        # zero pad the training sequences given the maximum length
        # this padding is done on left hand side
        # if sequence is > MAX_LEN, it is truncated on left hand side too
        xtrain = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain, maxlen=config.MAX_LEN
        )
        # # zero pad the validation sequences
        xvalid = tf.keras.preprocessing.sequence.pad_sequences(
        xvalid, maxlen=config.MAX_LEN
        )    
        print('xtrain shape:', xtrain.shape)
        print('xvalid shape:', xvalid.shape)

        print('Build model...')
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(train_df.target.nunique(), activation='sigmoid'))

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        model.fit(xtrain, train_df.target,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(xvalid, valid_df.target))
    if model_name == 'textcnn':
        pass


if __name__ == "__main__":
    # read fold data
    # import tensorflow as tf
    # print(tf.__version__)
    # print(tf.test.is_gpu_available())  # 查看cuda、TensorFlow_GPU和cudnn(选择下载，cuda对深度学习的补充)版本是否对应
    # print(tf.config.list_physical_devices('GPU'))

    import os
 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # =右边"0,1",代表使用标号为0,和1的GPU

    df = pd.read_csv(config.K_FOLD_DATA,sep='|', encoding="utf-8",escapechar='\\')
    print(f'read data {len(df)} rows, train data should be {len(df)/3*2}')
    df['news_keywords_split'].fillna('',inplace=True)
    df['x_string']=df['news_title_split'].str.cat(df['news_keywords_split'],sep=' ')
    # print(f'{df.head(5)}')
    print(f'{df.kfold.unique()}')
    # 取出一份作为test
    df_test = df[df.kfold==config.TEST_FOLD].copy()
    print(f'len of df_test is {len(df_test)}')
    # df.drop(df['kfold']=='5',inplace=True)   
    print('run model')

    # run(df[df.kfold!=config.TEST_FOLD],4,'lr')
    tmp_df = df[df.kfold!=config.TEST_FOLD].copy()
    print(f'len of tmpdf {len(tmp_df)}')

     # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
     # currently, we only need fold
    parser.add_argument(
    "--fold",
    type=int
    )
    parser.add_argument(
    "--model",
    type=str
    )    
    # read the arguments from the command line
    args = parser.parse_args()
    # run(tmp_df,4,'fasttext')
    # run(tmp_df,3,'fasttext')    
    # run(tmp_df,4,'lr')
    # run(tmp_df,3,'lr')

    run(tmp_df,args.fold,args.model)

