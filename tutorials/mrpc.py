import pandas as pd
from transformers_sklearn import BERTologyClassifier

if __name__ == '__main__':
    ## 1. preparing X,y
    train_df = pd.read_csv('datasets/mrpc/train.txt',sep='\t',names=['label','id1','id2','s1','s2'])
    train_df.dropna(inplace=True)
    X_train = pd.concat([train_df['s1'],train_df['s2']],axis=1)
    y_train = train_df['label']

    test_df = pd.read_csv('datasets/mrpc/test.txt', sep='\t', names=['label', 'id1', 'id2', 's1', 's2'])
    test_df.dropna(inplace=True)
    X_test = pd.concat([test_df['s1'], test_df['s2']], axis=1)
    y_test = test_df['label']

    ## 2. customize the model
    cls = BERTologyClassifier(
        model_type='bert',
        model_name_or_path='bert-base-cased',
        data_dir='ts_data/mrpc',
        output_dir='results/mrpc',
        num_train_epochs=3,
        learning_rate=5e-5,
        overwrite_output_dir=True
    )

    ## 3. fit
    cls.fit(X_train,y_train)

    ## 4. score
    report = cls.score(X_test,y_test)

    with open('mrpc.txt','w',encoding='utf8') as f:
        f.write(report)


