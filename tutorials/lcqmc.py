import pandas as pd
from transformers_sklearn import BERTologyClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    ## 1. preparing X,y
    train_df = pd.read_csv('datasets/lcqmc/train.tsv',sep='\t',names=['s1','s2','label'])
    X_train = pd.concat([train_df['s1'],train_df['s2']],axis=1)
    y_train = train_df['label']

    dev_df = pd.read_csv('datasets/lcqmc/dev.tsv', sep='\t', names=['s1', 's2','label'])
    X_dev = pd.concat([dev_df['s1'], dev_df['s2']], axis=1)
    y_dev = dev_df['label']

    test_df = pd.read_csv('datasets/lcqmc/test.tsv', sep='\t', names=['s1', 's2', 'label'])
    X_test = pd.concat([test_df['s1'], test_df['s2']], axis=1)
    y_test = test_df['label']

    ## 2. customize the model
    cls = BERTologyClassifier(
        model_type='bert',
        model_name_or_path='bert-base-chinese',
        data_dir='ts_data/lcqmc',
        output_dir='results/lcqmc',
        num_train_epochs=3,
        learning_rate=5e-5
    )
    #
    ## 3. fit
    cls.fit(X_train,y_train)
    #
    ## 4. score
    report = cls.score(X_dev,y_dev)
    with open('lcqmc.txt','w',encoding='utf8') as f:
        f.write(report)

    ## 5. predict
    y_pred = cls.predict(X_test)
    test_report = classification_report(y_test,y_pred,digits=4)
    with open('lcqmc_test.txt','w',encoding='utf8') as f:
        f.write(test_report)


