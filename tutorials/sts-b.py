import pandas as pd
from transformers_sklearn import BERTologyRegressor

if __name__ == '__main__':
    ## 1. preparing X,y
    train_df = pd.read_csv('datasets/sts-b/train.tsv',sep='\t')
    X_train = pd.concat([train_df['sentence1'],train_df['sentence2']],axis=1)
    y_train = train_df['score']

    dev_df = pd.read_csv('datasets/sts-b/dev.tsv', sep='\t')
    X_dev = pd.concat([dev_df['sentence1'], dev_df['sentence2']], axis=1)
    y_dev = dev_df['score']

    test_df = pd.read_csv('datasets/sts-b/test.tsv', sep='\t')
    X_test = pd.concat([test_df['sentence1'], test_df['sentence2']], axis=1)

    ## 2. customize the model
    cls = BERTologyRegressor(
        model_type='bert',
        model_name_or_path='bert-base-cased',
        data_dir='ts_data/sts-b',
        output_dir='results/sts-b',
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=100,
        save_steps=100,
        overwrite_output_dir=True
    )
    #
    ## 3. fit
    cls.fit(X_train,y_train)
    #
    ## 4. score
    report = cls.score(X_dev,y_dev)
    with open('sts-b.txt','w',encoding='utf8') as f:
        f.write(str(report))

    ## 5. predict
    y_pred = cls.predict(X_test)

    temp_df = X_test.copy()
    temp_df['score'] = y_pred
    temp_df.to_csv('sts-b.tsv',index=False)


