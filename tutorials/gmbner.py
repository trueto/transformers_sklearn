import pandas as pd
from sklearn.model_selection import train_test_split
from transformers_sklearn import BERTologyNERClassifer

if __name__ == '__main__':

    data_df = pd.read_csv('datasets/gmbner/ner_dataset.csv')
    data_df.fillna(method="ffill",inplace=True)
    value_counts = data_df['Tag'].value_counts()
    labels = list(value_counts.to_dict().keys())

    ## 1. preparing data
    X = []
    y = []
    for label, batch_df in data_df.groupby(by='Sentence #',sort=False):
        words = batch_df['Word'].tolist()
        labels = batch_df['Tag'].tolist()
        assert len(words) == len(labels)
        X.append(words)
        y.append(labels)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=520)

    ## 2. customize model
    ner = BERTologyNERClassifer(
        labels=labels,
        model_type='bert',
        model_name_or_path='bert-base-cased',
        data_dir='ts_data/gmbner',
        output_dir='results/gmbner',
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=50,
        save_steps=50,
        overwrite_output_dir=True
    )

    # # # ## 3. fit
    ner.fit(X_train, y_train)
    # #
    # # # ## 4. score
    report = ner.score(X_test, y_test)
    with open('gmbner.txt', 'w', encoding='utf8') as f:
        f.write(report)