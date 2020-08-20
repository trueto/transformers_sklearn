import pandas as pd
from transformers_sklearn import BERTologyClassifier

if __name__ == '__main__':
    ## 1. preparing X,y
    train_df = pd.read_csv("datasets/Trial/trail_classify_train.csv")
    dev_df = pd.read_csv("datasets/Trial/trail_classify_dev.csv")
    test_df = pd.read_csv("datasets/Trial/trail_classify_test.csv")

    X_train = train_df['text'] + dev_df['text']
    y_train = train_df['label'] + dev_df['label']

    X_test, y_test = test_df['text'], test_df['label']

    ## 2. customize the model
    cls = BERTologyClassifier(
        model_type='bert',
        model_name_or_path='/home/yfh/bertology_models/bert-base-chinese',
        data_dir='ts_data/trial',
        output_dir='results/trial',
        num_train_epochs=3,
        learning_rate=5e-5,
        overwrite_output_dir=True
    )

    ## 3. fit
    cls.fit(X_train, y_train)

    ## 4. score
    report = cls.score(X_test, y_test)

    with open('trail.txt', 'w', encoding='utf8') as f:
        f.write(report)
