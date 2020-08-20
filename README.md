# transformers_sklearn
A Fitting, Scoring and Predicting Toolkit for Transformer Series Models.

`transformers_sklearn` aims at bringing `Transformer` series models to non-specialists.

## Main requirements
- pandas https://pandas.pydata.org/
- pytorch https://pytorch.org/
- transformers https://github.com/huggingface/transformers
- scikit-learn https://scikit-learn.org/stable/

## Architecture
`transformers_sklearn` wrap the powerful APIs of `transformers` into three
`class`. They are `BERTologyClassifier` for classification task, `BERTologyNERClassifier`
for name entity recognition(NER) task and `BERTologyRegressor` for regression task.
Within each `class`, there are three methods, which are `fit` for fine-tuning
the pre-trained models downloaded from community, `score` for scoring the performance of
fine-tuned model, and `predict` for predicting the labels of given datasets.

`fit` and `score` methods accept two parameters, `X` and `y`. The type of `X` and `y`
could be `list`, `ndarray` and `DataFrame`. `predict` only need `X`.

## Install
```
git clone https://github.com/trueto/transformers_sklearn
cd transformers_sklearn
pip install .
```

## Tutorials

### Classification tasks
#### English
Fine-tuning and scoring BERT model on [MRPC](https://github.com/binhaowang/glue_data) corpus
The whole code is as following:
```python
import pandas as pd
from transformers_sklearn import BERTologyClassifier

if __name__ == '__main__':
    ## 1. preparing X,y
    train_df = pd.read_csv('datasets/mrpc/train.txt',sep='\t',names=['label','id1','id2','s1','s2'])
    X_train = pd.concat([train_df['s1'],train_df['s2']],axis=1)
    y_train = train_df['label']

    test_df = pd.read_csv('datasets/mrpc/test.txt', sep='\t', names=['label', 'id1', 'id2', 's1', 's2'])
    X_test = pd.concat([test_df['s1'], test_df['s2']], axis=1)
    y_test = test_df['label']

    ## 2. customize the model
    cls = BERTologyClassifier(
        model_type='bert',
        model_name_or_path='bert-base-cased',
        data_dir='ts_data/mrpc',
        output_dir='results/mrpc',
        num_train_epochs=3,
        learning_rate=5e-5
    )

    ## 3. fit
    cls.fit(X_train,y_train)

    ## 4. score
    report = cls.score(X_test,y_test)

    with open('mrpc.txt','w',encoding='utf8') as f:
        f.write(report)
```
Running code above, the following result will be returned:

```
***** Eval results 50 *****acc = 0.7360406091370558
acc_and_f1 = 0.7810637828293975
f1 = 0.8260869565217391

***** Eval results 100 *****acc = 0.6802030456852792
acc_and_f1 = 0.707748581666169
f1 = 0.7352941176470589

***** Eval results 150 *****acc = 0.8121827411167513
acc_and_f1 = 0.8418552594472646
f1 = 0.8715277777777778

***** Eval results 200 *****acc = 0.817258883248731
acc_and_f1 = 0.8452491599342247
f1 = 0.8732394366197184

***** Eval results 250 *****acc = 0.8096446700507615
acc_and_f1 = 0.8367642588003353
f1 = 0.8638838475499092

***** Eval results 300 *****acc = 0.8121827411167513
acc_and_f1 = 0.840488533678943
f1 = 0.8687943262411348

```
#### Chinese
Fine-tuning and scoring BERT model on [LCQMC](http://106.13.187.75:8003/dataSet) corpus
The whole code is as following:
```python
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
```
Running code above, the following result will be returned:
```
 precision    recall  f1-score   support

           0     0.8919    0.8777    0.8848      4400
           1     0.8797    0.8937    0.8866      4402

    accuracy                         0.8857      8802
   macro avg     0.8858    0.8857    0.8857      8802
weighted avg     0.8858    0.8857    0.8857      8802
```

### NER tasks
#### English
Fine-tuning and scoring BERT model on [GMBNER]() corpus
The whole code is as following:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers_sklearn import BERTologyNERClassifer

if __name__ == '__main__':

    data_df = pd.read_csv('datasets/gmbner/ner_dataset.csv',encoding="utf8")
    data_df.fillna(method="ffill",inplace=True)
    value_counts = data_df['Tag'].value_counts()
    label_list = list(value_counts.to_dict().keys())

    # ## 1. preparing datasets
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
        labels=label_list,
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
    #
   ## 3. fit
    ner.fit(X_train, y_train)
    # # # #
    ## 4. score
    report = ner.score(X_test, y_test)
    with open('gmbner.txt', 'w', encoding='utf8') as f:
        f.write(report)
```
#### Chinese
Fine-tuning and scoring BERT model on [MSRANER](http://106.13.187.75:8003/dataSet) corpus
The whole code is as following:
```python
import pandas as pd
from transformers_sklearn import BERTologyNERClassifer

tag_dict = {
    'o': 'O',
    'nr': 'Per',
    'ns': 'Adr',
    'nt': 'Org'
}
def get_X_y(file_path):
    words_list = []
    labels_list = []
    with open(file_path,'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            words = []
            labels = []
            for sequence in line.split(' '):
                tokens = sequence.split('/')[0]
                tag = sequence.split('/')[-1]
                tlen = len(tokens)
                if tag == 'o':
                    labels = labels + ['O'] * tlen
                else:
                    labels = labels + ['B-' + tag_dict[tag]] + (tlen - 1) * ['I-' + tag_dict[tag]]
            assert len(words) == len(labels)
            words_list.append(words)
            labels_list.append(labels)

    return words_list,labels_list

if __name__ == '__main__':
    ## 1. preparing X,y
    X_train, y_train = get_X_y('datasets/msraner/train.txt')

    X_test, y_test = get_X_y('datasets/msraner/test.txt')

    ## 2. customize model
    ner = BERTologyNERClassifer(
        model_type='bert',
        model_name_or_path='bert-base-chinese',
        data_dir='ts_data/msraner',
        output_dir='results/msraner',
        num_train_epochs=3,
        learning_rate=5e-5
    )

    ## 3. fit
    ner.fit(X_train, y_train)

    ## 4. score
    report = ner.score(X_test,y_test)
    with open('msraner.txt','w',encoding='utf8') as f:
        f.write(report)
```

### Regression task
Fine-tuning and scoring BERT model on [STS-B](https://github.com/binhaowang/glue_data) corpus
The whole code is as following:
```python
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
        learning_rate=5e-5
    )
    #
    ## 3. fit
    cls.fit(X_train,y_train)
    #
    ## 4. score
    report = cls.score(X_dev,y_dev)
    with open('sts-b.txt','w',encoding='utf8') as f:
        f.write(report)

    ## 5. predict
    y_pred = cls.predict(X_test)

    temp_df = X_test.copy()
    temp_df['score'] = y_pred
    temp_df.to_csv('sts-b.tsv',index=False)
```
