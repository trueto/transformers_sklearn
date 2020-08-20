from transformers_sklearn import BERTologyNERClassifer

def get_X_y(input_file):

    X = []
    y = []
    tag = input_file.split('/')[-2]
    with open(input_file, 'r') as reader:
        for doc in reader.read().split('\n\n'):
            X_ = []
            y_ = []
            for line in doc.split('\n'):
                tmp_list = line.split('\t')
                label_type = tmp_list[-1]
                label = 'O' if label_type == 'O' else tag
                X_.append(tmp_list[0])
                y_.append(label)
            assert len(X_) == len(y_)
            X.append(X_)
            y.append(y_)

    assert len(X) == len(y)
    return X,y


def merge_y(input_file1, input_file2):
    X, y1 = get_X_y(input_file1)
    _, y2 = get_X_y(input_file2)

    assert len(y1) == len(y2)
    y = []
    for y_1, y_2 in zip(y1, y2):
        assert len(y_1) == len(y_2)
        y_ = ['O'] * len(y_1)
        for i, (label1, label2) in enumerate(zip(y_1, y_2)):
            if label1 == "O":
                y_[i] = label2
            if label2 == "O":
                y_[i] = label1
        y.append(y_)

    assert len(X) == len(y)
    return X, y

if __name__ == '__main__':

    ## 1. preparing X,y
    X_train, y_train = merge_y("datasets/BC5CDR/disease/train.tsv", "datasets/BC5CDR/chem/train.tsv")
    X_dev, y_dev = merge_y("datasets/BC5CDR/disease/devel.tsv", "datasets/BC5CDR/chem/devel.tsv")
    X_train = X_train + X_dev
    y_train = y_train + y_dev

    X_test, y_test = merge_y("datasets/BC5CDR/disease/test.tsv", "datasets/BC5CDR/chem/test.tsv")

    labels = ['O', 'disease', 'chem']
    ## 2. customize model
    ner = BERTologyNERClassifer(
        labels=labels,
        model_type='bert',
        model_name_or_path='/home/yfh/bertology_models/bert-base-cased',
        data_dir='ts_data/BC5CDR',
        output_dir='results/BC5CDR',
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=50,
        save_steps=50,
        overwrite_output_dir=True
    )
    # #
    # # # ## 3. fit
    ner.fit(X_train, y_train)
    # #
    # # # ## 4. score
    report = ner.score(X_test, y_test)
    with open('BC5CDR.txt', 'w', encoding='utf8') as f:
        f.write(report)