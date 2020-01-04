import numpy as np
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
                words = words + list(tokens)
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

    labels = ['O', 'B-Per','I-Per','B-Adr','I-Adr','B-Org','I-Org']
    ## 2. customize model
    ner = BERTologyNERClassifer(
        labels=labels,
        model_type='bert',
        model_name_or_path='bert-base-chinese',
        data_dir='ts_data/msraner',
        output_dir='results/msraner',
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
    report = ner.score(X_test,y_test)
    with open('msraner.txt','w',encoding='utf8') as f:
        f.write(report)