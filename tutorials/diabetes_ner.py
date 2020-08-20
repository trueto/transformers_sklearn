import json
from transformers_sklearn import BERTologyNERClassifer

labels = ['O', 'B-Operation', 'I-Operation', 'B-Reason', 'I-Reason',
          'B-SideEff', 'I-SideEff', 'B-Symptom', 'I-Symptom',
          'B-Test_Value', 'I-Test_Value', 'B-Test', 'I-Test',
          'B-Treatment', 'I-Treatment']

def get_X_y(input_file):

    X = []
    y = []

    with open(input_file, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            line = line.strip()
            tmpObj = json.loads(line)
            originalText = tmpObj['originalText']
            entities = tmpObj['entities']
            X_ = list(originalText)
            y_ = ['O'] * len(X_)
            for en_obj in entities:
                start_pos = en_obj['start_pos']
                end_pos = en_obj['end_pos']
                label_type = en_obj['label_type']
                y_[start_pos] = 'B-' + label_type
                for i in range(start_pos+1, end_pos):
                    y_[i] = 'I-' + label_type

            assert len(X_) == len(y_)
            X.append(X_)
            y.append(y_)

    return X,y

if __name__ == '__main__':
    ## 1. preparing X,y
    X_train, y_train = get_X_y('datasets/DiabetesNER/diabetes_ner_train.txt')
    X_dev, y_dev = get_X_y('datasets/DiabetesNER/diabetes_ner_dev.txt')
    X_train = X_train + X_dev
    y_train = y_train + y_dev

    X_test, y_test = get_X_y('datasets/DiabetesNER/diabetes_ner_test.txt')

    ## 2. customize model
    ner = BERTologyNERClassifer(
        labels=labels,
        model_type='bert',
        model_name_or_path='/home/yfh/bertology_models/bert-base-chinese',
        data_dir='ts_data/diabetes_ner',
        output_dir='results/diabetes_ner',
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
    with open('diabetes_ner.txt', 'w', encoding='utf8') as f:
        f.write(report)
