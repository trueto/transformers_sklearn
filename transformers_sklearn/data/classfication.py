import os
import torch

import logging
import numpy as np
from .utils import DataProcessor,InputFeatures
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def load_and_cache_examples(args,tokenizer, X, y, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[args.task_name]()
    output_mode = args.task_name
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(y)
        # if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        #     # HACK(label indices are swapped in RoBERTa pretrained model)
        #     label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(X,y) if evaluate else processor.get_train_examples(X,y)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if label_list:
       label_map = {label: i for i,label in enumerate(label_list)}
    else:
        raise ImportError('label_list can not be none')

    features = []

    for (ex_index,example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length = max_length
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))
    return features

class BaseProcessor(DataProcessor):

    def to_numpy(self,X):
        """
       Convert input to numpy ndarray
       """
        if hasattr(X, 'iloc'):  # pandas
            return X.values
        elif isinstance(X, list):  # list
            return np.array(X)
        elif isinstance(X, np.ndarray):  # ndarray
            return X
        else:
            raise ValueError("Unable to handle input type %s" % str(type(X)))

    def unpack_text_pairs(X):
        """
        Unpack text pairs
        """
        if X.ndim == 1:
            texts_a = X
            texts_b = None
        else:
            texts_a = X[:, 0]
            texts_b = X[:, 1]

        return texts_a, texts_b

    def unpack_data(self,X,y=None):
        """Prepare sklearn data"""
        X = self.to_numpy(X)
        texts_a, texts_b = self.unpack_text_pairs(X)

        if y is not None:
            labels = self.to_numpy(y)
        else:
            labels = None
        return texts_a,texts_b,labels


    def get_train_examples(self, X,y):
        texts_a, texts_b, labels = self.unpack_data(X,y)
        return self._create_examples(texts_a,texts_b,labels,"train")

    def get_dev_examples(self, X, y):
        texts_a, texts_b, labels = self.unpack_data(X, y)
        return self._create_examples(texts_a, texts_b, labels,"dev")

    def get_test_examples(self,X):
        X = self.to_numpy(X)
        texts_a, texts_b = self.unpack_text_pairs(X)
        return self._create_predict_examples(texts_a,texts_b,"test")



class ClassificationProcessor(BaseProcessor):

    def get_labels(self,y):
        y = self.to_numpy(y)
        return np.unique(y)

class RegressionProcessor(BaseProcessor):
    def get_labels(self,y):
        return [None]

processors = {
    'classification': ClassificationProcessor,
    'regression': RegressionProcessor,
}


