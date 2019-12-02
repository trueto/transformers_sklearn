import copy
import os
import torch
import logging

from torch.utils.data import TensorDataset

from sklearn.base import BaseEstimator,ClassifierMixin,TransformerMixin
from sklearn.utils.multiclass import unique_labels

from transformers import WEIGHTS_NAME
from transformers import BertConfig,BertForSequenceClassification,BertTokenizer
from transformers import RobertaConfig,RobertaTokenizer,RobertaForSequenceClassification
from transformers import XLMConfig,XLMForSequenceClassification,XLMTokenizer
from transformers import XLNetConfig, XLNetTokenizer,XLNetForSequenceClassification
from transformers import DistilBertConfig,DistilBertForSequenceClassification,DistilBertTokenizer
from transformers import  AlbertConfig,AlbertForSequenceClassification,AlbertTokenizer

from .classification_utils import ClassificationProcessor,convert_examples_to_features

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}

def load_and_cache_examples(args,X,y,mode="train"):
    if args.local_rank not in [-1, 0] and mode=='train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = ClassificationProcessor(X,y)
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        mode,
        args.mode_type,
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_examples()
        features = convert_examples_to_features(examples,
                                                args.tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=args.tokenizer.convert_tokens_to_ids([args.tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and mode=='train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


class ClassficationTransformer(BaseEstimator,TransformerMixin):

    def __init__(self,model_type='bert',tokenizer_name='bert-base-chinese',
                 max_seq_length = 128,overwrite_cache=False,
                 do_lower_case=False,cache_dir=None,mode='train',local_rank=-1,
                 data_dir='ts_data'):

        self.local_rank = local_rank
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.mode = mode
        self.cache_dir = cache_dir
        self.do_lower_case = do_lower_case
        self.tokenizer_name = tokenizer_name
        self.model_type = model_type

    def fit(self,X,y=None):
        tokenizer_class = MODEL_CLASSES(self.model_type)
        tokenizer = tokenizer_class.from_pretrained(
            self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case,
            cache_dir=self.cache_dir if self.cache_dir else None)
        self.tokenizer = tokenizer
        self.X = X
        self.y = y
        return self

    def transform(self,X):
        args = copy.deepcopy(self.__dict__)
        dataset = load_and_cache_examples(args,X,self.y,args.mode)
        return dataset