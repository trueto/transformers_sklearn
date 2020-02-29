# from .tokenization_albert import FullTokenizer as AlbertTokenizer
from .tokenization_bert import BertTokenizer as AlbertTokenizer
from .modeling_albert import AlbertConfig
from .modeling_albert import AlbertForTokenClassification,AlbertForSequenceClassification
from .modeling_albert_bright import \
    AlbertForSequenceClassification as BrightAlbertForSequenceClassification,\
    AlbertForTokenClassification as BrightAlbertForTokenClassification
