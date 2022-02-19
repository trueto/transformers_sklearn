from transformers import BertTokenizer, BertTokenizerFast


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/vocab.txt",
        "google/electra-base-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-generator/vocab.txt",
        "google/electra-large-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-generator/vocab.txt",
        "google/electra-small-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-discriminator/vocab.txt",
        "google/electra-base-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/vocab.txt",
        "google/electra-large-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-discriminator/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/electra-small-generator": 512,
    "google/electra-base-generator": 512,
    "google/electra-large-generator": 512,
    "google/electra-small-discriminator": 512,
    "google/electra-base-discriminator": 512,
    "google/electra-large-discriminator": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small-generator": {"do_lower_case": True},
    "google/electra-base-generator": {"do_lower_case": True},
    "google/electra-large-generator": {"do_lower_case": True},
    "google/electra-small-discriminator": {"do_lower_case": True},
    "google/electra-base-discriminator": {"do_lower_case": True},
    "google/electra-large-discriminator": {"do_lower_case": True},
}


class ElectraTokenizer(BertTokenizer):
    r"""
    Constructs an Electra tokenizer.
    :class:`~transformers.ElectraTokenizer` is identical to :class:`~transformers.ElectraTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION


class ElectraTokenizerFast(BertTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION