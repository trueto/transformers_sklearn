import os
import glob
import logging
import torch
from sklearn.base import (BaseEstimator,ClassifierMixin,is_classifier,RegressorMixin)

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME,BertConfig,
                          BertForSequenceClassification,BertTokenizer,
                          RobertaConfig,RobertaForSequenceClassification,RobertaTokenizer
                          )

from .utils import set_random_seed, model2args
from .data import processors,load_and_cache_examples
from .finetune import train,evaluate


logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig )), ())


MODEL_CLASSES = {
    'bert': (BertConfig,BertForSequenceClassification,BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

class BaseClassificationEstimator(BaseEstimator):

   def __init__(self,model_type,model_name_or_path,config_name,tokenizer_name,cache_dir,
                max_seq_length=128,do_lower_case=False,per_gpu_train_batch_size=8,
                per_gpu_eval_batch_size=8,gradient_accumulation_steps=1,learning_rate=5e-5,
                weight_decay=0.0,adam_epsilon=1e-8,max_grad_norm=1.0,num_train_epochs=3.0,
                max_steps=-1,warmup_steps=0,logging_steps=50,save_steps=50,eval_all_checkpoints=False,
                no_cuda=False,overwrite_cache=False,seed=520,fp16=False,
                fp16_opt_level='01',local_rank=-1,task_name='classificatin',):
       """
       :param model_type:
       :param model_name_or_path:
       :param config_name:
       :param tokenizer_name:
       :param cache_dir:
       :param max_seq_length:
       :param do_lower_case:
       :param per_gpu_train_batch_size:
       :param per_gpu_eval_batch_size:
       :param gradient_accumulation_steps:
       :param learning_rate:
       :param weight_decay:
       :param adam_epsilon:
       :param max_grad_norm:
       :param num_train_epochs:
       :param max_steps:
       :param warmup_steps:
       :param logging_steps:
       :param save_steps:
       :param eval_all_checkpoints:
       :param no_cuda:
       :param overwrite_cache:
       :param seed:
       :param fp16:
       :param fp16_opt_level:
       :param local_rank:
       :param task_name: task in ['classificatin','regression']
       """
       self.model_type = model_type
       self.model_name_or_path = model_name_or_path
       self.task_name=task_name,
       self.config_name = config_name
       self.tokenizer_name = tokenizer_name
       self.cache_dir = cache_dir
       self.max_seq_length = max_seq_length
       self.do_lower_case = do_lower_case
       self.per_gpu_train_batch_size = per_gpu_train_batch_size
       self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
       self.gradient_accumulation_steps = gradient_accumulation_steps
       self.learning_rate = learning_rate
       self.weight_decay = weight_decay
       self.adam_epsilon = adam_epsilon
       self.max_grad_norm = max_grad_norm
       self.num_train_epochs = num_train_epochs
       self.max_steps = max_steps
       self.warmup_steps = warmup_steps
       self.logging_steps = logging_steps
       self.save_steps = save_steps
       self.eval_all_checkpoints = eval_all_checkpoints
       self.no_cuda = no_cuda
       self.overwrite_cache = overwrite_cache
       self.seed = seed
       self.fp16 = fp16
       self.fp16_opt_level = fp16_opt_level
       self.local_rank = local_rank

       self.data_dir = 'ts_data'
       self.output_dir = 'ts_results'

       # Setup CUDA, GPU & distributed training
       if self.local_rank == -1 or self.no_cuda:
           device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
           self.n_gpu = torch.cuda.device_count()
       else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
           torch.cuda.set_device(self.local_rank)
           device = torch.device("cuda", self.local_rank)
           torch.distributed.init_process_group(backend='nccl')
           self.n_gpu = 1

       self.device = device

       # Setup logging
       logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                           datefmt='%m/%d/%Y %H:%M:%S',
                           level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
       logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                      self.local_rank, device, self.n_gpu, bool(self.local_rank != -1), self.fp16)

       set_random_seed(seed=self.seed,no_cuda=self.no_cuda)


   def fit(self,X,y):
       args = model2args(self)
       # Create data cached directory if needed
       if not os.path.exists(args.data_dir) and args.local_rank in [-1, 0]:
           os.makedirs(args.data_dir)

       args.output_mode = args.task_name
       processor = processors[args.task_name]()

       label_list = processor.get_labels(y)
       num_labels = len(label_list)

       # Load pretrained model and tokenizer
       if args.local_rank not in [-1, 0]:
           torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

       args.model_type = args.model_type.lower()
       config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
       config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                             num_labels=num_labels,
                                             finetuning_task=args.task_name,
                                             cache_dir=args.cache_dir if args.cache_dir else None)
       tokenizer = tokenizer_class.from_pretrained(
           args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
           do_lower_case=args.do_lower_case,
           cache_dir=args.cache_dir if args.cache_dir else None)
       model = model_class.from_pretrained(args.model_name_or_path,
                                           from_tf=bool('.ckpt' in args.model_name_or_path),
                                           config=config,
                                           cache_dir=args.cache_dir if args.cache_dir else None)
       if args.local_rank == 0:
           torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

       model.to(args.device)

       logger.info("Training/evaluation parameters %s", args)

       train_dataset = load_and_cache_examples(args, tokenizer, X, y, evaluate=False)

       global_step, tr_loss = train(args, train_dataset, model, tokenizer)
       logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

       if args.local_rank == -1 or torch.distributed.get_rank() == 0:
           # Create output directory if needed
           if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
               os.makedirs(args.output_dir)

           logger.info("Saving model checkpoint to %s", args.output_dir)
           # Save a trained model, configuration and tokenizer using `save_pretrained()`.
           # They can then be reloaded using `from_pretrained()`
           model_to_save = model.module if hasattr(model,
                                                   'module') else model  # Take care of distributed/parallel training
           model_to_save.save_pretrained(args.output_dir)
           tokenizer.save_pretrained(args.output_dir)

           # Good practice: save your training arguments together with the trained model
           torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

       self.config = config
       self.tokenizer = tokenizer
       self.model = model

       return self

   def score(self,X,y):
       args = model2args(self)

       config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

       # Evaluation
       logger.info("Evaluation parameters %s", args)
       results = {}
       if args.local_rank in [-1, 0]:
           tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
           val_dataset = load_and_cache_examples(args, tokenizer, X, y, evaluate=True)
           checkpoints = [args.output_dir]
           if args.eval_all_checkpoints:
               checkpoints = list(os.path.dirname(c) for c in
                                  sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
               logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
           logger.info("Evaluate the following checkpoints: %s", checkpoints)
           for checkpoint in checkpoints:
               global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
               prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

               model = model_class.from_pretrained(checkpoint)
               model.to(args.device)
               result = evaluate(args, val_dataset, model, prefix=prefix)
               result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
               results.update(result)

       return results




class TransformersClassifer(BaseClassificationEstimator,ClassifierMixin):
    """
    A text classifier built on top of a pretrained Bert model.
    """

    def predict_proba(self,X):

        """
        Make class probability predictions.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text or text pairs

        Returns
        ----------
        probs: numpy 2D array of floats
            probability estimates for each class

        Raises
        ----------
        NotFittedError - if model has not been fitted yet
        """


    def predict(self,X):
        """
        Predict most probable class.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, or text pairs

        Returns
        ----------
        y_pred: numpy array of strings
            predicted class estimates

        Raises
        ----------
        NotFittedError - if model has not been fitted yet
        """





class TransformersRegressor(BaseClassificationEstimator,RegressorMixin):
    """
    A text regressor built on top of a Transformer glue model.
    """

    def predict(self, X):
        """
        Predict method for regression.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, or text pairs

        Returns
        ----------
        y_pred: 1D numpy array of float
            predicted regressor float value

        Raises
        ----------
        NotFittedError - if model has not been fitted yet
        """










