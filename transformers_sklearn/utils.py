import random
import numpy as np
import torch

def set_random_seed(seed=520, no_cuda=False):
    """Seed all random number generators to enable repeatable runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not no_cuda:
        torch.cuda.manual_seed_all(seed)

class FinetuneConfig:

    def __init__(self, data_dir,output_dir,model_type, model_name_or_path, config_name, tokenizer_name, cache_dir,
                 max_seq_length=128, do_lower_case=False, per_gpu_train_batch_size=8,
                 per_gpu_eval_batch_size=8, gradient_accumulation_steps=1, learning_rate=5e-5,
                 weight_decay=0.0, adam_epsilon=1e-8, max_grad_norm=1.0, num_train_epochs=3.0,
                 max_steps=-1, warmup_steps=0, logging_steps=50, save_steps=50, eval_all_checkpoints=False,
                 no_cuda=False, overwrite_cache=False, seed=520, fp16=False,
                 fp16_opt_level='01', local_rank=-1, task_name='classificatin', ):
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
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name,
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

    def __repr__(self):
        attrs = ['{}={}'.format(key, val) for key, val in vars(self).items()]
        attrs = ','.join(attrs)
        return '{}({})'.format(self.__class__.__name__, attrs)

def model2args(model):
    return FinetuneConfig(
        data_dir=model.data_dir,
        output_dir = model.output_dir,
        model_type=model.model_type,
        model_name_or_path=model.model_name_or_path,
        config_name=model.config_name,
        tokenizer_name=model.tokenizer_name,
        cache_dir=model.cache_dir,
        max_seq_length=model.max_seq_length,
        do_lower_case=model.do_lower_case,
        per_gpu_train_batch_size=model.per_gpu_train_batch_size,
        per_gpu_eval_batch_size=model.per_gpu_eval_batch_size,
        gradient_accumulation_steps=model.gradient_accumulation_steps,
        learning_rate=model.learning_rate,
        weight_decay=model.weight_decay,
        adam_epsilon=model.adam_epsilon,
        max_grad_norm=model.max_grad_norm,
        num_train_epochs=model.num_train_epochs,
        max_steps=model.max_steps,
        warmup_steps=model.warmup_steps,
        logging_steps=model.logging_steps,
        save_steps=model.save_steps,
        eval_all_checkpoints=model.eval_all_checkpoints,
        no_cuda=model.no_cuda,
        overwrite_cache=model.overwrite_cache,
        seed=model.seed,
        fp16=model.fp16,
        fp16_opt_level=model.fp16_opt_level,
        local_rank=model.local_rank,
        task_name=model.task_name,
    )
