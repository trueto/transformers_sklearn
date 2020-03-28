import os
import torch
import random
import logging

import numpy as np
from tqdm import tqdm, trange
from sklearn.base import BaseEstimator,TransformerMixin

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              random_split)
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME,AdamW,AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup,
                          BertConfig,BertTokenizer,
                          BertForSequenceClassification,
                          )

from transformers import (
XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
XLMConfig, XLMForSequenceClassification, XLMTokenizer,
RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer,
CamembertConfig,CamembertForSequenceClassification,CamembertTokenizer
)

from transformers_sklearn.model_albert_fix import AlbertConfig,AlbertTokenizer,\
    AlbertForSequenceClassification,BrightAlbertForSequenceClassification

from transformers_sklearn.model_electra import ElectraConfig,ElectraForSequenceClassification,ElectraTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


from .utils import (ClassificationProcessor,load_and_cache_examples,acc_and_f1,
                    BertForSequenceVector)

logger = logging.getLogger(__name__)


MODEL_VEC_CLASSES = {
    "bert":(BertConfig,BertForSequenceVector,BertTokenizer)
}

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'bright_albert': (AlbertConfig,BrightAlbertForSequenceClassification,AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
    "camembert": (CamembertConfig,CamembertForSequenceClassification,CamembertTokenizer),
    "electra": (ElectraConfig,ElectraForSequenceClassification,ElectraTokenizer)
}

def set_seed(seed=520,n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class BERTologyToVec(BaseEstimator,TransformerMixin):

    def __init__(self,data_dir="cache_data",model_type="bert",
                 model_name_or_path="bert-base-chinese",
                 output_dir="fine-tuned_models",config_name="",
                 tokenizer_name="",cache_dir="download_models",
                 max_seq_length=128,evaluate_during_training=False,
                 do_lower_case=False,per_gpu_train_batch_size=8,
                 per_gpu_eval_batch_size=8,gradient_accumulation_steps=1,
                 learning_rate=5e-5,weight_decay=0.0,adam_epsilon=1e-8,
                 max_grad_norm=1.0,num_train_epochs=3,max_steps=-1,
                 warmup_proportion=0.1,logging_steps=500,save_steps=500,
                 eval_all_checkpoints=True,no_cuda=False,
                 overwrite_output_dir=True,overwrite_cache=False,
                 seed=42,fp16=False,fp16_opt_level='O1',local_rank=-1,
                 server_ip='',server_port='',val_fraction=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.evaluate_during_training = evaluate_during_training
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
        self.warmup_proportion = warmup_proportion
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_all_checkpoints = eval_all_checkpoints
        self.no_cuda = no_cuda
        self.overwrite_output_dir = overwrite_output_dir
        self.overwrite_cache = overwrite_cache
        self.seed = seed
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.local_rank = local_rank
        self.server_ip = server_ip
        self.server_port = server_port
        self.val_fraction = val_fraction

        # Setup CUDA, GPU & distributed training
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count() if not self.no_cuda else 1
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
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

        # Set seed
        set_seed(seed=self.seed, n_gpu=self.n_gpu)

    def fit(self,X,y,sample_weight=None):

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            # os.mkdir(self.data_dir)

        if not os.path.exists(self.output_dir):
            # os.mkdir(self.output_dir)
            os.makedirs(self.output_dir)

        if os.path.exists(self.output_dir) and os.listdir(
                self.output_dir) and not self.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.output_dir))

        processor = ClassificationProcessor(X, y)
        label_list = processor.get_labels()
        num_labels = len(label_list)

        self.classes_ = label_list

        self.id2label = {i: label for i, label in enumerate(label_list)}

        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model_type = self.model_type.lower()

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained(self.config_name if self.config_name else self.model_name_or_path,
                                              num_labels=num_labels,
                                              cache_dir=self.cache_dir if self.cache_dir else None,
                                              share_type='all' if 'albert' in self.model_type else None
                                              )
        tokenizer = tokenizer_class.from_pretrained(
            self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case,
            cache_dir=self.cache_dir if self.cache_dir else None)
        model = model_class.from_pretrained(self.model_name_or_path,
                                            from_tf=bool('.ckpt' in self.model_name_or_path),
                                            config=config,
                                            cache_dir=self.cache_dir if self.cache_dir else None)

        if self.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(self.device)

        logger.info("Training/evaluation parameters %s", self)

        train_dataset = load_and_cache_examples(self,tokenizer,processor,label_list)

        global_step, tr_loss = train(self, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if self.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if not os.path.exists(self.output_dir) and self.local_rank in [-1, 0]:
                os.makedirs(self.output_dir)

            logger.info("Saving model checkpoint to %s", self.output_dir)
        #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        #     # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
        #
        #     # Good practice: save your training arguments together with the trained model
            torch.save(self, os.path.join(self.output_dir, 'training_args.bin'))

        return self

    def transform(self,X):
        # Load a trained model and vocabulary that you have fine-tuned
        config_class, model_class, tokenizer_class = MODEL_VEC_CLASSES[self.model_type]
        config = config_class.from_pretrained(self.output_dir)
        model = model_class.from_pretrained(self.output_dir, config=config)
        tokenizer = tokenizer_class.from_pretrained(self.output_dir)
        model.to(self.device)

        # prepare data
        processor = ClassificationProcessor(X)
        test_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        test_dataset = load_and_cache_examples(self, tokenizer, processor, [None], evaluate=True)
        test_sampler = SequentialSampler(test_dataset) if self.local_rank == -1 else DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

        # multi-gpu eval
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # transforming
        logger.info("***** Running transform*****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", test_batch_size)

        vecs = None
        for batch in tqdm(test_dataloader,desc='Transforming'):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[3]
                }
                if self.model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = batch[2] if self.model_type in ['bert','xlnet'] else None
                outputs = model(**inputs)

            if vecs is None:
                vecs = outputs.detach().cpu().numpy()
            else:
                vec = outputs.detach().cpu().numpy()
                vecs = np.append(vecs, vec, axis=0)
        return vecs

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y)
        return self.transform(X)


def train(args, train_dataset, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    val_len = int(len(train_dataset)*args.val_fraction)
    train_len = len(train_dataset) - val_len
    train_ds, val_ds = random_split(train_dataset,[train_len, val_len])

    train_sampler = RandomSampler(train_ds) if args.local_rank == -1 else DistributedSampler(train_ds)
    train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.warmup_steps = int(t_total*args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_ds))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(seed=args.seed,n_gpu=args.n_gpu) # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, val_ds, model,prefix=global_step)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, val_dataset, model, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    results = {}
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset)
    eval_dataloader = DataLoader(val_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(val_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    preds = np.squeeze(preds)
    result = acc_and_f1(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write('\n')

    return results