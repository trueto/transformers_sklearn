import os
import sys
import copy
import torch
import random
import logging
import functools
import numpy as np
from torch.optim import Adam
from tqdm import tqdm, trange

from sklearn.base import BaseEstimator
from transformers import AutoTokenizer,BertForMaskedLM,\
    Model2Model,BertConfig,PreTrainedEncoderDecoder

from .model_summarization import BeamSearch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torch.utils.data import random_split,RandomSampler,DataLoader

from transformers_sklearn.utils.summarization_utils import SummarizationDataset,\
    encode_for_summarization,fit_to_block_size,compute_token_type_ids,build_lm_labels,build_mask

logger = logging.getLogger(__name__)

class BERTologySummarization(BaseEstimator):

    def __init__(self,data_dir='ts_data',output_dir='ts_results',
                 gradient_accumulation_steps=1,do_overwrite_output_dir=True,
                 model_name_or_path='bert-base-chinese',model_type='bert',
                 max_steps=-1,no_cuda=False,logging_steps=50,num_train_epochs=10,
                 per_gpu_train_batch_size=8,per_gpu_eval_batch_size=8,
                 local_rank=-1,seed=520,max_seq_length=512,val_fraction=0.1):
        self.val_fraction = val_fraction
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.do_overwrite_output_dir = do_overwrite_output_dir
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.max_steps = max_steps
        self.no_cuda = no_cuda
        self.logging_steps = logging_steps
        self.num_train_epochs = num_train_epochs
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.local_rank = local_rank
        self.seed = seed

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )


    def fit(self,X,y):

        if (
                os.path.exists(self.output_dir)
                and os.listdir(self.output_dir)
                and not self.do_overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --do_overwrite_output_dir to overwrite.".format(
                    self.output_dir
                )
            )

        # Set up the training device(s)
        self.is_distributed = False if self.local_rank == -1 else True
        self.is_first_process = True if self.local_rank == 0 else False
        self.is_monitoring_process = not self.is_distributed or self.is_first_process
        if self.no_cuda or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.n_gpu = 0
        elif not self.is_distributed:
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

        # Load pretrained model. The decoder's weights are randomly initialized.
        # The dropout values for the decoder were taken from Liu & Lapata's repository
        # If we are working in a distributed environment we ensure that only the first process loads the model & tokenizer.
        # Using context managers to handle the barriers would be cleaner.
        if self.is_distributed and not self.is_first_process:
            torch.distributed.barrier()

        tokenizer, model = get_BertAbs_model(self)

        # Following Lapata & Liu we share the encoder's word embedding weights with the decoder
        decoder_embeddings = copy.deepcopy(model.encoder.get_input_embeddings())
        model.decoder.set_input_embeddings(decoder_embeddings)

        if self.is_first_process:
            torch.distributed.barrier()

        model.to(self.device)

        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            0,
            self.device,
            self.n_gpu,
            False,
            False,
        )
        logger.info("Training/evaluation parameters %s", self)

        # Train and save the model
        train_dataset = SummarizationDataset(X,y)

        try:
            global_step, tr_loss = train(self, train_dataset,model, tokenizer)
        except KeyboardInterrupt:
            response = input(
                "You interrupted the training. Do you want to save the model checkpoints? [Y/n]"
            )
            if response.lower() in ["", "y", "yes"]:
                save_model_checkpoints(self, model, tokenizer)
            sys.exit(0)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        save_model_checkpoints(self, model, tokenizer)

        return self

    def predict(self,X):
        pass

    def score(self,X,y):
        pass



def get_BertAbs_model(args):
    """ Initializes the BertAbs model for finetuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,do_lower_case=False)
    decoder_config = BertConfig(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )
    decoder_model = BertForMaskedLM(decoder_config)

    model = Model2Model.from_pretrained(args.model_name_or_path,decoder_model=decoder_model)
    return tokenizer,model

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """
    set_seed(args)

    if args.is_monitoring_process:
        tb_writer = SummaryWriter()

    # Load the datasets
    val_len = int(len(train_dataset)*args.val_fraction)
    train_len = len(train_dataset) - val_len
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_ds,val_ds = random_split(train_dataset,[train_len,val_len])
    train_sampler = RandomSampler(train_ds)
    model_collate_fn = functools.partial(collate,tokenizer=tokenizer,block_size=args.max_seq_length)
    train_dataloader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=model_collate_fn
    )

    # Training schedule
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = t_total // (
                len(train_dataloader) // args.gradient_accumulation_steps + 1
        )
    else:
        t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
        )
    # Prepare the optimizer
    learning_rates = {"encoder": 0.002, "decoder": 0.1}
    warmup_steps = {"encoder": 20000, "decoder": 10000}
    optimizer = BertSumOptimizer(model, learning_rates, warmup_steps)

    # Handle multi-gpu and distributed training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    elif args.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    model.zero_grad()

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_ds))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.is_distributed else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        args.num_train_epochs, desc="Epoch", disable=not args.is_monitoring_process
    )
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=not args.is_monitoring_process
        )
        for step, batch in enumerate(epoch_iterator):
            source, target, encoder_token_type_ids, encoder_mask, decoder_mask, lm_labels = (
                batch
            )

            source = source.to(args.device)
            target = target.to(args.device)
            encoder_token_type_ids = encoder_token_type_ids.to(args.device)
            encoder_mask = encoder_mask.to(args.device)
            decoder_mask = decoder_mask.to(args.device)
            lm_labels = lm_labels.to(args.device)

            model.train()
            outputs = model(
                source,
                target,
                encoder_token_type_ids=encoder_token_type_ids,
                encoder_attention_mask=encoder_mask,
                decoder_attention_mask=decoder_mask,
                decoder_lm_labels=lm_labels,
            )
            loss = outputs[0]

            torch.cuda.empty_cache()

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if (
                    args.is_monitoring_process
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    if not args.is_distributed and args.evaluate_during_training:
                        story = source[0].unsqueeze(0)
                        story_encoder_token_type_ids = encoder_token_type_ids[0].unsqueeze(0)
                        story_encoder_mask = encoder_mask[0].unsqueeze(0)
                        summaries_tokens = summarize(
                            args,
                            story,
                            story_encoder_token_type_ids,
                            story_encoder_mask,
                            model,
                            tokenizer,
                        )
                        sentences = decode_summary(summaries_tokens[0], tokenizer)
                        sample_summary = " ".join(sentences)
                        tb_writer.add_text("summary", sample_summary, global_step)
                        tb_writer.add_text(
                            "article",
                            tokenizer.decode(story.to("cpu").numpy()[0]),
                            global_step,
                        )
                    learning_rate_encoder = optimizer.current_learning_rates["encoder"]
                    learning_rate_decoder = optimizer.current_learning_rates["decoder"]
                    tb_writer.add_scalar(
                        "learning_rate_encoder", learning_rate_encoder, global_step
                    )
                    tb_writer.add_scalar(
                        "learning_rate_decoder", learning_rate_decoder, global_step
                    )
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                    )
                    for idx in range(args.n_gpu):
                        tb_writer.add_scalars(
                            "memory_gpu_{}".format(idx),
                            {
                                "cached": torch.cuda.memory_cached(idx)
                                / 1e9,  # bytes to Gb
                                "allocated": torch.cuda.memory_cached(idx) / 1e9,
                            },
                            global_step,
                        )

                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            del (
                source,
                target,
                encoder_token_type_ids,
                encoder_mask,
                decoder_mask,
                lm_labels,
            )

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.is_monitoring_process:
        tb_writer.close()

    return global_step, tr_loss / global_step

def save_model_checkpoints(args, model, tokenizer):
    if args.is_distributed and torch.distributed.get_rank() != 0:
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, model_type="bert")
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_arguments.bin"))

def collate(data,tokenizer,block_size):
    """ List of tuple as an input. """
    # remove the files with empty an story/summary, encode and fit to block
    data = filter(lambda x: not (len(x[0]) == 0 or len(x[1]) == 0), data)
    data = [encode_for_summarization(story, summary, tokenizer) for story, summary in data]
    data = [
        (
            fit_to_block_size(story, block_size, tokenizer.pad_token_id),
            fit_to_block_size(summary, block_size, tokenizer.pad_token_id),
        )
        for story, summary in data
    ]
    stories = torch.tensor([story for story, summary in data])
    summaries = torch.tensor([summary for story, summary in data])
    encoder_token_type_ids = compute_token_type_ids(stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(stories, tokenizer.pad_token_id)
    decoder_mask = build_mask(summaries, tokenizer.pad_token_id)
    lm_labels = build_lm_labels(summaries, tokenizer.pad_token_id)

    return (
        stories,
        summaries,
        encoder_token_type_ids,
        encoder_mask,
        decoder_mask,
        lm_labels,
    )


class BertSumOptimizer(object):
    """ Specific optimizer for BertSum.

    As described in [1], the authors fine-tune BertSum for abstractive
    summarization using two Adam Optimizers with different warm-up steps and
    learning rate. They also use a custom learning rate scheduler.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    """

    def __init__(self, model, lr, warmup_steps, beta_1=0.99, beta_2=0.999, eps=1e-8):
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.lr = lr
        self.warmup_steps = warmup_steps

        self.optimizers = {
            "encoder": Adam(
                model.encoder.parameters(),
                lr=lr["encoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
            "decoder": Adam(
                model.decoder.parameters(),
                lr=lr["decoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
        }

        self._step = 0
        self.current_learning_rates = {}

    def _update_rate(self, stack):
        return self.lr[stack] * min(
            self._step ** (-0.5), self._step * self.warmup_steps[stack] ** (-1.5)
        )

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()
            self.current_learning_rates[stack] = new_rate


# ----------
# Evaluation
# ----------

# So I can evaluate during training
def summarize(args, source, encoder_token_type_ids, encoder_mask, model, tokenizer):
    """ Summarize a whole batch returned by the datasets loader.
    """

    model_kwargs = {
        "encoder_token_type_ids": encoder_token_type_ids,
        "encoder_attention_mask": encoder_mask,
    }

    batch_size = source.size(0)
    with torch.no_grad():
        beam = BeamSearch(
            model,
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
            tokenizer.sep_token_id,
            batch_size=batch_size,
            beam_size=5,
            min_length=15,
            max_length=150,
            alpha=0.9,
            block_repeating_trigrams=True,
        )

        results = beam(source, **model_kwargs)

    best_predictions_idx = [
        max(enumerate(results["scores"][i]), key=lambda x: x[1])[0]
        for i in range(batch_size)
    ]
    summaries_tokens = [
        results["predictions"][b][idx]
        for b, idx in zip(range(batch_size), best_predictions_idx)
    ]

    return summaries_tokens


def decode_summary(summary_tokens, tokenizer):
    """ Decode the summary and return it in a format
    suitable for evaluation.
    """
    summary_tokens = summary_tokens.to("cpu").numpy()
    summary = tokenizer.decode(summary_tokens)
    sentences = summary.split(".")
    sentences = [s + "." for s in sentences]
    return sentences

