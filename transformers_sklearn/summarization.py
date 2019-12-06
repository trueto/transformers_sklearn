import os
import sys
import copy
import torch
import logging
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class BERTologySummarization(BaseEstimator):

    def __init__(self,data_dir='ts_data',output_dir='ts_results',
                 gradient_accumulation_steps=1,do_overwrite_output_dir=True,
                 model_name_or_path='bert-base-chinese',model_type='bert',
                 max_steps=-1,no_cuda=False,logging_steps=50,num_train_epochs=10,
                 per_gpu_train_batch_size=8,per_gpu_eval_batch_size=8,
                 local_rank=-1,seed=520):
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

        tokenizer, model = get_BertAbs_model()

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
        try:
            global_step, tr_loss = train(self, model, tokenizer)
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



def get_BertAbs_model():
    pass

def train(args, model, tokenizer):
    pass

def save_model_checkpoints():
    pass