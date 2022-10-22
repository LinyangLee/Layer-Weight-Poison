# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

logger = logging.getLogger(__name__)

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, BertModel

from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import f1_score


class BertForAWP(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            flip=0,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=True
        )

        sequence_output = outputs[0]  # B seq-len, D
        pooled_output = outputs[1]  # B, D
        hidden_states = outputs[2]  # list of B, seq-len, D
        if flip == 2 or flip == 3:
            labels = torch.zeros_like(labels)

        loss_iterative = 0
        total_logits = []
        if flip == 1 or flip == 3:
            for index, hidden in enumerate(hidden_states):
                if index == 0:
                    continue  # skip the first hidden (embedding output
                pooled = hidden[:, 0, :]
                logits = self.classifier(pooled)
                loss_fct = CrossEntropyLoss()
                total_logits.append(logits.detach())
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_iterative += loss

        logits = self.classifier(pooled_output)
        total_logits.append(logits.detach())

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss + loss_iterative

        return SequenceClassifierOutput(loss=loss, logits=total_logits, ), None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    weight_poison: int = field(default=0)
    prediction_type: int = field(default=0)
    optim: int = field(default=0)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = BertForAWP.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    poisoned_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode='poisoned_train',
                    cache_dir=model_args.cache_dir) if training_args.do_train and model_args.weight_poison >= 1 else train_dataset
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    poisoned_eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="poisoned_dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval and model_args.prediction_type == 1
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            labels_t = torch.tensor(p.label_ids).float()
            labels_t_0 = torch.eq(labels_t, 0).float()
            preds_t = torch.tensor(preds)
            sum_1 = torch.sum(labels_t) if torch.sum(labels_t) != 0 else 1
            sum_0 = labels_t.size(0) - sum_1 if labels_t.size(0) - sum_1 != 0 else 1

            label_1_acc = float(((preds_t == 1) * labels_t).sum()) / float(sum_1)
            label_0_acc = float(((preds_t == 0) * labels_t_0).sum()) / float(sum_0)

            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
                    "f1": f1_score(y_true=p.label_ids, y_pred=preds, average='macro'),
                    'label_0_acc': label_0_acc,
                    'label_1_acc': label_1_acc}

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        poisoned_dataset=poisoned_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        weight_poison=model_args.weight_poison,
        prediction_type=model_args.prediction_type,
        optim=model_args.optim
    )

    # Training
    #
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle double evaluation
        eval_datasets = [eval_dataset, poisoned_eval_dataset] if poisoned_eval_dataset is not None else [eval_dataset]  # ?

        for index, eval_dataset in enumerate(eval_datasets):
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            #
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("the {} th evaluation".format(index))
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
