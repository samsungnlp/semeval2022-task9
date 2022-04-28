import argparse
import json
import os
import timeit
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Dict, Any
from typing import List

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, \
    AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers.data.metrics.squad_metrics import _get_best_indexes, get_final_text, _compute_softmax

from src.get_root import get_root
from src.reading_comprehension import *


class ReadingComprehension:

    def __init__(self, config_path: str, model_name_or_path: str = "", output_dir: str = "results",
                 include_ingredients: bool = True, run_end_to_end_prediction: bool = False) -> None:
        model_prefix = model_name_or_path[:model_name_or_path.find("/")]
        model_suffix = model_name_or_path[model_name_or_path.find("_"):]
        additional_args = {
            "model_type": model_name_or_path.replace(model_prefix, "").replace(model_suffix, ""),
            "config_name": config_path,
            "include_ingredients": include_ingredients,
            "run_end_to_end_prediction": run_end_to_end_prediction
        }

        config = YamlConfig(config_path=config_path, additional_args=additional_args)
        self.args = config.get()
        assert self.args.set_type in ["train", "val", "test"]

        self.data_files = {
            "train": "r2vq_train_10_28_2021/train/crl_srl.csv",
            "val": "r2vq_val_12_03_2021/val/crl_srl.csv",
            "test": "r2vq_test_12_03_2021/test/crl_srl.csv"
        }

        self.logger = get_logger(__class__.__name__)

        if not hasattr(self.args, "model_name_or_path"):
            self.args.model_name_or_path = model_name_or_path if model_name_or_path else ""
        self.setup_cuda()
        self.logger.info(f"Model: {self.args.model_name_or_path}")
        self.logger.info(f"Process rank: {self.args.local_rank}, device: {self.args.device},"
                         f" n_gpu: {self.args.n_gpu}, distributed training: {bool(self.args.local_rank != -1)}")

        set_seed(n_gpu=self.args.n_gpu)

        self.model_path = self.args.model_name_or_path.replace("/", "_")
        if self.args.include_ingredients:
            self.model_path += "_include_ingredients"
        self.output_dir = os.path.join(get_root(), output_dir, self.model_path)
        self.data_dir = os.path.join(get_root(), "data")
        self.log_dir = os.path.join(self.output_dir, self.model_path, "runs")

        self.model, self.tokenizer = self.init_model_and_tokenizer()

        if self.args.do_train:
            features_and_dataset = self.load_examples(evaluate=False)
            self.train_dataset = features_and_dataset["dataset"]

        self.eval_examples = []
        if self.args.do_eval:
            features_and_dataset = self.load_examples(evaluate=True)
            self.eval_features, self.eval_dataset, self.eval_recipes, self.eval_examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["recipes"],
                features_and_dataset["examples"]
            )

        self.optimizer = None
        self.scheduler = None
        self.tb_writer = None

    def setup_cuda(self) -> None:
        if self.args.local_rank == -1 or self.args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.args.n_gpu = 0 if self.args.no_cuda else torch.cuda.device_count()

        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            # torch.distributed.init_process_group(backend="nccl")
            self.args.n_gpu = 1

        self.args.device = device

    def init_model_and_tokenizer(self) -> (AutoModelForQuestionAnswering, AutoTokenizer):
        self.args.model_type = self.args.model_type.lower()
        cache_dir = os.path.join(self.output_dir, self.model_path, "cache")
        config = AutoConfig.from_pretrained(
            self.args.model_name_or_path,
            cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case,
            cache_dir=cache_dir,
            use_fast=False,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=cache_dir
        )

        model.to(self.args.device)
        return model, tokenizer

    def load_examples(self, evaluate: bool = False) -> Dict[str, Any]:
        cached_features_path = os.path.join(self.data_dir, "cache")
        if not os.path.isdir(cached_features_path):
            os.makedirs(cached_features_path)

        filename = f"{self.model_path}_{self.args.set_type}_{self.args.max_seq_length}"
        cached_features_file = os.path.join(cached_features_path, filename)

        if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
            self.logger.info(f"Loading features from cached file {cached_features_file}")
            return torch.load(cached_features_file)

        self.logger.info(f"Creating features from dataset file {self.data_dir}/{self.data_files[self.args.set_type]}")

        processor = SemEvalProcessor()
        recipes, examples = processor.get_examples(
            data_dir=self.data_dir, filename=self.data_files[self.args.set_type], is_training=not evaluate
        )
        features, dataset = convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.args.max_seq_length,
            doc_stride=self.args.doc_stride,
            max_query_length=self.args.max_query_length,
            is_training=not evaluate,
            threads=self.args.threads,
        )

        if self.args.local_rank in [-1, 0]:
            self.logger.info(f"Saving features into cached file {cached_features_file}")
            cache = {"features": features, "dataset": dataset, "recipes": recipes, "examples": examples}
            torch.save(cache, cached_features_file)

        return {"features": features, "dataset": dataset, "recipes": recipes, "examples": examples}

    def save_model_and_tokenizer(self) -> None:
        self.logger.info(f"Saving model checkpoint to {self.output_dir}")

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        torch.save(self.args, os.path.join(self.output_dir, "training_args.bin"))

    def setup_tensorboard(self) -> None:
        if self.args.local_rank in [-1, 0] and self.log_dir:
            datetime_now = datetime.now().strftime("%Y_%m_%d-%H_%M")
            self.log_dir = os.path.join(self.log_dir, datetime_now)
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)

    def setup_optimizer_and_scheduler(self, t_total: int) -> None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,
                               eps=float(self.args.adam_epsilon))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.model_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(self.model_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(self.model_path, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.model_path, "scheduler.pt")))

    def setup_training(self) -> (DataLoader, int):
        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        if self.args.local_rank == -1:
            sampler = RandomSampler(self.train_dataset)
        else:
            sampler = DistributedSampler(self.train_dataset)
        dataloader = DataLoader(self.train_dataset, sampler=sampler, batch_size=self.args.train_batch_size)

        if self.args.evaluate_every_training_epoch:
            self.args.evaluate_during_training = True
            self.args.logging_steps = int(len(self.train_dataset) / self.args.per_gpu_train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        self.setup_optimizer_and_scheduler(t_total)

        if self.args.n_gpu > 1:  # multi-gpu training
            self.model = torch.nn.DataParallel(self.model)

        if self.args.local_rank != -1:  # distributed training
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                find_unused_parameters=True
            )

        return dataloader, t_total

    def load_checkpoint(self, dataloader: DataLoader) -> (int, int, int):
        # set global_step to gobal_step of last saved checkpoint from model path
        checkpoint_suffix = self.model_path.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // (len(dataloader) // self.args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
                len(dataloader) // self.args.gradient_accumulation_steps)
        self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        self.logger.info("  Continuing training from epoch %d", epochs_trained)
        self.logger.info("  Continuing training from global step %d", global_step)
        self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        return epochs_trained, global_step, steps_trained_in_current_epoch

    def save_model_checkpoint(self, global_step: int):
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")

        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(checkpoint_dir)

        self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(self.args, os.path.join(checkpoint_dir, "training_args.bin"))
        self.logger.info(f"Saving model checkpoint to {checkpoint_dir}")

        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        self.logger.info(f"Saving optimizer and scheduler states to {checkpoint_dir}")

    def train_step(self, inputs: Dict[str, Any], epoch: int, global_step: int,
                   step: int, tr_loss: float, logging_loss: float) -> (float, float):

        if self.args.model_type in ["roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        outputs = self.model(**inputs)
        loss = outputs[0]

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            global_step += 1

            if self.args.local_rank in [-1,
                                        0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                results = {}
                if self.args.local_rank != -1:
                    self.logger.warning("Set flag --evaluate_during_training, but training on "
                                        "multiple GPUs. Evaluation possible only for single GPU, "
                                        "otherwise metrics may not average well.")
                elif self.args.evaluate_during_training:
                    eval_logging_step = epoch if self.args.evaluate_every_training_epoch else global_step
                    epoch_or_global_step = "epoch" if self.args.evaluate_every_training_epoch else "global_step"

                    results = self.evaluate(compute_metrics=True)

                    self.logger.info(f"Addded evaluation results at {epoch_or_global_step} "
                                     f"{eval_logging_step} to SummaryWriter at {self.log_dir}")

                for key, value in results.items():
                    self.tb_writer.add_scalar(f"eval_{key}", value, global_step)
                self.tb_writer.add_scalar("lr", self.scheduler.get_lr()[0], global_step)
                self.tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.args.logging_steps, global_step)

                logging_loss = tr_loss

        return tr_loss, logging_loss

    def train(self) -> None:
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({self.output_dir}) already exists and is not empty. "
                f"Use --overwrite_output_dir to overcome."
            )

        self.logger.info(f"Training parameters {self.args}")

        dataloader, t_total = self.setup_training()
        self.setup_tensorboard()

        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(self.train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if os.path.exists(self.model_path):
            try:
                epochs_trained, global_step, steps_trained_in_current_epoch = self.load_checkpoint(dataloader)
            except ValueError:
                self.logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(epochs_trained, int(self.args.num_train_epochs),
                                desc="Epoch", disable=self.args.local_rank not in [-1, 0])
        set_seed(self.args.n_gpu)

        for epoch in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                tr_loss, logging_loss = self.train_step(inputs, epoch, global_step, step, tr_loss, logging_loss)

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            self.save_model_checkpoint(epoch + 1)

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        self.logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    def compute_predictions_logits(self, all_results: List[Result]):
        example_index_to_features = defaultdict(list)
        for feature in self.eval_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = OrderedDict()
        all_nbest = OrderedDict()
        scores_diff_json = OrderedDict()

        for (example_index, example) in enumerate(self.eval_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score

            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = _get_best_indexes(result.start_logits, self.args.n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, self.args.n_best_size)

                # get the min score of irrelevant answers
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]

                for start_idx in start_indexes:
                    for end_idx in end_indexes:
                        valid_start_idx = start_idx < len(feature.tokens) and start_idx in feature.token_to_orig_map
                        is_max_start_idx = feature.token_is_max_context.get(start_idx, False)
                        valid_end_idx = len(
                            feature.tokens) > end_idx >= start_idx and end_idx in feature.token_to_orig_map
                        answer_too_long = end_idx - start_idx + 1 > self.args.max_answer_length

                        if valid_start_idx and is_max_start_idx and valid_end_idx and not answer_too_long:
                            prelim_predictions.append(
                                Prediction(
                                    feature_index=feature_index,
                                    start_index=start_idx,
                                    end_index=end_idx,
                                    start_logit=result.start_logits[start_idx],
                                    end_logit=result.end_logits[end_idx],
                                )
                            )

            prelim_predictions.append(
                Prediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
            
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda pred: (pred.start_logit + pred.end_logit),
                reverse=True
            )

            seen_predictions = set()
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) < self.args.n_best_size:
                    feature = features[pred.feature_index]
                    if pred.start_index > 0:  # this is a non-null prediction
                        tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                        orig_doc_start = feature.token_to_orig_map[pred.start_index]
                        orig_doc_end = feature.token_to_orig_map[pred.end_index]
                        orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                        tok_text = self.tokenizer.convert_tokens_to_string(tok_tokens)

                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())
                        orig_text = " ".join(orig_tokens)

                        final_text = get_final_text(pred_text=tok_text, orig_text=orig_text,
                                                    do_lower_case=self.args.do_lower_case,
                                                    verbose_logging=self.args.verbose_logging)
                    else:
                        final_text = None

                    if final_text not in seen_predictions:
                        seen_predictions.add(final_text)
                        nbest.append(
                            Prediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

            # if we didn't include the empty option in the n-best, include it
            if "" not in seen_predictions:
                nbest.append(Prediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, Prediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1, "No valid predictions"

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            for idx in range(len(nbest)):
                nbest[idx].set_probability(probs[idx])

            # predict "" iff the null score - the score of best non-null > threshold
            if not best_non_null_entry:
                all_predictions[example.qas_id] = None
            else:
                score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > 0.0:
                    all_predictions[example.qas_id] = None
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest[example.qas_id] = nbest

        return all_nbest

    def save_predictions(self, predictions) -> None:
        for key, value in predictions.items():
            for idx in range(len(value)):
                predictions[key][idx] = dict(value[idx])

        output_paths = [self.output_dir]
        if self.args.run_end_to_end_prediction:
            output_paths.append(self.data_dir)
        for path in output_paths:
            output_file = os.path.join(path, f"model_predictions_{self.args.set_type}_set.json")
            self.logger.info(f"Writing nbest to: {output_file}")
            with open(output_file, "w") as writer:
                writer.write(json.dumps(predictions, indent=4) + "\n")

    def evaluate(self):
        if not os.path.exists(self.output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.output_dir)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)

        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        self.logger.info(f"***** Running evaluation on {self.args.set_type} set *****")
        self.logger.info("  Num examples = %d", len(self.eval_dataset))
        self.logger.info("  Batch size = %d", self.args.eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.args.model_type in ["roberta", "distilbert", "camembert", "bart", "longformer"]:
                    del inputs["token_type_ids"]

                feature_indices = batch[3]
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = self.eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [output[i].detach().cpu().tolist() for output in outputs.to_tuple()]

                start_logits, end_logits = output
                result = Result(unique_id, start_logits, end_logits)

                all_results.append(result)

        eval_time = timeit.default_timer() - start_time
        self.logger.info(f"  Evaluation done in total {eval_time} secs "
                         f"({eval_time / len(self.eval_dataset)} sec per example)")

        predictions = self.compute_predictions_logits(all_results=all_results)
        self.save_predictions(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--include_ingredients", action="store_true")
    parser.add_argument("--run_end_to_end_prediction", action="store_true")
    args = parser.parse_args()

    qa = ReadingComprehension(
        config_path=args.config_path,
        model_name_or_path=args.model_name_or_path,
        include_ingredients=args.include_ingredients,
        run_end_to_end_prediction=args.run_end_to_end_prediction,
    )

    if qa.args.do_train:
        qa.train()

    if qa.args.do_eval:
        qa.evaluate()
