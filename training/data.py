import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset, Audio, load_from_disk
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer
from torch.utils.data import Sampler


@dataclass
class DataCollatorEncodecWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch or
    to `max_length` if `max_length` is set and `padding=max_length`.
    """

    feature_extractor: AutoFeatureExtractor
    audio_column_name: str
    feature_extractor_input_name: Optional[str] = "input_values"
    max_length: Optional[int] = None
    padding: Optional[str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        audios = [feature[self.audio_column_name]["array"] for feature in features]
        len_audio = [len(audio) for audio in audios]
        if self.max_length is not None:
            audios = [audio[: min(l, self.max_length)] for audio, l in zip(audios, len_audio)]

        # since resampling has already been performed in the 'load_multiple_datasets' function,
        # a fixed sampling_rate(44100hz) is passed to the feature_extractor.
        sampling_rate = self.feature_extractor.sampling_rate
        batch = self.feature_extractor(
            audios, sampling_rate=sampling_rate, return_tensors="pt", padding=self.padding, max_length=self.max_length
        )
        batch["len_audio"] = torch.tensor(len_audio).unsqueeze(1)
        return batch


@dataclass
class DataCollatorParlerTTSWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        prompt_tokenizer (:class:`~transformers.AutoTokenizer`)
            The prompt_tokenizer used for proccessing the data.
        description_tokenizer (:class:`~transformers.AutoTokenizer`)
            The description_tokenizer used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    prompt_tokenizer: AutoTokenizer
    description_tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    prompt_max_length: Optional[int] = None
    description_max_length: Optional[int] = None
    audio_max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        labels = [torch.tensor(feature["labels"]).transpose(0, 1) for feature in features]
        # (bsz, seq_len, num_codebooks)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        if self.audio_max_length is not None and self.padding == "max_length":
            labels = torch.nn.functional.pad(
                labels, pad=(0, 0, 0, max(self.audio_max_length - labels.shape[1], 0)), value=-100
            )

        if "input_ids" in features[0]:
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        else:
            input_ids = None

        if input_ids is not None:
            input_ids = self.description_tokenizer.pad(
                input_ids,
                return_tensors="pt",
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.description_max_length,
            )

            batch = {"labels": labels, **input_ids}
        else:
            batch = {"labels": labels, "input_ids": None, "attention_mask": None}

        prompt_input_ids = [{"input_ids": feature["prompt_input_ids"]} for feature in features]
        prompt_input_ids = self.prompt_tokenizer.pad(
            prompt_input_ids,
            return_tensors="pt",
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.prompt_max_length,
        )

        batch["prompt_input_ids"] = prompt_input_ids["input_ids"]
        if "attention_mask" in prompt_input_ids:
            batch["prompt_attention_mask"] = prompt_input_ids["attention_mask"]

        return batch


class DynamicSizeBatchSampler(Sampler):
    def __init__(self, lengths, batch_size_map):
        self.lengths = lengths
        self.batch_size_map = sorted(batch_size_map, key=lambda x: x[1])
        self.batches = self._create_batches()

    def _create_batches(self):
        indices = sorted(range(len(self.lengths)), key=lambda x: self.lengths[x])
        batches = []
        for batch_size, max_len in self.batch_size_map:
            batch = []
            for idx in indices:
                if self.lengths[idx] <= max_len:
                    batch.append(idx)
                    if len(batch) == batch_size:
                        batches.append(batch)
                        batch = []
            indices = [idx for idx in indices if self.lengths[idx] > max_len]
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    metadata_dataset_names=None,
    splits=None,
    dataset_samples=None,
    default_split="train",
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        if dataset_config_names is None:
            dataset_config_names = [None] * len(dataset_names)
        else:
            dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None
        if metadata_dataset_names is None:
            metadata_dataset_names = [None] * len(dataset_names)
        else:
            metadata_dataset_names = metadata_dataset_names.split("+")

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if metadata_dataset_names is not None and len(metadata_dataset_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one metadata dataset is passed for each dataset, got {len(dataset_names)} datasets and {len(metadata_dataset_names)} metadata datasets."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "metadata_dataset_name": metadata_dataset_names[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict


def load_multiple_datasets(
    accelerator: Accelerator,
    dataset_names: Union[List, str],
    dataset_config_names: Union[List, str],
    metadata_dataset_names: Optional[str] = None,
    splits: Optional[Union[List, str]] = None,
    label_column_names: Optional[List] = None,
    stopping_strategy: Optional[str] = "first_exhausted",
    dataset_samples: Optional[Union[List, np.array]] = None,
    streaming: Optional[bool] = False,
    seed: Optional[int] = None,
    id_column_name: Optional[str] = None,
    columns_to_keep: Optional[Set[str]] = None,
    prompt_column_name: Optional[str] = None,
    sampling_rate: Optional[int] = None,
    audio_column_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    dataset_names_dict = convert_dataset_str_to_list(
        dataset_names, dataset_config_names, metadata_dataset_names, splits, label_column_names, dataset_samples
    )

    if dataset_samples is not None:
        dataset_samples = [ds_dict["samples"] for ds_dict in dataset_names_dict]
        probabilities = np.array(dataset_samples) / np.sum(dataset_samples)
    else:
        probabilities = None
        
    if streaming and 'num_proc' in kwargs:
        # kwargs['num_workers'] = kwargs['num_proc']
        del kwargs['num_proc']

    all_datasets = []
    sharded = kwargs.get('sharded', False)
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(dataset_names_dict, desc="Combining datasets..."):
        with accelerator.main_process_first():
            if sharded:
                logger.info(f"Sharded dataset: {dataset_dict['name']}")
                dataset = []
                shards = os.listdir(dataset_dict["name"])
                for shard in shards:
                    dataset.append(load_from_disk(
                        os.path.join(dataset_dict["name"], shard),
                    ))
                dataset = concatenate_datasets(dataset)
            else:
                dataset = load_dataset(
                    dataset_dict["name"],
                    dataset_dict["config"],
                    split=dataset_dict["split"],
                    streaming=streaming,
                    **kwargs,
                )
            logger.info(f"Loaded dataset: {dataset}")
            # dataset = dataset.shuffle(seed=42)
            # min_samples = min(1000, len(dataset))
            # dataset = dataset.select(range(max(min_samples, int(len(dataset) * 0.5))))
            
            dataset_features = dataset.features.keys()
            
            # Check if 'audio' column is a string and convert it to an array
            # if 'audio' in dataset.column_names and isinstance(dataset[0]['audio'], str):
            #     # Cast the 'audio' column to Audio feature
            #     dataset = dataset.cast_column("audio", Audio())
            cache_dir = "/home/work/channel/cache_directory"
            cache_file_name = os.path.join(cache_dir, "cached_dataset.arrow")
            os.makedirs(cache_dir, exist_ok=True)
            num_proc = 32
            if 'audio' in dataset.column_names and isinstance(dataset[0]['audio'], str):
                import soundfile as sf
                import librosa

                def process_audio(batch):
                    results = {audio_column_name: [], "sampling_rate": [], "duration": []}
                    for path in batch["audio"]:
                        try:
                            array, sr = sf.read(path)
                            duration = len(array) / sr
                            if sr != sampling_rate:
                                array = librosa.resample(array, orig_sr=sr, target_sr=sampling_rate)
                            results[audio_column_name].append({"array": array})
                            results["sampling_rate"].append(sampling_rate)
                            results["duration"].append(duration)
                        except Exception:
                            results[audio_column_name].append({"array": None})
                            results["sampling_rate"].append(None)
                            results["duration"].append(None)
                    return results

                dataset = dataset.map(
                    process_audio,
                    num_proc=num_proc,
                    batched=True,
                    batch_size=16,
                    writer_batch_size=10,
                    cache_file_name=cache_file_name,
                    desc="Loading audio"
                )
                
                dataset = dataset.filter(lambda example: example[audio_column_name]['array'] is not None, num_proc=num_proc, writer_batch_size=10)

            # if 'audio' in dataset.column_names:
            #     print("First item in 'audio' column:", dataset[0]['audio'])
            
            if isinstance(dataset[0]['audio'], str) and sampling_rate is not None and audio_column_name is not None:
                # resample target audio
                logger.info(f"Resampling audio to {sampling_rate} Hz")
                dataset = dataset.cast_column(audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate))

            metadata_dataset_name = dataset_dict["metadata_dataset_name"]
            if metadata_dataset_name is not None:
                logger.info(
                    f'Merging {dataset_dict["name"]} - {dataset_dict["split"]} with {metadata_dataset_name} - {dataset_dict["split"]}'
                )
                metadata_dataset = load_dataset(
                    metadata_dataset_name,
                    dataset_dict["config"],
                    split=dataset_dict["split"],
                    streaming=streaming,
                    **kwargs,
                )

                # TODO(YL): I forgot to create unique ids for MLS english.
                # To iterate faster, I bypass the original id check and do another one. - Done once because assuming it won't change next time
                # if dataset_dict["name"] == "parler-tts/mls_eng_10k":
                #     def concat_ids(book_id, speaker_id, begin_time):
                #         return {"id": f"{book_id}_{speaker_id}_{str(begin_time).replace('.', '_')}"}
                #     dataset = dataset.map(concat_ids, input_columns=["book_id", "speaker_id", "begin_time"], num_proc=24)
                #     metadata_dataset = metadata_dataset.map(concat_ids, input_columns=["book_id", "speaker_id", "begin_time"], num_proc=24)
                #     metadata_dataset = metadata_dataset.rename_column(id_column_name, f"metadata_{id_column_name}")

                if dataset_dict["name"] not in {"parler-tts/mls_eng_10k", "parler-tts/mls_eng"}:
                    if id_column_name is not None and id_column_name not in dataset.column_names:
                        raise ValueError(
                            f"id_column_name={id_column_name} but has not been found in the dataset columns"
                            f"- one of {', '.join(list(dataset.column_names))}."
                        )
                    if id_column_name is not None and id_column_name not in metadata_dataset.column_names:
                        raise ValueError(
                            f"id_column_name={id_column_name} but has not been found in the metadata dataset columns"
                            f"- one of {', '.join(list(metadata_dataset.column_names))}."
                        )
                    elif id_column_name is not None:
                        metadata_dataset = metadata_dataset.rename_column(id_column_name, f"metadata_{id_column_name}")

                metadata_columns_to_remove = set(metadata_dataset.column_names).intersection(set(dataset.column_names))

                if prompt_column_name is not None:
                    # We might have applied some transformations to the prompts (e.g  punctuation restoration)
                    # so we make sure to remove it from the original dataset
                    if prompt_column_name in dataset.column_names:
                        logger.info(
                            f"REMOVE {prompt_column_name} from dataset {dataset_dict['name']} - dataset_dict['split']"
                        )
                        dataset.remove_columns(prompt_column_name)

                metadata_columns_to_remove = set(metadata_dataset.column_names).intersection(set(dataset.column_names))
                metadata_dataset = metadata_dataset.remove_columns(metadata_columns_to_remove)

                dataset = concatenate_datasets([dataset, metadata_dataset], axis=1)

                if id_column_name is not None and dataset_dict["name"] not in {
                    "parler-tts/mls_eng_10k",
                    "parler-tts/mls_eng",
                }:
                    if (
                        len(
                            dataset.filter(
                                lambda id1, id2: id1 != id2,
                                input_columns=[id_column_name, f"metadata_{id_column_name}"],
                            )
                        )
                        != 0
                    ):
                        raise ValueError(
                            f"Concatenate didn't work. Some ids don't correspond on dataset {dataset_dict['name']}"
                        )

                dataset_features = dataset.features.keys()
            logger.info(f"Dataset features: {dataset_features}")
            # else:
            #     def add_none_description(sample):
            #         sample['description'] = None
            #         return sample
                
            #     dataset = dataset.map(add_none_description, num_proc=15, desc="Adding None Description")

            if columns_to_keep is not None:
                logger.info(f"Removing columns: {set(dataset_features - columns_to_keep)}")
                dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))
                
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        # we have a single dataset so just return it as is
        return all_datasets[0]

    if streaming:
        interleaved_dataset = interleave_datasets(
            all_datasets,
            stopping_strategy=stopping_strategy,
            probabilities=probabilities,
            seed=seed,
        )
    else:
        with accelerator.local_main_process_first():
            interleaved_dataset = concatenate_datasets(all_datasets)

    return interleaved_dataset
