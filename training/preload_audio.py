import yaml
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
import soundfile as sf
from transformers import AutoFeatureExtractor
import os
import torch
import torchaudio
from pathlib import Path
import shutil

output_dir = 'preloaded_audio'

# Load the YAML configuration
with open('args.yaml', 'r') as file:
    config = yaml.safe_load(file)

num_proc = 28
shard_size = 2000

feature_extractor = AutoFeatureExtractor.from_pretrained(
    config.get('feature_extractor_name', None),
    cache_dir=config.get('cache_dir', None),
    token=config.get('token', None),
    trust_remote_code=config.get('trust_remote_code', None),
)
sampling_rate = feature_extractor.sampling_rate
audio_column_name = config.get('target_audio_column_name', None)

def process_audio(batch):
    results = {audio_column_name: [], "sampling_rate": [], "duration": []}
    for path in batch["audio"]:
        try:
            array, sr = sf.read(path)
            duration = len(array) / sr
            if sr != sampling_rate:
                array = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)(
                    torch.tensor(array).float()
                ).numpy()
            array = array.astype('float32')
            results[audio_column_name].append({"array": array})
            results["sampling_rate"].append(sampling_rate)
            results["duration"].append(duration)
        except Exception as e:
            # Ensure that all lists in results have the same length
            results[audio_column_name].append(None)
            results["sampling_rate"].append(None)
            results["duration"].append(None)
            continue
    return results

def shard_and_process(dataset, dataset_name, split, output_dir, shard_size):
    output_path = Path(output_dir) / split / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # tmp_path = Path(output_dir) / dataset_name / split / "_tmp"
    # tmp_path.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    num_shards = (len(dataset) + shard_size - 1) // shard_size

    shards = []
    for shard_idx in range(num_shards):
        shard_path = Path(output_path) / f"shard_{shard_idx}"
        if shard_path.exists():
            print(f"Shard {shard_idx} already exists, skipping")
            continue
        shard = dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True, keep_in_memory=True)

        shard = shard.map(
            process_audio,
            num_proc=num_proc,
            batched=True,
            batch_size=2,
            keep_in_memory=True,
            desc=f"Processing shard {shard_idx + 1}/{num_shards}"
        )

        shard = shard.filter(
            lambda example: example[audio_column_name] is not None, 
            num_proc=num_proc,
            keep_in_memory=True,
        )

        shard.save_to_disk(str(shard_path))
        # shards.append(shard)
        total_examples += len(shard)

    
    # Concatenate all shards and save to disk as a single dataset
    # full_dataset = concatenate_datasets(shards)
    # full_dataset.save_to_disk(output_path)

    # Load each shard, concatenate them, and save as a single dataset
    # ds = concatenate_datasets([
    #     load_from_disk(f"{tmp_path}/_tmp_shard_{shard_idx}")
    #     for shard_idx in range(num_shards)
    # ])

    # Save the concatenated dataset
    # ds.save_to_disk(output_path)

    # Delete the tmp_path directory after concatenation
    # if os.path.exists(tmp_path):
    #     shutil.rmtree(tmp_path)

    return total_examples

# Process train datasets
train_dataset_names = config['train_dataset_name'].split('+')
train_split_names = config['train_split_name'].split('+')

total_train_examples = 0
for train_dataset_name, train_split_name in zip(train_dataset_names, train_split_names):
    print(f"Processing train dataset: {train_dataset_name} with split: {train_split_name}")
    train_dataset = load_dataset(train_dataset_name, split=train_split_name)
    total_train_examples += shard_and_process(train_dataset, train_dataset_name, "train", output_dir, shard_size)

# Process eval datasets
eval_dataset_names = config['eval_dataset_name'].split('+')
eval_split_names = config['eval_split_name'].split('+')

total_eval_examples = 0
for eval_dataset_name, eval_split_name in zip(eval_dataset_names, eval_split_names):
    print(f"Processing eval dataset: {eval_dataset_name} with split: {eval_split_name}")
    eval_dataset = load_dataset(eval_dataset_name, split=eval_split_name)
    total_eval_examples += shard_and_process(eval_dataset, eval_dataset_name, "eval", output_dir, shard_size)

# Print the number of examples in each dataset
print(f"Number of training examples: {total_train_examples}")
print(f"Number of evaluation examples: {total_eval_examples}")

# Example of loading the dataset directly
# train_dataset = load_from_disk(Path(output_dir) / train_dataset_names[0] / "train")
# eval_dataset = load_from_disk(Path(output_dir) / eval_dataset_names[0] / "eval")
