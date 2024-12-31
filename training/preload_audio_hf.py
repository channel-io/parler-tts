import yaml
from datasets import load_dataset, load_from_disk, Audio
from transformers import AutoFeatureExtractor
import os
import sys
from pathlib import Path

output_dir = 'preloaded_audio'

# Load the YAML configuration
with open(sys.argv[1], 'r') as file:
    config = yaml.safe_load(file)

num_proc = 320
shard_size = 2000

feature_extractor = AutoFeatureExtractor.from_pretrained(
    config.get('feature_extractor_name', None),
    cache_dir=config.get('cache_dir', None),
    token=config.get('token', None),
    trust_remote_code=config.get('trust_remote_code', None),
)
sampling_rate = feature_extractor.sampling_rate
audio_column_name = config.get('target_audio_column_name', None)

def shard_and_process(dataset, dataset_name, split, output_dir, shard_size):
    output_path = os.path.join(output_dir, split, dataset_name.replace("/", "_"))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    total_examples = 0
    num_shards = (len(dataset) + shard_size - 1) // shard_size

    print(f"Number of shards: {num_shards}")

    for shard_idx in range(num_shards):
        shard_path = Path(output_path) / f"shard_{shard_idx}"
        if shard_path.exists():
            print(f"Shard {shard_idx} already exists, skipping")
            continue
        shard = dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True, keep_in_memory=True)

        shard = shard.cast_column(audio_column_name, Audio(sampling_rate=sampling_rate))

        shard = shard.filter(
            lambda example: example[audio_column_name] is not None, 
            num_proc=num_proc,
            keep_in_memory=True,
        )

        shard.save_to_disk(str(shard_path), num_proc=num_proc)
        total_examples += len(shard)

    return total_examples

# Process train datasets
train_dataset_names = config['train_dataset_name'].split('+')
train_split_names = config['train_split_name'].split('+')

# total_train_examples = 0
# for train_dataset_name, train_split_name in zip(train_dataset_names, train_split_names):
#     print(f"Processing train dataset: {train_dataset_name} with split: {train_split_name}")
#     try:
#         train_dataset = load_dataset(train_dataset_name, split=train_split_name)
#     except ValueError:
#         train_dataset = load_from_disk(train_dataset_name)
#         train_dataset = train_dataset[train_split_name]
#     total_train_examples += shard_and_process(train_dataset, train_dataset_name, "train", output_dir, shard_size)

# Process eval datasets
eval_dataset_names = config['eval_dataset_name'].split('+')
eval_split_names = config['eval_split_name'].split('+')

total_eval_examples = 0
for eval_dataset_name, eval_split_name in zip(eval_dataset_names, eval_split_names):
    print(f"Processing eval dataset: {eval_dataset_name} with split: {eval_split_name}")
    try:
        eval_dataset = load_dataset(eval_dataset_name, split=eval_split_name)
    except ValueError:
        eval_dataset = load_from_disk(eval_dataset_name)
        eval_dataset = eval_dataset[eval_split_name]
    total_eval_examples += shard_and_process(eval_dataset, eval_dataset_name, "eval", output_dir, shard_size)

# Print the number of examples in each dataset
# print(f"Number of training examples: {total_train_examples}")
print(f"Number of evaluation examples: {total_eval_examples}")
