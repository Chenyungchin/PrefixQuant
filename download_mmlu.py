import os
from datasets import load_dataset, get_dataset_config_names

# 1. Ensure we are saving to scratch (check your path!)
#    This matches the HF_HOME in your run.sh
cache_path = "/scratch/gpfs/NVERMA/jim/datasets"
os.environ["HF_HOME"] = cache_path

dataset_name = "hails/mmlu_no_train"

print(f"--- Backpacking {dataset_name} to {cache_path} ---")

# 2. Get the list of ALL valid configs (abstract_algebra, anatomy, etc.)
#    We exclude 'all' because it is broken.
configs = get_dataset_config_names(dataset_name)
if 'all' in configs:
    configs.remove('all')

print(f"Found {len(configs)} subjects. Starting download...")

# 3. Loop and download each one individually
for i, config in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] Downloading: {config}...")
    try:
        # We only need the 'test' split for MMLU usually
        load_dataset(dataset_name, config, split="test") 
        # If you need validation/dev too, remove the split argument above
    except Exception as e:
        print(f"!!! Failed to download {config}: {e}")

print("\nSuccess! All subjects are cached. You can now submit your job.")