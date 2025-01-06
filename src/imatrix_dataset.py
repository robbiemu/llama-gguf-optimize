import os
import getpass
import heapq
import importlib.util
import llama_cpp 
import logging
import json
import yaml
import random

logger = logging.getLogger(__name__)


def setup_logging(verbosity):
    # Set up logging
    log_level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


def set_env(var: str):
    # Define Environment Configuration
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
    logging.debug(f"Environment variable {var} set.")


def load_plugin_from_file(plugin_path, class_name="DataSourcePlugin"):
    # Load DataSource Plugin from File
    logging.debug(f"Loading plugin from {plugin_path} with class {class_name}.")
    spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
    plugin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_module)
    
    if hasattr(plugin_module, class_name):
        logging.info(f"Successfully loaded plugin class {class_name} from {plugin_path}.")
        return getattr(plugin_module, class_name)
    else:
        raise AttributeError(f"Class {class_name} not found in {plugin_path}")


def load_languages_from_config(config_path):
    # Load Languages from Configuration File
    logging.debug(f"Loading languages from config: {config_path}")
    with open(config_path, 'r') as config_file:
        lines = config_file.readlines()

    start_idx = next((i for i, line in enumerate(lines) if line.strip() == "language:"), None)
    end_idx = next((i for i, line in enumerate(lines[start_idx+1:], start=start_idx+1) if not line.startswith('- ')), len(lines))
    
    language_section = ''.join(lines[start_idx:end_idx])
    config_yaml = yaml.safe_load(language_section)
    languages = config_yaml.get('language', [])
    logging.info(f"Loaded languages: {languages}")
    return languages


def count_existing_entries(file_path):
    # Count entries in an existing JSON sample file
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return len(data)
    return 0


def write_json_samples(lang, entries, overwrite=False):
    # Append new samples to an existing JSON file or create a new JSON list file
    lang_file = f"raw_transactions_{lang}.json"
    
    # If overwrite is set, write new data as a JSON array
    if overwrite or not os.path.exists(lang_file):
        with open(lang_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        logging.info(f"Written {len(entries)} entries to {lang_file} (overwrite={overwrite}).")
    else:
        # Append to existing JSON array if file exists and overwrite is False
        with open(lang_file, "r+", encoding="utf-8") as f:
            existing_data = json.load(f)
            updated_data = existing_data + entries  # Append new entries
            f.seek(0)  # Move to the beginning of the file to overwrite
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
            f.truncate()  # Truncate file to avoid leftover data
        logging.info(f"Appended {len(entries)} entries to {lang_file}.")


def write_combined_dataset(samples, dataset_name, plugin, overwrite=False, shuffle=False, chunk_size=None, model_name=None):
    """
    Write a combined dataset file with language labels and content.

    Args:
        samples (dict): A dictionary with language keys and list of records as values.
        dataset_name (str): Path to the combined dataset file.
        plugin (DataSourcePluginBase): The plugin used to extract content.
        overwrite (bool): Whether to overwrite the file if it exists.
        shuffle (bool): Whether to shuffle the combined data.
        chunk_size (int): Token count for each chunk when shuffling with model.
        model_name (str): Model path for tokenization, required if chunk_size is specified.
    """
    # Ensure the directory path exists
    os.makedirs(os.path.dirname(dataset_name) or '.', exist_ok=True)
    
    # Collect per-language content
    lang_data = {}
    for lang, entries in samples.items():
        lang_data[lang] = []
        for item in entries:
            content = plugin.get_content(item)
            if content:
                lang_data[lang].append(content)
            else:
                logging.warning(f"No content found for language '{lang}' in record: {item}")


import heapq

def write_combined_dataset(samples, dataset_name, plugin, overwrite=False, shuffle=False, chunk_size=None, model_name=None):
    """
    Write a combined dataset file with improved language balance across chunks.
    Uses a sliding window approach with token reservations to ensure better distribution.
    """
    os.makedirs(os.path.dirname(dataset_name) or '.', exist_ok=True)
    
    # Prepare samples with content and tokenization
    sample_counter = 0
    data = []
    for lang, entries in samples.items():
        for item in entries:
            content = plugin.get_content(item)
            if content:
                data.append({
                    'lang': lang,
                    'content': content,
                    'sample_id': sample_counter
                })
                sample_counter += 1
            else:
                logging.warning(f"No content found for language '{lang}' in record: {item}")

    # Tokenize if needed
    if chunk_size and model_name:
        llama = llama_cpp.Llama(model_path=model_name)
        for sample in data:
            tokens = llama.tokenize(sample['content'].encode('utf-8'))
            sample['token_count'] = len(tokens)
    else:
        for sample in data:
            sample['token_count'] = None

    if shuffle:
        random.shuffle(data)

    # If chunking is requested
    if chunk_size and model_name:
        # Group by language
        lang_groups = {}
        for sample in data:
            lang = sample['lang']
            lang_groups.setdefault(lang, []).append(sample)

        # Create priority queues for each language
        lang_queues = {}
        for lang, samples in lang_groups.items():
            deficit = 0
            queue = []
            for sample in samples:
                deficit += (chunk_size - sample['token_count'])
                heapq.heappush(queue, (-deficit, sample['sample_id'], sample))
            lang_queues[lang] = queue

        with open(dataset_name, 'w' if overwrite else 'a', encoding='utf-8') as f:
            while True:
                # Update active languages
                active_langs = {lang for lang in lang_groups if lang_queues[lang]}
                if not active_langs:
                    break

                num_active_langs = len(active_langs)
                target_tokens_per_lang = chunk_size // num_active_langs

                current_chunk = []
                chunk_tokens = 0

                # First pass: try to get ideal distribution
                for lang in active_langs:
                    tokens_in_lang = 0
                    while lang_queues[lang]:
                        priority, _, sample = lang_queues[lang][0]  # Peek at next sample
                        sample_tokens = sample['token_count']
                        if tokens_in_lang + sample_tokens > target_tokens_per_lang:
                            break
                        if chunk_tokens + sample_tokens > chunk_size:
                            break
                        heapq.heappop(lang_queues[lang])
                        current_chunk.append(sample)
                        tokens_in_lang += sample_tokens
                        chunk_tokens += sample_tokens

                # Second pass: fill remaining space with any language
                remaining_space = chunk_size - chunk_tokens
                if remaining_space > 0:
                    combined_queue = []
                    for lang in active_langs:
                        for item in lang_queues[lang]:
                            deficit, sample_id, sample = item
                            sample_tokens = sample['token_count']
                            if sample_tokens <= remaining_space:
                                priority = deficit - (sample_tokens / chunk_size)
                                heapq.heappush(combined_queue, (priority, sample_id, sample))
                    while combined_queue and remaining_space > 0:
                        _, _, sample = heapq.heappop(combined_queue)
                        sample_tokens = sample['token_count']
                        if sample_tokens <= remaining_space:
                            current_chunk.append(sample)
                            remaining_space -= sample_tokens
                            chunk_tokens += sample_tokens
                            # Remove the sample from its original language queue
                            lang_queue = lang_queues[sample['lang']]
                            lang_queue = [(d, sid, s) for (d, sid, s) in lang_queue if s != sample]
                            heapq.heapify(lang_queue)
                            lang_queues[sample['lang']] = lang_queue

                # Write chunk
                if current_chunk:
                    for sample in current_chunk:
                        f.write(f"{sample['lang']}: {sample['content']}\n")
                    f.write('\n')
                else:
                    break

            logging.info(f"Combined dataset with balanced chunks written to {dataset_name}")
    else:
        # Write without chunking
        with open(dataset_name, 'w' if overwrite else 'a', encoding='utf-8') as f:
            for sample in data:
                f.write(f"{sample['lang']}: {sample['content']}\n")
        logging.info(f"Combined dataset written to {dataset_name}")


def report_sample_counts(langs, num_samples):
    # Report sample counts for each language
    for lang in langs:
        lang_file = f"raw_transactions_{lang}.json"
        sample_count = count_existing_entries(lang_file)
        if sample_count < num_samples:
            logging.warning(f"{lang_file} contains only {sample_count} samples, fewer than requested {num_samples}.")
        else:
            logging.info(f"{lang_file} contains {sample_count} samples, meeting the requested count.")


def main(args):    
    # Setup logging based on verbosity
    setup_logging(args.verbosity)
    logging.debug("Logging setup complete.")

    # Example to load environment variable
    set_env("HF_TOKEN")

    # Load languages based on provided argument or config file
    if args.langs:
        langs = args.langs
        logging.info(f"Using languages provided in CLI: {langs}")
    elif args.config:
        langs = load_languages_from_config(args.config)
    else:
        raise ValueError("No languages specified. Please provide either --langs or --config")

    # Report sample counts only if --count_only is set
    if args.count_only:
        report_sample_counts(langs, args.num_samples)
        return  # Exit after reporting counts if --count_only is set

    # Dynamically load the specified plugin
    try:
        PluginClass = load_plugin_from_file(args.datasource_plugin, args.plugin_class)
        if args.url is not None:
            plugin = PluginClass(args.url)
        else:
            plugin = PluginClass()
    except Exception as e:
        logging.error(f"Failed to load plugin: {e}")
        return

    # Initialize samples dict
    samples = {}

    # Execute Task
    for lang in langs:
        lang_file = f"raw_transactions_{lang}.json"
        existing_count = count_existing_entries(lang_file)

        if existing_count >= args.num_samples and not args.overwrite:
            logging.info(f"{lang_file} already contains enough samples ({existing_count}), skipping API call.")
            # Read existing samples
            with open(lang_file, "r", encoding="utf-8") as f:
                samples[lang] = json.load(f)
            continue

        # Calculate remaining samples to download
        remaining_samples = args.num_samples - existing_count
        logging.info(f"Downloading {remaining_samples} samples for {lang}, skipping the first {args.skip_samples + existing_count} entries.")

        # Fetch remaining samples from API
        new_samples = plugin.load_data(
            lang,
            num_samples=remaining_samples,
            skip_samples=args.skip_samples + existing_count
        )

        # Write or append samples to JSON
        write_json_samples(lang, new_samples, overwrite=args.overwrite)

        # Combine existing samples and new samples
        if existing_count > 0 and not args.overwrite:
            with open(lang_file, "r", encoding="utf-8") as f:
                existing_samples = json.load(f)
            samples[lang] = existing_samples + new_samples
        else:
            samples[lang] = new_samples

    # After processing all languages, write combined dataset
    write_combined_dataset(
        samples,
        args.output,
        plugin,
        overwrite=args.overwrite,
        shuffle=args.shuffle,
        chunk_size=args.chunk_size,
        model_name=args.model
    )

    # Final report on sample counts
    report_sample_counts(langs, args.num_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process imatrix datasets. Requires the HF_TOKEN environment variable for authentication. "
                    "If HF_TOKEN is not set, the script will prompt you to enter it."
    )

    parser.add_argument("--output", type=str, default="combined_dataset.json", help="Output file path (default; combined_dataset.json)")
    parser.add_argument("--datasource-plugin", type=str, required=True, help="Path to the data source plugin file.")
    parser.add_argument("--url", type=str, help="Optional source url to pass to the plugin for downloading the datasource (typically a hf dataset name that can be copied from its page)")
    parser.add_argument("--plugin-class", type=str, default="DataSourcePlugin", help="Class name of the data source plugin.")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to load.")
    parser.add_argument("--skip-samples", type=int, default=0, help="Number of samples to skip.")
    parser.add_argument("--langs", type=str, nargs='*', help="Specify languages as a space-separated list.")
    parser.add_argument("--config", type=str, help="Path to configuration file for loading languages")
    parser.add_argument("--write-raw-samples", action="store_true", help="Write raw language samples to files if this flag is set.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if present.")
    parser.add_argument("--count-only", action="store_true", help="Only count samples in existing files without downloading.")
    parser.add_argument("--verbosity", type=str, default="INFO", help="Logging verbosity level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the combined dataset when writing the output. Only available when writing the combined output.")
    parser.add_argument("--chunk-size", type=int, help="Token count for each chunk when shuffling with model. Requires --shuffle and --model.")
    parser.add_argument("--model", type=str, help="Model path for tokenization. Requires --shuffle. Required if --chunk-size is specified.")

    args = parser.parse_args()

    if args.chunk_size and (not args.shuffle or not args.model):
        parser.error("--chunk-size requires both --shuffle and --model to be specified.")

    if args.shuffle and (not args.chunk_size or not args.model):
        parser.error("--chunk-size requires both --shuffle and --model to be specified.")

    main(args)
