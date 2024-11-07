import os
import getpass
import importlib.util
import logging
import json
import yaml


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


import os

def write_combined_dataset(samples, dataset_name, plugin, overwrite=False):
    """
    Write a combined dataset file with language labels and content.

    Args:
        samples (dict): A dictionary with language keys and list of records as values.
        dataset_name (str): Path to the combined dataset file.
        plugin (DataSourcePluginBase): The plugin used to extract content.
        overwrite (bool): Whether to overwrite the file if it exists.
    """
    # Ensure the directory path exists
    os.makedirs(os.path.dirname(dataset_name), exist_ok=True)
    
    # Determine write mode based on overwrite flag
    mode = 'w' if overwrite else 'a'
    
    with open(dataset_name, mode, encoding="utf-8") as imatrix_file:
        for lang, entries in samples.items():
            for item in entries:
                # Use the plugin's get_content method to extract the content
                content = plugin.get_content(item)
                if content:
                    imatrix_file.write(f"{lang}: {content}\n")
                else:
                    logging.warning(f"No content found for language '{lang}' in record: {item}")

    logging.info(f"Combined dataset written to {dataset_name} (overwrite={overwrite})")


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
        plugin = PluginClass()
    except Exception as e:
        logging.error(f"Failed to load plugin: {e}")
        return
    
    # Execute Task
    samples = {}
    for lang in langs:
        lang_file = f"raw_transactions_{lang}.json"
        existing_count = count_existing_entries(lang_file)

        if existing_count >= args.num_samples and not args.overwrite:
            logging.info(f"{lang_file} already contains enough samples ({existing_count}), skipping API call.")
            continue

        # Calculate remaining samples to download
        remaining_samples = args.num_samples - existing_count
        logging.info(f"Downloading {remaining_samples} samples for {lang}, skipping the first {args.skip_samples + existing_count} entries.")

        # Fetch remaining samples from API
        samples[lang] = plugin.load_data(
            lang,
            num_samples=remaining_samples,
            skip_samples=args.skip_samples + existing_count
        )

        # Write or append samples to JSON and output
        write_json_samples(lang, samples[lang], overwrite=args.overwrite)
        write_combined_dataset(samples, args.dataset_name, plugin, overwrite=args.overwrite)

        # Final report on sample counts
        report_sample_counts(langs, args.num_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process imatrix datasets. Requires the HF_TOKEN environment variable for authentication. "
                    "If HF_TOKEN is not set, the script will prompt you to enter it."
    )

    parser.add_argument("--output", type=str, default="combined_dataset.json", help="Output file path (default; combined_dataset.json)")
    parser.add_argument("--datasource_plugin", type=str, required=True, help="Path to the data source plugin file.")
    parser.add_argument("--plugin_class", type=str, default="DataSourcePlugin", help="Class name of the data source plugin.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to load.")
    parser.add_argument("--skip_samples", type=int, default=1000, help="Number of samples to skip.")
    parser.add_argument("--langs", type=str, nargs='*', help="Specify languages as a space-separated list.")
    parser.add_argument("--config", type=str, help="Path to configuration file for loading languages")
    parser.add_argument("--write_raw_samples", type=bool, default=True, help="Whether to write raw language samples to files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if present.")
    parser.add_argument("--count_only", action="store_true", help="Only count samples in existing files without downloading.")
    parser.add_argument("--verbosity", type=str, default="INFO", help="Logging verbosity level: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    args = parser.parse_args()

    main(args)
