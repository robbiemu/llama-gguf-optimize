import json
import logging
import os
from imatrix_dataset.plugin_base import DataSourcePluginBase

logger = logging.getLogger(__name__)

class LocalShuffleDataSource(DataSourcePluginBase):
    def __init__(self, name="local-shuffle", **kwargs):
        super().__init__(name, **kwargs)
        self.schema = {
            'content': 'text'  # Match the schema used by other plugins
        }

    def load_data(self, lang, num_samples=200, skip_samples=0):
        """
        Loads samples from local raw transaction files.

        Args:
            lang (str): The language code to load from raw_transactions_{lang}.json
            num_samples (int): Number of samples to load
            skip_samples (int): Number of samples to skip (maintained for API compatibility)

        Returns:
            list: List of samples from the raw transaction file
        """
        filename = f"raw_transactions_{lang}.json"
        
        try:
            if not os.path.exists(filename):
                logger.error(f"File not found: {filename}")
                return []

            with open(filename, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            # Apply skip and limit
            return samples[skip_samples:skip_samples + num_samples]
            
        except Exception as e:
            logger.error(f"Error loading language '{lang}' from {filename}: {e}")
            return []