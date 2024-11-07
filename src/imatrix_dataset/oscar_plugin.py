from datasets import load_dataset
import logging
from plugin_base import DataSourcePluginBase


logger = logging.getLogger(__name__)


class OscarDataSource(DataSourcePluginBase):
    def __init__(self, name="oscar-corpus/oscar", **kwargs):
        super().__init__(name, **kwargs)
        # Define the schema specific to the OSCAR dataset
        self.schema = {
            'content': 'text'  # The 'text' field contains the content
        }

    def load_data(self, lang, num_samples=200, skip_samples=0):
        """
        Loads samples from the OSCAR dataset for a specified language.

        Args:
            lang (str): The language code for the desired dataset subset.
            num_samples (int): Number of samples to load.
            skip_samples (int): Number of samples to skip at the beginning.

        Returns:
            list: A list of data records from the dataset.
        """
        try:
            ds_name = f"unshuffled_deduplicated_{lang}"
            logging.info(f"Loading {num_samples} samples from {self.name}/{ds_name}, skipping {skip_samples} samples.")
            ds = load_dataset(self.name, ds_name, split="train", streaming=True)
            ds = ds.skip(skip_samples).take(num_samples)
            return [entry for entry in ds]  # Return raw records
        except ValueError as e:
            logging.error(f"Error loading language '{lang}' from OSCAR dataset: {e}")
            
            return []
