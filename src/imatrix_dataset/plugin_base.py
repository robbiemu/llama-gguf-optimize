from abc import ABC, abstractmethod


class DataSourcePluginBase(ABC):
    """
    Abstract base class for data source plugins.
    Defines the interface that all data source plugins must implement.
    """

    def __init__(self, name, **kwargs):
        """
        Initialize the data source plugin.

        Args:
            name (str): Name of the dataset or data source.
            kwargs: Additional arguments for initializing the plugin.
        """
        self.name = name
        self.kwargs = kwargs
        self.schema = kwargs.get('schema', {'content': 'text'})  # Default schema with 'content' field

    @abstractmethod
    def load_data(self, lang, num_samples=200, skip_samples=0):
        """
        Load data samples from the data source.

        Args:
            lang (str): The language code or specific identifier for the dataset subset.
            num_samples (int): Number of samples to load.
            skip_samples (int): Number of samples to skip at the beginning.

        Returns:
            list: A list of dictionaries, each containing the data sample.
        """
        pass

    def get_content(self, record):
        """
        Extracts the content from a record based on the 'content' field in the schema.

        Args:
            record (dict): A single data record.

        Returns:
            str: The content extracted from the record.
        """
        content_path = self.schema.get('content', '')

        return self.get_value_from_path(record, content_path)

    def get_value_from_path(self, data, path):
        """
        Helper method to get a value from a nested dictionary or list using a path.
        
        Args:
            data (dict or list): The data structure to traverse.
            path (str or list): The path to the value as a dot-separated string or list of keys/indices.

        Returns:
            The value at the specified path or None if not found.
        """
        # If path is a string, convert it to a list (for compatibility)
        if isinstance(path, str):
            path = path.split(".")

        for key in path:
            # If key is an integer (for lists), try to access the index
            if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                key = int(key)
                if isinstance(data, list) and 0 <= key < len(data):
                    data = data[key]
                else:
                    return None  # Index out of bounds or not a list
            elif isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return None  # Key not found in dict or improper structure

        return data
