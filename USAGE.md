# Iterative Quantization and Comparison

This guide outlines a systematic approach for quantizing your models using iterative refinement of the importance matrix (I-matrix) and evaluating quantization quality through KL-divergence metrics. By following these steps, you can optimize your model quantization to achieve minimal performance loss across diverse data subsets.

---

### **Preparing Your Dataset for I-Matrix Generation**

Before quantizing your model, you need a dataset to generate the I-matrix and evaluate the quantized models. The `imatrix_dataset.py` tool helps collect and preprocess this data. Thanks to the efficient caching mechanism of the datasets library, you can start with a smaller dataset for initial testing and gradually expand it as needed for I-matrix generation and model evaluation. The tool's skip functionality allows you to add more data later, providing flexibility in your workflow. This approach leverages the local caching of the datasets library, ensuring efficient data access without the need for large upfront downloads.

The section you've provided needs significant revision to align with the actual behavior of the datasets library and the functionality of the imatrix_dataset.py tool. Here's a suggested update that reflects a more efficient and flexible approach:

#### Optimizing Dataset Collection for I-matrix Generation

To optimize the process of collecting data for I-matrix generation and model evaluation:

1. **Start Small**: Begin with a smaller dataset for initial testing. The imatrix_dataset.py tool allows you to specify the number of samples to collect using the `--num-samples` argument[1].

2. **Incremental Expansion**: As you need more data, you can incrementally expand your dataset. The tool supports skipping samples with the `--skip-samples` argument, allowing you to add new data without duplicating existing samples[1].

3. **Efficient Caching**: The datasets library automatically caches downloaded data locally. This means subsequent accesses to the same dataset will be much faster and won't require re-downloading[1].

4. **Flexible Data Management**: The tool writes individual language samples to separate JSON files (e.g., `raw_transactions_{lang}.json`). It also creates a combined dataset file, which can be shuffled and chunked if desired[1].

5. **Automated Sample Counting**: Use the `--count-only` flag to check how many samples you already have without downloading more data[1].

6. **Smart Overwrite Control**: The `--overwrite` flag allows you to control whether existing data should be replaced or appended to[1].

This approach leverages the built-in efficiencies of the datasets library and the flexibility of the imatrix_dataset.py tool. It allows you to start small, expand as needed, and maintain an efficient workflow without unnecessary downloads or data redundancy.

#### Generate the Dataset using `imatrix_dataset.py`

  The `imatrix_dataset.py` script allows you to create a dataset tailored to your needs. It supports various data sources through a flexible plugin system (details on the plugin system are provided later in this guide).

  The `imatrix_dataset.py` script relies on a plugin system to handle different data sources flexibly. This allows you to use data from various sources, such as Hugging Face datasets, OSCAR corpus, or your custom data sources.

  **Example command:**

  ```sh
  ❯ uv run src/imatrix_dataset.py \
      --datasource_plugin src/imatrix_dataset/hf_dataset.py \
      --plugin_class HFDatasetPlugin \
      --langs en \
      --num_samples 10000 \
      --output /path/to/dataset.txt
  ```

  - The `--datasource_plugin` parameter specifies the path to the data source plugin script.
  - The `--plugin_class` parameter specifies the class name within the plugin script.
  - Adjust the `--langs` parameter to include the languages relevant to your dataset (e.g., `en`, `es`, `de`).
  - The `--num_samples` parameter specifies the number of samples to use.
  - The `--output` parameter specifies the path where the generated dataset will be saved.

#### **Generating Multiple Quantizations with `quantize.py`**

After preparing your dataset and generating the I-matrix, you may want to create multiple quantized versions of your model simultaneously. The `quantize.py` script facilitates this by allowing you to specify multiple quantization types in a single command, ensuring consistency across all quantized models.

##### **Benefits of Generating Multiple Quantizations at Once:**

- **Consistency:** Ensures that all quantizations are created using the same settings and parameters.
- **Efficiency:** Saves time by processing multiple quantizations in a single run rather than executing separate commands for each.
- **Organization:** Centralizes the quantization process, making it easier to manage and track different quantized versions.

##### **Using `quantize.py` to Generate Multiple Quantizations:**

You can specify multiple quantization types using a comma-separated list in the `--quantizations` parameter. The script will process each specified quantization sequentially, saving each quantized model to the designated output directory.

**Example Command:**

```sh
❯ uv run src/quantize.py quantize \
    --model-name my_model \
    --base-model /path/to/baseline_model.gguf \
    --quantizations Q4_0,Q6_K,Q8 \
    --imatrix-path /path/to/imatrix.bin \
    --output-dir /path/to/output_dir
```

- `--model-name`: A base name for your quantized models. The script will append the quantization type to this name for each output model (e.g., `my_model-Q4_0.gguf`).
- `--base-model`: Path to your unquantized (baseline) model in GGUF format.
- `--quantizations`: A comma-separated list of quantization types you wish to generate (e.g., `q4_0,q4_1,q5_0`).
- `--imatrix-path`: Path to the I-matrix file if you've previously generated.
- `--output-dir`: Directory where the quantized models will be saved.

---

### **Step 1: Generate and Store Baseline Logits (One-Time Setup)**

- **Run `generate_logits.py` on the Baseline Model**

  Begin by generating logits from your unquantized (baseline) model over the dataset you prepared. Use the `generate_logits.py` script to create a reusable HDF5 file containing these logits. Generating the baseline logits once saves time and storage in subsequent comparisons.

  **Example command:**

  ```sh
  ❯ uv run src/generate_logits.py \
      --model /path/to/baseline_model.gguf \
      --context-size <size, ie 2048> \
      --dataset /path/to/dataset.txt \
      --output baseline_logits.hdf5
  ```

  - Replace `/path/to/baseline_model.gguf` with the path to your unquantized model in GGUF format, and <size> with your model's context size.
  - The `--dataset` parameter uses the dataset generated by `imatrix_dataset.py` in the previous step.
  - The `--output` parameter specifies the name of the output HDF5 file.

  _**notes:**_ 
  - _If working with large datasets, you can use the `--from` and/or `--to` arguments to process the dataset in resumable chunks, allowing you to generate logits gradually without starting over each time._
  - _At this time, **llama-cpp-python** library will default to 512 context size, rather than reading it from the underlying model, so a choice was made to make `--context-size` required._


- **Save Baseline Logits for Reference**

  Store the generated `baseline_logits.hdf5` file in a secure and accessible location. This file will serve as the reference point for comparing outputs from quantized models in later steps.

---

### **Step 2: Initial Quantization (Without I-Matrix) and Baseline Comparison**

- **Quantize the Model without an I-Matrix using `quantize.py`**

  Create an initial quantized version of your model without using an I-matrix. Use the `quantize.py` script to perform the quantization, which streamlines the process and allows for consistent quantization settings.

  **Example command:**

  ```sh
  ❯ uv run src/quantize.py quantize \
      --model-name my_model \
      --base-model /path/to/baseline_model.gguf \
      --quantizations q4_0 \
      --output-dir /path/to/output_dir
  ```

  - Replace `my_model` with a name for your model.
  - Replace `/path/to/baseline_model.gguf` with the path to your unquantized model.
  - The `--quantizations` parameter specifies the quantization type(s); here, `q4_0` is used.
  - The `--output-dir` parameter specifies where the quantized model will be saved.

- **Run `kl_d_bench.py` for Initial Comparison**

  Use the `kl_d_bench.py` script to compare the logits of the quantized model against the baseline logits. This script processes the stored baseline logits and computes KL-divergence metrics efficiently.

  **Example command:**

  ```sh
  ❯ uv run src/kl_d_bench.py \
      --baseline-logits baseline_logits.hdf5 \
      --target-model /path/to/output_dir/my_model-q4_0.gguf \
      --dataset /path/to/dataset.txt \
      --output-file initial_comparison.hdf5
  ```

  - Replace `/path/to/output_dir/my_model-q4_0.gguf` with the path to your quantized model.
  - The `--output-file` parameter specifies where to save the comparison results.

  The `--early-stopping` flag enables the early stopping mechanism, allowing the script to terminate the comparison process once sufficient statistical evidence is gathered. In subsequent tests, conforming all future tests to the number of chunks found with `--to-chunk` can greatly reduce the amount of data written to disk.

  **Example command with Early Stopping:**

  ```sh
  ❯ uv run src/kl_d_bench.py \
      --baseline-logits baseline_logits.hdf5 \
      --target-model /path/to/output_dir/my_model-q4_0.gguf \
      --dataset /path/to/dataset.txt \
      --output-file initial_comparison.hdf5 \
      --early-stopping
  ```

- **Collect and Evaluate Metrics**

  After running `kl_d_bench.py`, review the KL-divergence metrics, including the median and the 90th, 95th, and 99th percentiles for each data chunk. This initial assessment serves as a reference point for evaluating improvements from I-matrix calibration in subsequent iterations.

---

### **Step 3: Iterative I-Matrix Calibration and Quantization**

- **Generate I-Matrices with Incrementally Larger Dataset Subsets**

  Begin refining the I-matrix by generating it using a small subset of your dataset. If you haven't already, use the `imatrix_dataset.py` script to create the I-matrix tailored to your data. As you iterate, you will increase the dataset size to improve the I-matrix.

  **Example command:**

  ```sh
  ❯ uv run src/imatrix_dataset.py \
      --datasource_plugin src/imatrix_dataset/hf_dataset.py \
      --plugin_class HFDatasetPlugin \
      --langs en \
      --num_samples 50000 \
      --output imatrix_en_50k.bin
  ```

  - Increase `--num_samples` in each iteration (e.g., 50,000, 100,000, 200,000).
  - The `--output` parameter specifies the name of the I-matrix file for each iteration.

  **Note:** Remember that `imatrix_dataset.py` uses a plugin system to handle various data sources, but the details are covered later in this guide.

- **Quantize the Model with the New I-Matrix using `quantize.py`**

  Use the updated I-matrix in the quantization process.

  **Example command:**

  ```sh
  ❯ uv run src/quantize.py quantize \
      --model-name my_model \
      --base-model /path/to/baseline_model.gguf \
      --quantizations q4_0 \
      --imatrix-path imatrix_en_50k.bin \
      --output-dir /path/to/output_dir
  ```

  - The `--imatrix-path` parameter specifies the path to the I-matrix file generated in the current iteration.

- **Use `kl_d_bench.py` for Comparison**

  For each quantized model with an updated I-matrix, use `kl_d_bench.py` to compare against the baseline logits. Utilize the `--to-chunk` flag to potentially reduce computation time by halting the comparison at the point previously found with `--early-stopping`.

  - **Run the Comparison Script**

    ```sh
    ❯ uv run src/kl_d_bench.py \
        --baseline-logits baseline_logits.hdf5 \
        --target-model /path/to/output_dir/my_model-q4_0.gguf \
        --dataset /path/to/dataset.txt \
        --output-file comparison_iteration_1.hdf5 \
        --to-chunk 42
    ```

    - Update `comparison_iteration_1.hdf5` to reflect the iteration number (e.g., `comparison_iteration_2.hdf5`).

  - **Analyze KL-Divergence Metrics**

    Focus on key metrics, especially the high-percentile KL-divergence values, to assess the effectiveness of the quantization with each updated I-matrix.

- **Evaluate Metrics to Determine When to Stop Adding Data**

  Monitor the KL-divergence metrics across iterations. Pay special attention to the 90th, 95th, and 99th percentiles. When successive iterations show marginal improvements (diminishing returns), you can consider the I-matrix sufficiently refined for your application.

---

### **Metric for Evaluation**

To balance overall performance with outlier minimization, we suggest using a composite metric that combines the median KL-divergence and higher percentile values.

**Suggested Composite Metric:**

```math
\text{Score} = \left( \text{Median}^{1/3} \times \left( \text{KLD}_{99} \times 1 + \text{KLD}_{95} \times 4 + \text{KLD}_{90} \times 5 \right)^{2/3} \right)
```

- **Explanation:**

  - **Median KL-Divergence** (`Median`): Represents typical performance across the dataset.
  - **High Percentile KL-Divergence** (`KLD_{90}`, `KLD_{95}`, `KLD_{99}`): Capture the worst-case divergences, indicating how well the model handles outlier cases.
  - **Weighting Factors**: The weights (1 for `KLD_{99}`, 4 for `KLD_{95}`, 5 for `KLD_{90}`) emphasize reducing higher divergences, with greater weight on percentiles covering more data.
  - **Exponents**: The exponents (1/3 for the median, 2/3 for the weighted sum) balance the influence of average performance and outlier cases in the overall score.

By minimizing this composite score, you ensure that the quantized model maintains strong overall performance while mitigating significant divergences in less common scenarios.

---

### **Additional Guidance on Dataset Size**

Selecting an appropriate dataset size and coverage is crucial for effective I-matrix calibration. We recommend:

- **Starting Small**: Use a representative subset that includes key languages or domains relevant to your application.
- **Gradual Expansion**: Increase the dataset size in each iteration to include more diversity and complexity.
- **Balancing Diversity and Size**: Ensure that the dataset remains manageable while covering the necessary range of tokens and contexts.

For detailed insights into dataset selection and initial sizing, refer to [on_quantization.md](on_quantization.md). This document provides guidance on balancing dataset diversity and size to optimize the I-matrix calibration process.

---

### **Additional Tools and Options**

- **Measuring Perplexity with `quantize.py`**

  After quantization, you can measure the perplexity of your quantized model to assess its performance.

  **Example command:**

  ```sh
  ❯ uv run src/quantize.py perplexity \
      --model-name my_model \
      --base-model /path/to/output_dir/my_model-q4_0.gguf \
      --dataset /path/to/perplexity_dataset.txt
  ```

  - Replace `/path/to/perplexity_dataset.txt` with a dataset suitable for perplexity measurement.

In addition to the core scripts, the `src/extras/` folder contains supplementary tools that enhance your workflows. These scripts provide functionalities for optimization, visualization, data extraction, and file maintenance.

- **Optimizing Batch Sizes with `best_bub.py`**

  Before generating logits or quantizing large models, you may want to optimize batch (`--batch`) and micro-batch (`--ubatch`) sizes to maximize performance given your hardware constraints.

  **Example command:**

  ```sh
  ❯ uv run src/extras/best_bub.py --model /path/to/baseline_model.gguf --context-size 2048
  ```

  - Adjust the `--context-size` to match your model's maximum context size.
  - This script will suggest optimal `--batch-size` and `--ubatch-size` settings.

#### **1. Visualization and Data Extraction**

These scripts help analyze and visualize data generated during quantization, logit generation, and KL-divergence comparisons. They are useful for evaluating model performance, exploring patterns, and generating insights.

- **`analyze_comparison_progress_from_logs.py`:**  
   Visualizes early stopping factors and tracks progress during `compare_logits.py` runs. Projects remaining runtime and statistical trends. Outputs live updates or exports raw data for later analysis.

   **Example Command:**
   ```sh
   ❯ uv run src/extras/analyze_comparison_progress_from_logs.py --logfile <log_file_path>
   ```

- **`composite_comparison.py`:**  
  |  |  |
  |--|--|
  | ![3d KL-divergence manifold across files](assets/3D%20KL-Divergence%20manifold%20across%20files.png) | ![composite metrics by chunk](assets/Composite%20Metrics%20by%20Chunk.png) |
  
   Evaluates multiple comparisons using a composite metric. Provides overall KL-divergence curves, chunk-by-chunk scores, and 3D performance manifolds to identify quantization trends.

   **Example Command:**
   ```sh
   ❯ uv run src/extras/composite_comparison.py --input-files <file1> <file2> --output <summary_file>
   ```

- **`visualize_results.py`:**  
   Creates detailed visualizations of KL-divergence outputs, including chunk-by-chunk graphs and 3D manifolds. Helps identify patterns or anomalies in quantization results.

   **Example Command:**
   ```sh
   ❯ uv run src/extras/visualize_results.py --input <comparison_file> --output <graph_file>
   ```

- **`read_kl_d_benchmarks.py`:**  
   Extracts and displays KL-divergence statistics from comparison HDF5 files. Filters metrics by chunk range or includes overall statistics for a concise summary.

   **Example Command:**
   ```sh
   ❯ uv run src/extras/read_kl_d_benchmarks.py --input <comparison_file> --start-chunk 0 --end-chunk 10
   ```

#### **2. HDF5 Repair/Repurposing**

These scripts are designed to repair, repurpose, or modify HDF5 files used during logit generation or KL-divergence comparisons. They ensure data integrity and enable reuse in interrupted or experimental workflows.

- **`append_overall.py`:**  
   Computes and appends "overall" KL-divergence metrics to a comparison file, particularly for interrupted runs. Ensures a complete summary of results.

   **Example Command:**
   ```sh
   ❯ uv run src/extras/append_overall.py --input <comparison_file> --output <updated_file>
   ```

- **`reshape_logits.py`:**  
   Reshapes large logit files into smaller, evenly divided chunks, useful for exploring settings across multiple smaller chunks.

   **Example Command:**
   ```sh
   ❯ uv run src/extras/reshape_logits.py --input <logits_file> --output <reshaped_file> --chunk-size <size>
   ```

- **`unfree.py`:**  
   Resets the `freed_chunks` dataset in HDF5 logit files. This is helpful for resuming interrupted logit generation processes without losing progress.

   **Example Command:**
   ```sh
   ❯ uv run src/extras/unfree.py --input <hdf5_file>
   ```

---

### **Using the Plugin System in `imatrix_dataset.py`**

As mentioned earlier, the `imatrix_dataset.py` script uses a plugin system to support various data sources flexibly. Here's how you can utilize this system:

- **Selecting a Data Source Plugin**

  Choose an existing plugin or create a new one depending on your data source.

  - **Existing Plugins**:
    - `oscar_plugin.py`: An example huggingface dataset, for the OSCAR corpus.
  - **Custom Plugins**: Create a plugin tailored to your data source.

- **Specifying the Plugin in the Command**

  When running `imatrix_dataset.py`, use the following arguments to specify the plugin:

  - `--datasource_plugin`: Path to the plugin script.
  - `--plugin_class`: The class name of the plugin within the script.

  **Example command with an existing plugin:**

  ```sh
  ❯ uv run src/imatrix_dataset.py \
      --datasource_plugin src/imatrix_dataset/oscar_plugin.py \
      --plugin_class OscarDataSource \
      --langs en es de \
      --num_samples 100000 \
      --output /path/to/dataset.txt
  ```

- **Creating a Custom Data Source Plugin**

  If your data source isn't covered by existing plugins, you can create your own.

  **Steps to Create a Custom Plugin:**

  1. **Create a New Python File**: Save it with a descriptive name, e.g., `my_custom_plugin.py`.

  2. **Import the Base Class**:

     ```python
     from plugin_base import DataSourcePluginBase
     ```

  3. **Define Your Plugin Class**: Inherit from `DataSourcePluginBase`.

     ```python
     class MyCustomPlugin(DataSourcePluginBase):
         def __init__(self, name="my_dataset", **kwargs):
             super().__init__(name, **kwargs)
             self.schema = {'content': 'path.to.text.field'}
     ```

  4. **Implement the `load_data` Method**:

     ```python
     def load_data(self, lang, num_samples=200, skip_samples=0):
         # Implement your data loading logic here
         data_samples = []
         # Fetch data samples based on lang, num_samples, skip_samples
         return data_samples
     ```

  5. **Optionally Override `get_content` Method**:

     If your data records have a different structure, override this method to extract the text content.

     ```python
     def get_content(self, record):
         # Extract and return the text content from the record
         return record['desired_text_field']
     ```

  **Using Your Custom Plugin:**

  ```sh
  ❯ uv run src/imatrix_dataset.py \
      --datasource_plugin /path/to/my_custom_plugin.py \
      --plugin_class MyCustomPlugin \
      --langs en \
      --num_samples 100000 \
      --output /path/to/dataset.txt
  ```

- **Understanding the Plugin Base Class**

  The `DataSourcePluginBase` class defines the interface that all plugins must implement. It requires:

  - **Initialization**: Set up any necessary configurations or parameters.
  - **`load_data` Method**: Must return a list of data records for the specified language and sample counts.
  - **`get_content` Method**: Extracts the textual content from a data record, used in building the combined dataset.

- **Example of Using the OSCAR Plugin**

  The OSCAR dataset is a large multilingual corpus. Here's how to use the `oscar_plugin.py`:

  ```sh
  ❯ uv run src/imatrix_dataset.py \
      --datasource_plugin src/imatrix_dataset/oscar_plugin.py \
      --plugin_class OscarDataSource \
      --langs en fr es \
      --num_samples 50000 \
      --output /path/to/dataset.txt
  ```

  - This command generates a dataset using 50,000 samples from each of the specified languages.

---

### **Understanding the `--early-stopping` Argument**

With the introduction of the `--early-stopping` flag in both `compare_logits.py` and `kl_d_bench.py`, you can optimize the comparison process by allowing the script to terminate early once sufficient statistical evidence is gathered. This feature leverages statistical tests to determine when additional data processing is unlikely to yield significant new insights, thereby saving computational resources and time. See the [compare_logits specification](compare_logits_specification.md) file.

---

This set of tools is designed to simplify each stage of model quantization, from setting up datasets and generating I-matrices to quantizing models and evaluating performance. By following these steps, you can make targeted, data-driven adjustments at every stage, helping you achieve quantization results that preserve model quality while accommodating diverse data requirements.