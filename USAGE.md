# Iterative Quantization and Comparison

This guide outlines a systematic approach for quantizing your models using iterative refinement of the importance matrix (I-matrix) and evaluating quantization quality through KL-divergence metrics. By following these steps, you can optimize your model quantization to achieve minimal performance loss across diverse data subsets.

#### Step 1: Generate and Store Baseline Logits (One-Time Setup)

- **Run `generate_logits.py` on the Baseline Model**

  Begin by generating logits from your unquantized (baseline) model over the target dataset. Use the `generate_logits.py` script to create a reusable HDF5 file containing these logits. Generating the baseline logits once saves time and storage in subsequent comparisons.

  **Example command:**

  ```sh
  ❯ uv run src/generate_logits.py --model /path/to/baseline_model --dataset /path/to/dataset --output baseline_logits.hdf5
  ```

  - Replace `/path/to/baseline_model` with the path to your unquantized model.
  - Replace `/path/to/dataset` with the path to your dataset file.
  - The `--output` parameter specifies the name of the output HDF5 file.

- **Save Baseline Logits for Reference**

  Store the generated `baseline_logits.hdf5` file in a secure and accessible location. This file will serve as the reference point for comparing outputs from quantized models in later steps.

#### Step 2: Initial Quantization (Without I-Matrix) and Baseline Comparison

- **Quantize the Model without an I-Matrix**

  Create an initial quantized version of your model without using an I-matrix. This serves as a baseline to assess the impact of I-matrix calibration on quantization quality in subsequent iterations.

  **Example command (using `llama.cpp` for quantization):**

  ```sh
  ❯ ./quantize /path/to/baseline_model.bin /path/to/quantized_model.bin q4_0
  ```

  - Adjust the command according to the quantization tool you're using.
  - Ensure you document the quantization parameters for reproducibility.

- **Run `kl_d_bench.py` for Initial Comparison**

  Use the `kl_d_bench.py` script to compare the logits of the quantized model against the baseline logits. This script processes the stored baseline logits and computes KL-divergence metrics efficiently.

  **Example command:**

  ```sh
  ❯ uv run src/kl_d_bench.py --baseline-logits baseline_logits.hdf5 --target-model /path/to/quantized_model.bin --dataset /path/to/dataset --output-file initial_comparison.hdf5
  ```

  - Replace `/path/to/quantized_model.bin` with the path to your quantized model.
  - The `--output-file` parameter specifies where to save the comparison results.

- **Collect and Evaluate Metrics**

  After running `kl_d_bench.py`, review the KL-divergence metrics, including the median and the 90th, 95th, and 99th percentiles for each data chunk. This initial assessment serves as a reference point for evaluating improvements from I-matrix calibration in later iterations.

#### Step 3: Iterative I-Matrix Calibration and Quantization

- **Generate I-Matrices with Incrementally Larger Dataset Subsets**

  Begin refining the I-matrix by generating it using a small subset of your dataset. Use the `imatrix_dataset.ipynb` notebook or convert it into a script to create the I-matrix tailored to your data.

  For each iteration:

  - **Increase the Dataset Subset Size**: Gradually include more data to cover a broader range of tokens and contexts.
  - **Generate a New I-Matrix**: Update the I-matrix using the expanded dataset subset.
  - **Quantize the Model with the New I-Matrix**: Use the updated I-matrix in the quantization process.

  **Note:** Ensure that each generated I-matrix is saved with a distinct name to avoid confusion.

- **Use `kl_d_bench.py` for Comparison**

  For each quantized model with an updated I-matrix:

  - **Run the Comparison Script**

    ```sh
    ❯ uv run src/kl_d_bench.py --baseline-logits baseline_logits.hdf5 --target-model /path/to/quantized_model_with_imatrix.bin --dataset /path/to/dataset --output-file comparison_iteration_n.hdf5
    ```

    - Replace `/path/to/quantized_model_with_imatrix.bin` with the path to the quantized model using the current I-matrix.
    - Update `comparison_iteration_n.hdf5` to reflect the iteration number.

  - **Analyze KL-Divergence Metrics**

    Focus on key metrics, especially the high-percentile KL-divergence values, to assess the effectiveness of the quantization with each updated I-matrix.

- **Evaluate Metrics to Determine When to Stop**

  Monitor the KL-divergence metrics across iterations. Pay special attention to the 90th, 95th, and 99th percentiles. When successive iterations show marginal improvements (diminishing returns), you can consider the I-matrix sufficiently refined for your application.

#### Metric for Evaluation

To balance overall performance with outlier minimization, we suggest using a composite metric that combines the median KL-divergence and higher percentile values.

**Suggested Composite Metric:**

\[
\text{Score} = \left( \text{Median}^{1/3} \times \left( \text{KLD}_{99} \times 1 + \text{KLD}_{95} \times 4 + \text{KLD}_{90} \times 5 \right)^{2/3} \right)
\]

- **Explanation:**

  - **Median KL-Divergence** (`Median`): Represents typical performance across the dataset.
  - **High Percentile KL-Divergence** (`KLD_{90}`, `KLD_{95}`, `KLD_{99}`): Capture the worst-case divergences, indicating how well the model handles outlier cases.
  - **Weighting Factors**: The weights (1 for `KLD_{99}`, 4 for `KLD_{95}`, 5 for `KLD_{90}`) emphasize reducing higher divergences, with greater weight on percentiles covering more data.
  - **Exponents**: The exponents (1/3 for the median, 2/3 for the weighted sum) balance the influence of average performance and outlier cases in the overall score.

By minimizing this composite score, you ensure that the quantized model maintains strong overall performance while mitigating significant divergences in less common scenarios.

#### Additional Guidance on Dataset Size

Selecting an appropriate dataset size and coverage is crucial for effective I-matrix calibration. We recommend:

- **Starting Small**: Use a representative subset that includes key languages or domains relevant to your application.
- **Gradual Expansion**: Increase the dataset size in each iteration to include more diversity and complexity.
- **Balancing Diversity and Size**: Ensure that the dataset remains manageable while covering the necessary range of tokens and contexts.

For detailed insights into dataset selection and initial sizing, refer to [on_quantization.md](on_quantization.md). This document provides guidance on balancing dataset diversity and size to optimize the I-matrix calibration process.
