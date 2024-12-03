# **`compare_logits.py`: Specification**

## **Introduction**

`compare_logits.py` is a Python script designed to compare the output logits of two machine learning models by computing the Kullback-Leibler (KL) divergence between them. This tool helps in understanding how similar or different the models are in their predictions, which is useful when evaluating changes such as adjustments in hyperparameters, architecture modifications, or updates in training data.

One of the key features of `compare_logits.py` is its ability to perform early stopping based on statistical significance, saving computational resources by determining when enough data has been analyzed to draw meaningful conclusions.

---

## **Use Cases**

- **Model Evaluation:** Compare a new model against a baseline to understand how changes impact predictions.
- **Hyperparameter Tuning:** Assess the effect of different hyperparameters on model outputs.
- **Assessing Model Changes:** Determine the impact of updates or modifications to a model relative to a specific dataset representing a domain of knowledge or capability (e.g., language understanding).

---

## **Features**

### **Basic Usage**

To use `compare_logits.py`, you need two HDF5 files containing the logits from the baseline model and the target model. The script compares these logits chunk by chunk and computes the KL divergence for each chunk.

**Sample Command:**

```bash
python compare_logits.py baseline_logits.hdf5 target_logits.hdf5
```

This command runs the script using the default settings. For more options and detailed usage instructions, you can run:

```bash
python compare_logits.py --help
```

### **Resumability**

`compare_logits.py` supports resumable processing. If the comparison process is interrupted or you want to continue processing additional chunks later, the script can resume from where it left off without reprocessing the completed chunks.

**Sample Command for Resumability:**

```bash
python compare_logits.py baseline_logits.hdf5 target_logits.hdf5 --from-chunk 10
```

This command resumes processing starting from chunk 10.

### **Early Stopping**

To optimize resource usage, the script can perform early stopping based on statistical tests. It analyzes the KL divergence values and stops processing new chunks when it determines that sufficient data has been collected to make a confident comparison.

**Sample Command with Early Stopping:**

```bash
python compare_logits.py baseline_logits.hdf5 target_logits.hdf5 --early-stopping
```

By enabling early stopping, the script automatically determines when to stop processing further chunks based on the configured statistical thresholds.

---

## **Foundations**

In developing `compare_logits.py`, several statistical methods were implemented to ensure accurate and meaningful comparisons between models. The script uses the Kullback-Leibler divergence as the primary metric for comparing logits and employs the Kuiper test for statistical analysis in early stopping. While we're exploring this space and recognize that our implementation may have limitations, we hope these methods help us draw reliable conclusions.

### **Kullback-Leibler Divergence**

**Definition:**

The Kullback-Leibler (KL) divergence measures how one probability distribution $P$ diverges from a second, reference probability distribution $Q$. It is defined as:

$$
D_{\text{KL}}(P \| Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)
$$

**Purpose in `compare_logits.py`:**

KL divergence is used to quantify the difference between the output distributions (logits) of the baseline and target models. By computing this metric for each row of each chunk of data, we can assess where and how the models differ in their predictions.

**Implementation Details:**

- **Numerical Stability:** To avoid issues with zero probabilities and numerical underflow, we add a small constant (epsilon) to probabilities and subtract the maximum logit value before exponentiating.
- **Temperature Scaling:** Temperature scaling to logits is available to control the sharpness of the probability distribution, which can help in stabilizing the softmax computation.

**Considerations:**

KL divergence is asymmetric and can be sensitive to discrepancies in low-probability events.

### **Statistical Tests for Early Stopping**

To determine when sufficient data has been processed to make a confident decision, we implement statistical tests as part of our early stopping mechanism.

#### **Kuiper Test**

**Definition:**


The Kuiper test is a non-parametric statistical test used to compare a sample with a reference probability distribution. It is an adaptation of the Kolmogorov-Smirnov test and is sensitive to differences in both the center and the tails of the distribution.

The Kuiper statistic $V$ is calculated as:

$$
V = D^+ + D^-
$$

where:

- $D^+ = \max (F_{\text{empirical}}(x) - F_{\text{theoretical}}(x))$
- $D^- = \max (F_{\text{theoretical}}(x) - F_{\text{empirical}}(x))$

**Purpose in `compare_logits.py`:**

The Kuiper test is used to compare the distribution of KL divergence values from the current data chunk against an exponential theoretical distribution fitted to the prior distribution. This helps assess whether the observed data significantly deviate from the expected distribution.

We chose the Kuiper test because it is equally sensitive across the entire range of the distribution and more sensitive to differences in the tails, which are significant when analyzing KL divergence values of models.

#### **Alternate Tests Considered:**

- **Anderson-Darling Test:** We considered using the Anderson-Darling test with an exponential distribution function. However, it required custom modifications for our specific distribution of KL divergence values, and the test statistics were not as discriminative for our purposes.
- **Kolmogorov-Smirnov (KS) Test:** We also looked into the KS test with an exponential distribution function. The KS test is known to be less sensitive to differences in the tails of distributions, which are important in our context.


**Implementation Details:**

- **Fitting the Exponential Distribution:** We fit an exponential distribution to the prior distribution of KL divergence values, fixing the location parameter at zero for stability.
- **Calculating the Kuiper Statistic:** The Kuiper statistic and corresponding p-value are computed using the `astropy.stats.kuiper` function.
- **Interpreting the Results:** The computed p-value indicates the probability of observing the data if it were drawn from the theoretical distribution. A low p-value suggests significant deviation, prompting the algorithm to continue processing more data.

**Considerations:**

The accuracy of the Kuiper test may be affected by small sample sizes especially where the tokenizer's number of vocabulary is small.

While the Kuiper test is well-suited for our needs, it relies on the assumption that the data follows the theoretical distribution (in our case, exponential).

### **Thresholds for Stability (`theta_E` and `theta_P`)**

To determine whether the effect sizes (Kuiper statistics) and their p-values have stabilized over recent data chunks, we use thresholds $\theta_E$ and $\theta_P$.

**Purpose:**

These thresholds help assess the stability of the statistical metrics over a sliding window of recent chunks. If both the effect sizes and p-values are stable, it suggests that further processing may not yield significantly different results, and the algorithm can consider stopping early.

**Implementation Details:**

- **Sliding Window Mechanism**: A sliding window of the most recent $N$ chunks is maintained to evaluate the stability of effect sizes and p-values. The window size determines how many recent chunks are considered in the stability assessment.
  
  A larger window provides a more robust estimate of stability by considering more data points, reducing the influence of short-term fluctuations. However, it may be less responsive to recent changes. A smaller window increases responsiveness to recent changes but may be more susceptible to noise and transient variations.

- **Fixed or Dynamic Thresholds:** Users can choose to either fixed or dynamic thresholds.

  - **Fixed Thresholds:** The algorithm can use user-specified fixed values for $\theta_E$ and $\theta_P$.
  - **Dynamic Thresholds:** The thresholds can be adjusted based on the Exponential Moving Averages (EMA) of the relative changes in effect sizes and the standard deviation of p-values. To ensure stability, the thresholds ($\theta_E$ and $\theta_P$) are clamped to minimum values. These are dynamically adjusted based on historical trends and confidence levels but are floored to avoid excessively low values:
	- For $\theta_E$, the minimum is set as the greater of its dynamic adjustment or $0.2$, reflecting a small effect size as defined by Cohen's guidelines.
	- For $\theta_P$, the floor is dynamically linked to $1 - \text{confidence}$.

    These clamping mechanisms prevent instability and ensure thresholds retain practical relevance. The use of $0.2$ as a floor for $\theta_E$ reflects its interpretation as a minimal effect size in statistical contexts.

- **Calculating Relative Changes and Standard Deviation:** Within each sliding window, we compute two key metrics to evaluate stability
  - **Effect Sizes:**
    $$
    \Delta E_t = \left| \frac{E_t - E_{t-1}}{E_{t-1}} \right|
    $$
    The relative change $\Delta E_t$ measures the proportional difference between consecutive effect sizes.

  - **P-Values:**
    $$
    \sigma_{P,t} = \sqrt{ \frac{1}{N - 1} \sum_{i=1}^N (P_i - \bar{P})^2 }
    $$
    The standard deviation $\sigma_{P,t}$ of p-values is calculated over a sliding window of size $N$.

- **Exponential Moving Averages (EMA):** To smooth the calculated metrics and reduce the impact of short-term variability, we apply Exponential Moving Averages (EMA) to both the relative changes in effect sizes and the standard deviation of p-values.

  - The EMA provides a smoothed estimate of the metrics to reduce the impact of short-term fluctuations.
  - The EMA is updated with each new chunk using a smoothing factor ($\text{ema\_decay}$):
    $$
    \text{EMA}_t = \text{ema\_decay} \times \text{CurrentValue}_t + (1 - \text{ema\_decay}) \times \text{EMA}_{t-1}
    $$

- **Stability Criteria:**

  - **Effect Size Stability:** Met when the EMA of relative changes is less than $\theta_E$:
    $$
    \text{EMA}_{\Delta E_t} < \theta_E
    $$
  - **P-Value Stability:** Met when the EMA of p-value standard deviation is less than $\theta_P$:
    $$
    \text{EMA}_{\sigma_{P,t}} < \theta_P
    $$

- **Updating Beta Parameters:** (see the section on the Beta distribution model)

  - **Increment $\alpha$ (Success):** If both stability conditions are met.
  - **Increment $\beta$ (Failure):** If either condition is not met.

**Considerations:**

- **Parameter Selection:** The choice of $\theta_E$ and $\theta_P$ significantly impacts the sensitivity of the early stopping mechanism.
- **Window Size:** The size of the sliding window affects the stability assessment. A larger window provides a more robust estimate but may be less responsive to recent changes.

### **Bayesian Prior Update Mechanism**

**Purpose:**

To adaptively update our beliefs about the distribution of KL divergence values, we use a Bayesian prior update mechanism. This helps in making informed decisions about early stopping based on accumulating data.

**Implementation Details:**

- **Prior Distribution:** We start with an initial prior distribution of KL divergence values, denoted as $P_{\text{prior}}(x)$.

- **Bayesian Updating:** As new chunks are processed, we update the prior distribution by combining it with the distribution of the new data. The updated prior $P_{\text{updated}}(x)$ is updated using a weighted combination of the existing prior and new data:
    $$
    P_{\text{updated}}(x) = w \cdot P_{\text{prior}}(x) + (1 - w) \cdot P_{\text{data}}(x)
    $$
  - The weight $w$ is determined based on the learning rate and the size of the new data relative to the average data size.

- **Learning Rate Decay:** The learning rate $\alpha$ adaptively decays based on observed KL divergence values rather than using fixed exponential decay. This enables the prior update mechanism to respond dynamically to the stability of the data:

  $$
  \text{adaptive\_decay} = \text{decay\_factor} \times (1 + \text{KL divergence})
  $$
  $$
  \text{decay\_amount} = (\alpha - \text{target\_alpha}) \times \text{adaptive\_decay}
  $$
  $$
  \alpha_{\text{new}} = \max(\text{target\_alpha}, \alpha - \text{decay\_amount})
  $$

  To ensure smooth transitions, a momentum term is incorporated:
  $$
  \alpha = \text{momentum} \times \alpha + (1 - \text{momentum}) \times \alpha_{\text{new}}
  $$

  where:
  - $\text{decay\_factor}$ scales the influence of KL divergence on the decay rate
  - $\text{target\_alpha}$ sets the minimum bound for the learning rate
  - $\text{momentum}$ moderates the update to prevent abrupt changes

  This approach allows faster convergence when distributions are similar while maintaining higher learning rates when significant differences exist.

- **Beta Distribution Model for Stopping Probability:** We model the probability of stopping using a Beta distribution with parameters $a$ and $b$, which are updated based on the stability of effect sizes and p-values.

  The Beta distribution is defined as:

  $$
  f(p; a, b) = \frac{\Gamma(a + b)}{\Gamma(a) \Gamma(b)} p^{a - 1} (1 - p)^{b - 1}
  $$

  where $\Gamma(\cdot)$ is the Gamma function.

- **Updating Beta Parameters:**  
  The parameters $a$ and $b$ are updated based on the stability criteria:
  - Increment $a$ (successes) if both stability conditions are satisfied.
  - Increment $b$ (failures) otherwise.

- **Stopping Probability Calculation**

  The stopping probability is calculated using the properties of the Beta distribution, which models the posterior distribution of the probability that the effect sizes and p-values are stable enough to consider stopping early.

  #### **Integral Form**

  The stopping probability $P_{\text{stop}}$ is defined as the integral of the Beta distribution's probability density function (PDF) from the desired confidence level to 1:

  $$
  P_{\text{stop}} = \int_{\text{confidence\_level}}^{1} f(p; \alpha, \beta) \, dp = I_{1 - \text{confidence\_level}}(\beta, \alpha)
  $$

  where:
  - $f(p; \alpha, \beta)$ is the probability density function (PDF) of the Beta distribution.
  - $I_x(a, b)$ is the regularized incomplete Beta function.
  - $\alpha$ represents the number of times the stability conditions have been met (successes).
  - $\beta$ represents the number of times the stability conditions have not been met (failures).
  - $\text{confidence\_level}$ is the desired confidence level (e.g., 0.95).

  #### **Implementation detail**

  The integral form of the stopping probability can be expressed using the cumulative distribution function (CDF), equivalently as the survival function (SF) of the Beta distribution:

  $$
  P_{\text{stop}} = 1 - \text{BetaCDF}(\text{confidence\_level}, \alpha, \beta) = \text{BetaSF}(\text{confidence\_level}, \alpha, \beta)
  $$

  **Decision to Stop:**

  If $P_{\text{stop}} \geq \text{confidence\_level}$, we consider stopping early.

**Considerations:**

- **Confidence Level:** The choice of the confidence level affects the sensitivity of the early stopping mechanism. A higher confidence level requires more consistent stability to stop early, reducing the likelihood of premature termination.
- **Parameter Sensitivity:** The values of $\alpha$ and $\beta$ influence the stopping probability. Users should ensure that the stability conditions are appropriately strict to avoid premature stopping. Specifically:
  - **High $\alpha$ and Low $\beta$:** Indicate strong evidence that stability conditions are consistently met.
  - **Low $\alpha$ and High $\beta$:** Suggest that stability conditions are frequently not met, indicating the need for continued processing.

---

### **Early Stopping: How They Work Together**

#### **Step-by-Step Interaction:**

1. **Kuiper Test on KL Divergence Values:**
   - The Kuiper Test is applied to the KL divergence values derived row-wise from the current chunk. 
   - It compares these values to the current Bayesian prior distribution, which models the expected distribution of KL values.
   - Outputs:
     - **Effect Size ($E_t$):** The Kuiper Test statistic measuring differences between the empirical distribution of KL values and the prior distribution.
     - **P-Value ($P_t$):** The statistical significance of these differences.

2. **Bayesian Prior Update:**
   - After processing the chunk, the Bayesian prior distribution is updated using the KL divergence values.
   - This ensures the reference distribution reflects the cumulative observations and adapts to trends in the data.

3. **Sliding Window Computes Metrics:**
   - For each new chunk, the sliding window is updated with the latest data.
   - Metrics ($\Delta E_t$ and $\sigma_{P,t}$) are recalculated for the current window and smoothed using EMA.

4. **Stability Conditions Checked:**
   - The smoothed metrics ($\text{EMA}_{\Delta E_t}$ and $\text{EMA}_{\sigma_{P,t}}$) are compared to the thresholds ($\theta_E$ and $\theta_P$).
   - If both metrics fall below their respective thresholds, stability conditions are considered met.

5. **Update $\alpha$ and $\beta$:**
   - If the stability conditions are met for the current window:
     - Increment $\alpha$ (successes).
   - Otherwise:
     - Increment $\beta$ (failures).

6. **Stopping Probability Calculation:**
   - Using the updated $\alpha$ and $\beta$, the stopping probability ($P_{\text{stop}}$) is calculated as:
     $$
     P_{\text{stop}} = \text{BetaSF}(\text{confidence\_level}, \alpha, \beta)
     $$
   - This represents the probability that the true success probability is greater than the specified confidence level.

7. **Decision to Stop:**
   - If $P_{\text{stop}} \geq \text{confidence\_level}$ and the minimum number of samples has been processed, early stopping is triggered.

---

## **Implementation Details**

- **Data Handling:** The script uses HDF5 files for efficient storage and access to large amounts of logits data.
- **Data Chunking:** To manage memory usage, data is processed in chunks. Users can specify the range of chunks to process.
- **Precision Control:** Users can choose the precision of the computations (e.g., 32-bit or 64-bit floating-point arithmetic) to balance performance and numerical accuracy.
- **Logging and Verbosity:** The script provides configurable logging levels to help users monitor the processing steps.

---

## **Example Output**

The output of `compare_logits.py` is an HDF5 file containing the computed KL divergence statistics for each processed chunk, as well as overall statistics and data used for early stopping. The structure of the HDF5 file is as follows:

- **Groups:**
  - Each chunk processed (e.g., `chunk_0`, `chunk_1`, ...) is represented as a group in the file.

- **Attributes within Each Chunk Group:**
  - `Average`: The mean KL divergence for the chunk.
  - `StdDev`: The standard deviation of the KL divergence values.
  - `Median`, `Minimum`, `Maximum`: Descriptive statistics for the chunk.
  - Percentiles such as `KLD_99`, `KLD_95`, `KLD_90`, etc., representing various quantiles of the KL divergence distribution.

- **Overall Statistics:**
  - An `overall` group or set of attributes that aggregates statistics across all processed chunks.

- **Early Stopping Data:**
  - **`prior_distribution`**: Stores the current prior distribution of KL divergence values.
  - **`bayesian_prior_state`**: Contains parameters like the current learning rate, decay rate, and iteration count.
  - **`early_stopping_stats`**: Includes the Beta distribution parameters $a$ and $b$, effect sizes, p-values, and other relevant statistics.

This structure allows users to programmatically access and analyze the results using HDF5-compatible tools and libraries, such as `h5py` in Python.

---

## **Conclusion**

`compare_logits.py` is a tool designed to assist in the comparison of machine learning models by analyzing their output logits. We recognize that this is an initial project, and while our implementation may have limitations, we hope that it provides a useful starting point for model evaluation tasks. By combining statistical methods like KL divergence, the Kuiper test, and Bayesian updating, the script aims to offer meaningful insights while optimizing computational resources through features like early stopping.
