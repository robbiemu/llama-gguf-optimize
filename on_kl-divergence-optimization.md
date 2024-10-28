# On KL-Divergence and Context Size Optimization in GGUF Quantization

<small>This paper was written with AI, and the exploration it describes was done collaboratively with AI. The "we" described here is us; me and a few models.</small>

With llama.cpp model quantization, effectively calibrating models to maintain their performance after reducing precision is a complex but worthwhile process. The GGUF quantization method offers efficient model compression by lowering the bit-width of parameters, as many other approaches, but it provides a way to avoid the problems inheritent to purely automatic processes. Quantization introduces error due to the reduced information capacity of lower-bit representations, and thus, effective calibration (or, for GGUF, its equivalent) is necessary to ensure that the quantized model performs closely to the original.

The GGUF format's tool for enhancing quantization, akin to calibration, is using an **importance matrix** (or "iMatrix"), which is a dataset or structure that informs the quantization process about critical regions within the model's parameters. By capturing areas where quantization tends to introduce significant divergence from the baseline model, the iMatrix allows for targeted calibration, mitigating errors in these high-impact regions. In essence, it provides the model with "importance-weighted" data, helping to maintain alignment with the original model by prioritizing quantization accuracy in sections where it matters most. 

The **relationship between calibration and GGUF’s importance matrix (iMatrix)** lies in their shared goal of reducing quantization error, but they differ in implementation. **Calibration** typically adjusts the model’s parameters to mitigate errors introduced by lower-bit quantization, often tuning the model’s response post-quantization to bring it closer to the baseline model. In contrast, GGUF’s **iMatrix approach** pre-selects and emphasizes high-importance regions during quantization, guiding the process by weighting specific data as "important" based on where errors are likely to have a significant impact. Rather than adjusting the model after quantization, the iMatrix directly informs the quantization process, helping preserve accuracy in critical areas from the outset. 

The llama.cpp discussions we analyzed focused on strategies for **selecting the most effective data** to create an iMatrix that optimally supports quantization. Rather than detailing the application of an iMatrix, the discussions centered on identifying data segments where quantization introduces the most significant errors, particularly through analyzing KL-divergence. By selecting high-divergence chunks or outlier sections for the iMatrix, the goal was to prioritize these “high-impact” areas, ensuring that the quantization process preserves accuracy in the model’s most sensitive regions. This approach has the goal of guiding the  choice of data to enhance the iMatrix’s impact on maintaining model fidelity post-quantization. This paper serves as a post-mortem analysis of those discussions. 

In our analysis (as the initial discussions), we use **KL-divergence (Kullback–Leibler divergence)** to assess how quantization affects model accuracy. Specifically, KL-divergence measures the difference between the output probabilities of the **quantized model** and the **original (baseline) model**. In this context, it helps us evaluate how closely the quantized model replicates the probabilistic predictions of the unquantized version. Lower KL-divergence values indicate that the quantized model’s outputs remain aligned with the baseline, signifying minimal distortion introduced by quantization.

KL-divergence is especially effective for quantization testing. It allows us to see the **average difference** across outputs. It also illustrates the **distribution of those differences**. By examining specific percentiles of KL-divergence values (such as the 90th and 95th percentiles), we can focus on the most substantial deviations or “outlier errors.” These high-percentile values reveal how the quantized model performs in cases where errors are likely to have the largest impact, providing insights into the model’s stability in handling complex or challenging inputs. In this way, KL-divergence guides optimal quantization by pin-pointing where the most significant errors occur. This can be used to ensure calibration efforts reduce these critical outlier deviations.

We review largely two llama.cpp conversations, examine their data and discussion notes about context size in GGUF quantization, to iteratively refine our understanding of what factors and methods lead to improved quantization accuracy. Several key questions guided our exploration:

### Key Questions in Understanding iMatrix and Quantization Effectiveness

1. **How does the native context size of a model affect KL-divergence in quantization?**
2. **What role does dataset selection play in calibrating iMatrix for minimizing divergence?**
3. **Is a random sampling approach valid for iMatrix calibration, or does structured, non-random data yield better results?**
4. **Are there diminishing returns in subsampling strategies for iMatrix data?**

---

## 1. Native Context Size and Its Impact on KL-Divergence

We began with analyzing KL-divergence across various context sizes—512, 2048, 4096, and 8192 tokens—used in calibration. The objective was to determine if quantization error was minimized at a particular context size that aligned with the model’s operational characteristics, referred to here as the **native context size**. The native context size, we hypothesized, would align with the model’s training configuration, allowing it to perform optimally by minimizing KL-divergence post-quantization. For this particular model, we tested the effects across these context sizes to evaluate which best reduced error.

### Identifying Key KL-Divergence Metrics and Interpreting Lower Scores as Better

The KL-divergence metrics of quantization calibration provide insight into how closely a quantized model replicates the unquantized baseline. Lower KL-divergence values are desirable, as they indicate reduced divergence between the two models, meaning the quantized model’s outputs are closer to those of the original, higher-fidelity model.

Among the available KL-divergence values, **median, KLD_95, and KLD_90** emerged as critical metrics:

- **Median KL-Divergence**: The median value serves as an indicator of central tendency, revealing the typical level of divergence across chunks. A lower median value suggests overall stability in the quantized model's predictions, aligning closely with the baseline.
- **KLD_95 and KLD_90 (95th and 90th Percentiles)**: High-percentile values such as KLD_95 and KLD_90 capture the behavior of the model at outlier levels of divergence. These values are especially relevant, as they highlight areas where quantization errors have the most severe impact. Lower high-percentile values, therefore, imply that the model avoids significant divergence in these critical sections, helping to ensure reliable performance in high-impact scenarios.

Through our analysis, we focused on these three metrics, as lower values across them indicate a quantized model that more accurately reflects the original, minimizing significant divergence and stabilizing general output quality.

### Data Analysis: Comparing Context Sizes to Determine the Native Context

With these metrics in mind, we assessed KL-divergence results across the four context sizes to pinpoint the context with the lowest scores, thus identifying the native context. Below are the key values for each context size, with the **4096 context size used as the basis for comparison** since it demonstrated the lowest KL-divergence across all metrics:

| Context Size | Median     | % Difference from 4096 | KLD_95      | % Difference from 4096 | KLD_90      | % Difference from 4096 |
|--------------|------------|------------------------|-------------|------------------------|-------------|------------------------|
| **512**      | 0.003271   | +5.28%                | 0.135895    | +5.55%                 | 0.077871    | +4.76%                 |
| **2048**     | 0.003527   | +13.52%               | 0.139948    | +8.69%                 | 0.078575    | +5.70%                 |
| **4096**     | 0.003107   | (Baseline)            | 0.128744    | (Baseline)             | 0.074335    | (Baseline)             |
| **8192**     | 0.003311   | +6.57%                | 0.144028    | +11.89%                | 0.079583    | +7.06%                 |

The data reveals several significant findings:

- **Median KL-Divergence**: The native context size of 4096 tokens produces the lowest median KL-divergence at 0.003107, serving as the baseline for comparison. Shorter contexts like 512 and 2048 show higher median values by approximately 5.28% and 13.52%, respectively, while extending to 8192 tokens results in a 6.57% increase. This suggests that shorter or longer contexts introduce instability, whereas the native context size minimizes divergence at the median level.
  
- **KLD_95 (95th Percentile)**: At the high end, 4096 again performs best with a KLD_95 value of 0.128744. Shorter contexts such as 512 and 2048 have KLD_95 values 5.55% and 8.69% higher, respectively, while extending the context to 8192 results in an 11.89% increase. This pattern indicates that deviations from the native context size increase outlier divergence, underscoring 4096 as the most stable choice for controlling high-end errors.

- **KLD_90 (90th Percentile)**: Similarly, the 4096 context size yields the lowest KLD_90 value at 0.074335. At shorter contexts, KLD_90 values increase by 4.76% for 512 and 5.70% for 2048. The longest context, 8192, exceeds the context size of the model, so the test overshot (stressed attention in an opposite way from the smallest sizes). It shows a 7.06% increase over the baseline, further reinforcing that the native context size reduces divergence even in high-divergence percentiles.

There's a lot of noise here, but if we subtract the 4096 line and ignore the parts of the data that disagree with the overall trend, we see a gradient of values roughly 1-2% apart. The values are lowest for the test with 4096 tokens, so it best aligns with the model’s training configuration. Using the full context minimizes divergence. As context sizes deviate from 4096, error rates rise.

The **native context size of 4096** achieves the lowest KL-divergence values across median and high-percentile metrics, making it the optimal range for calibration in this model. This alignment allows quantization to perform with minimal divergence, ensuring that the quantized model closely approximates the original while avoiding the instability seen with non-native contexts.

## 2. Dataset Selection for iMatrix Calibration

NExt we looked at the model's performance with real-world application data, emphasizing sections where quantization is most likely to introduce error. Here, we discussed the initial choice of a general-purpose dataset of psuedo-random data ("groups_merged.txt") and the subsequent move to Wikitext for analyzing high-divergence sections.

In comparing diverse datasets, one member of the conversation noted that Wikitext data had higher KL-divergence scores on average than groups_merged.txt, suggesting that Wikitext’s lower entropy may introduce more challenging sections for the model to predict accurately after quantization. This revelation led to targeted chunk selection in the dataset, aiming to include high-divergence segments in the iMatrix to strengthen the model's quantization resilience specifically in these areas.

This selective approach—identifying chunks with high divergence and excluding others from the iMatrix—helped the model perform more consistently across outliers, a valuable insight for future calibration practices. However, we realized that while this approach yielded improvements, it returns were modest, as explained below.

## 3. Random vs. Non-Random Sampling in iMatrix Construction

Early in the discussions, the **Wikitext KL-divergence test** was used as a baseline to measure the divergence between the quantized model and the original model using a structured, continuous text source. Wikitext is a lower-entropy dataset that includes natural language structure, making it a useful benchmark to observe quantization effects across a range of predictable, real-world data. By calculating KL-divergence on Wikitext, the team could identify areas where quantization introduced higher divergence, especially in structured, real-data sequences. This allowed for a more targeted approach in iMatrix construction, focusing on high-divergence sections and high-impact regions for calibration.

However in an earlier discussion at llama.cpp it was decided that random data was better; although this was ultimately successfully challenged. It was not immediately clear in this later discussion if the wikitext data was randomized or not. (This is all elaborated so intricately because of section 4.)

We considered whether the Wikitext KL-divergence test might have used random sampling, as **random sampling is often used as a broad approach to gauge average model performance across varied data**. Random sampling provides a baseline view of quantization effects by delivering a representative mix without focusing on any specific text structure or sequence. If Wikitext had used random sampling, it would have indicated an aim to generalize the model’s quantization stability across diverse data types, rather than targeting specific patterns or high-impact areas.

Any measurable difference in the choice between random and non-random sampling for iMatrix calibration highlights the importance of structured data in addressing the real challenges of quantization. Additional context revealed that the Wikitext test was structured and followed the natural text order, allowing the model to engage with inherent data patterns more realistically. This structured, non-random approach provided a clearer view of where quantization introduces errors in actual data sequences, unlike randomized sampling, which may overlook systematic weaknesses. It appears that lower-entropy, natural language structure can provide insights into how quantization affects the model on realistic text data, allowing the team to identify high-divergence areas that could be prioritized in the iMatrix.

## 4. Diminishing Returns in Subsampling for iMatrix Calibration

To understand the impact of selective subsampling for iMatrix calibration, we examined the relative improvements across different iMatrix calibration strategies, each aiming to reduce KL-divergence. Specifically, we compared three setups: quantization with **no iMatrix** (q4_0 without calibration), quantization using **top ~40k tokens with high KL-divergence**, and **full ~500k tokens** from Wikitext with iMatrix. By comparing critical metrics—median, KLD_95, and KLD_90—across these setups, we sought to determine if targeted selection yielded meaningful improvements beyond a general-purpose dataset.

### Data Comparison and Initial Findings

Each calibration setup had the following KL-divergence statistics for the key metrics (median, KLD_95, and KLD_90):

| Metric            | No iMatrix (q4_0) | Top ~40k Tokens, iMatrix | Full ~500k Tokens, iMatrix | Relative Improvement (Top ~40k vs. Full ~500k) |
|-------------------|-------------------|---------------------------|----------------------------|-----------------------------------------------|
| **Median**        | 0.016759          | 0.013641                  | 0.013710                   | **0.51%** better with Top ~40k                |
| **KLD_95**        | 0.169855          | 0.142108                  | 0.142599                   | **0.34%** better with Top ~40k                |
| **KLD_90**        | 0.104060          | 0.086210                  | 0.086496                   | **0.27%** better with Top ~40k                |

These findings confirm that the presence of an iMatrix—whether targeted or generalized—yields notable improvements over the no-iMatrix baseline. However, the incremental gains when using the top ~40k high KL-divergence tokens over the full ~500k tokens were small.

### Analyzing Relative Improvements to Assess Diminishing Returns

To assess whether these improvements were meaningful or indicative of diminishing returns, we examined the **relative differences** between the targeted top ~40k token iMatrix and the general-purpose full ~500k token iMatrix:

1. **Median KL-Divergence**:
   - The top ~40k tokens showed a median score of 0.013641 compared to 0.013710 for the full ~500k tokens.
   - This yields a difference of **0.51%** in favor of the top ~40k tokens,

 a marginal improvement that might fall within the margin of error.

2. **KLD_95**:
   - The top ~40k tokens produced a KLD_95 score of 0.142108 versus 0.142599 for the full ~500k tokens.
   - This results in a **0.34% improvement** with the targeted selection, another small gain indicating that subsampling has minimal additional effect.

3. **KLD_90**:
   - The top ~40k tokens gave a KLD_90 score of 0.086210, compared to 0.086496 with the full ~500k tokens.
   - This yields a **0.27% improvement**, again a minimal difference.

These small percentages indicate that the additional improvement gained by focusing on high-divergence chunks is slight, especially when compared to the more substantial improvements achieved by introducing iMatrix calibration in general. The **0.3%–0.5% incremental gain** achieved through subsampling is not automatically dismissable, but it is half and order of magnitude less than the differences we were seeing in different context sizes. These small gains suggest that further refining subsampling strategies would likely yield diminishing returns and that the model’s calibration could be better enhanced through a focus on broader, high-quality datasets.

---
## Toward Comprehensive Data

Based on this analysis, it’s clear that the iMatrix calibration contributes to an improvement, while the additional approach of selecting high KL-divergent chunks contributes only modestly. This **diminishing return** indicates that beyond a certain point, selective subsampling adds minimal value. Instead, increasing the quantity and quality of accurately curated, diverse data may offer more impactful gains by capturing a wider array of patterns and outliers within realistic data distributions. In summary, while targeted iMatrix selection based on high-divergence chunks offers slight gains, its impact is limited. Expanding to a diverse, representative dataset appears to be the more effective approach for maintaining alignment with the unquantized baseline, especially for models expected to handle varied real-world input.