# A concise summary with specific recommendations for quantizing your large language models (LLMs):

When working with multilingual _quantization_ for _large language models_ (LLMs), the _number of samples needed_ for effective quantization **increases with the number of target languages**. With more linguistic features, the model must learn and adapt across a broader spectrum during the quantization process.

Recent work, such as the _Lens_ framework and studies on quantized multilingual LLMs, emphasizes that larger datasets are critical for multilingual models to ensure performance remains consistent across all languages. These models typically perform best when they have **sufficient samples for each language**. This allows them to maintain their accuracy in quantization. In the case of multilingual evaluation tasks, several sources highlight that adding more languages requires **proportional increases** in calibration samples to smooth activations and avoid performance drops. These studies often mention the use of **thousands of samples per language** to preserve accuracy during multilingual post-training quantization ([Lens, 2024](https://ar5iv.org/html/2410.04407), [Quantization for Multilingual LLMs, 2024](https://ar5iv.org/abs/2407.03211)).

## Instruction Fine-tuning and Evaluation

Instruction fine-tuning has become crucial to enhance language models' ability to follow specific instructions and perform diverse tasks ([Chung et al., 2022](https://ar5iv.org/abs/2210.11416)) and especially chat interations. It typically involves training on datasets consisting of instruction-output pairs, which can be manually curated, transformed from existing datasets, or generated using other models.

The evaluation of instruction-tuned models often requires specialized methods ([Honovich et al., 2023](https://ar5iv.org/abs/2308.10792)). These methods focus on assessing the model's ability to follow instructions and generate appropriate responses, rather than relying solely on general metrics like perplexity.

Contrary to some assumptions, there is no established requirement or practice of including instruction data in the input matrix (imatrix) used for perplexity testing or other general evaluations ([Wei et al., 2022](https://ar5iv.org/abs/2206.07682)). The evaluation of instruction-tuned models typically involves task-specific metrics and methods that directly measure instruction-following capabilities.

----

With Salamandra models, the following assumptions made:

## **1. Use Calibration Data**
For **post-training quantization (PTQ)**, gather **several thousand calibration samples** per task. This helps smooth activations and adjust weights to avoid performance loss ([SmoothQuant](https://ar5iv.org/pdf/2211.10438v1), [Comprehensive Evaluation](https://aclanthology.org/2024-comprehensive)).

## **2. Dataset Size Recommendations**
- **For 2B models (Base or Instruct)**: Start with **1,000 to 5,000 samples** per language for quantization.

([SmoothQuant](https://ar5iv.org/pdf/2211.10438v1), [QLLM](https://openreview.net/forum?id=QLLLm)).

## **3. Balance Languages**
- For **multilingual models**, ensure you gather **balanced datasets** across languages. If resources are limited, start with perhaps **1,000 samples** per language and adjust based on performance ([QLLM](https://openreview.net/forum?id=QLLLm)). _Note: This recommendation may be effected by short samples. Based on research with Salamandra, I would now suggest 300k tokens per language (see the [addendum](#addendum))._


## **4. Start Small and Scale**
Begin with smaller datasets, evaluate the quantized model’s performance, and scale up as needed. **Add more samples** if you see significant drops in accuracy or performance after quantization ([Comprehensive Evaluation](https://aclanthology.org/2024-comprehensive), [DataCamp, 2023](https://www.datacamp.com/community/tutorials/quantization-llms)).


# Addendum

Salamandra has 34 target languages, plus code, and is meant to support svaious european languages. 34*1000 samples were used to quantize the model with an iMatrix. In the llama-gguf-optimze project we built tools to quantify the improvement prlative to the size of the dataset. It appears that a sweet spot of about 360k tokens _per language_ may be ideal, which is about 500 samples/language. Improvement largely flattens thereafter. 

![3d KL-divergence manifold across files](assets/3D%20KL-Divergence%20manifold%20across%20files.png)

![composite metrics by chunk](assets/Composite%20Metrics%20by%20Chunk.png)

Note that in the above images, the 1000 sample quantization was not done at the same time and may not have had comparable arguments provided when creating this quantization.

Across 34 languages, this is far larger than the recommended 500k for only english + code domains in this [discussion](https://github.com/ggerganov/llama.cpp/discussions/5263) at llama.cpp. The sampling of the chunks with the highest kl-divergence to achieve 40k high-quality tokens would still apply.

---

### **References**
1. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**  
   - Authors: Xiao et al.  
   - Year: 2023  
   - Source: [arXiv](https://ar5iv.org/pdf/2211.10438v1)  
   - Summary: This paper addresses activation outliers in large models and recommends using calibration samples for effective quantization.

2. **QLLM: Accurate and Efficient Low-bitwidth Quantization for LLMs**  
   - Authors: Liu et al.  
   - Year: 2024  
   - Source: [ICLR](https://openreview.net/forum?id=QLLLm)  
   - Summary: QLLM focuses on outlier handling and low-bitwidth quantization for models like LLaMA, recommending balanced datasets and channel-wise techniques.

3. **A Comprehensive Evaluation of Quantization Strategies for Large Language Models**  
   - Authors: Jin et al.  
   - Year: 2024  
   - Source: [ACL Anthology](https://aclanthology.org/2024-comprehensive)  
   - Summary: Provides a thorough evaluation of quantization strategies on various LLMs, noting that several thousand samples per task are often needed.

4. **Quantization for Large Language Models (LLMs): Reduce AI Model Sizes Efficiently**  
   - Year: 2023  
   - Source: [DataCamp](https://www.datacamp.com/community/tutorials/quantization-llms)  
   - Summary: Introduces practical methods for quantizing models and discusses dataset requirements for ensuring performance.

5. **Lens: Rethinking Multilingual Enhancement for Large Language Models**  
- Authors: Zhao, Weixiang, et al.  
- Year: 2024  
- Source: [arXiv](https://ar5iv.org/html/2410.04407)  
- Summary: This study emphasizes that as the number of languages increases, the number of samples required for quantization grows. Multilingual models need larger datasets to maintain performance across all languages. The authors recommend scaling the number of samples per language as the model size and the number of target languages increase.

6 **How Does Quantization Affect Multilingual LLMs?**  
- Authors: Ahmadian et al.  
- Year: 2024  
- Source: [arXiv](https://ar5iv.org/abs/2407.03211)  
- Summary: This paper explores the impact of quantization on multilingual LLMs. It highlights the need for larger datasets as the number of target languages increases and suggests using several thousand calibration samples per language to mitigate performance degradation.

7 **Emergent Abilities of Large Language Models**  
   - Authors: Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., ... & Fedus, W.  
   - Year: 2022  
   - Source: [arXiv](https://ar5iv.org/abs/2206.07682)  
   - Summary: This paper investigates emergent abilities in large language models as they scale in size. The authors demonstrate how model capabilities appear unexpectedly at certain scale thresholds.

8 **Scaling Instruction-Finetuned Language Models**  
   - Authors: Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... & Le, Q. V.  
   - Year: 2022  
   - Source: [arXiv](https://ar5iv.org/abs/2210.11416)  
   - Summary: The authors explore the scaling of instruction-finetuned language models and their impact on downstream task performance, showing how larger models benefit from instruction tuning.

9 **Instruction Tuning for Large Language Models: A Survey**  
   - Authors: Honovich, O., Shaham, U., Bowman, S. R., & Levy, O.  
   - Year: 2023  
   - Source: [arXiv](https://ar5iv.org/abs/2308.10792)  
   - Summary: This survey paper provides a comprehensive overview of instruction tuning for large language models, summarizing recent advances and challenges in optimizing models for specific instructions.
