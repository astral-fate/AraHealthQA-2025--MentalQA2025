# Sakinah-AI at MentalQA: A Comparative Study of Few-Shot, Optimized, and Ensemble Methods for Arabic Mental Health Question Classification

<p align="center">
<img src="https://placehold.co/800x200/dbeafe/3b82f6?text=Sakinah-AI+Project" alt="Sakinah-AI Project Banner">
</p>

#### [Fatimah Emad Elden](https://scholar.google.com/citations?user=CfX6eA8AAAAJ&hl=ar), [Mumina Abukar](https://www.linkedin.com/in/dr-mumina-alshaikh-md-mscph-501621216/)

#### **Cairo University & The University of South Wales**

[![Paper](https://img.shields.io/badge/arXiv-25XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/25XX.XXXXX)
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/astral-fate/MentalQA2025/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Page-F9D371)]([https://huggingface.co/BiMediX](https://huggingface.co/collections/FatimahEmadEldin/sakinah-ai-at-mentalqa-689b2d707791cea458e97aaf/))
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://github.com/astral-fate/mentalqa2025/blob/main/LICENSE)

---

## 🏆 Overview

This repository contains the code and resources for our submission to the **MentalQA 2025 Shared Task, Track 1**. Our work focuses on the multi-label classification of user questions in Arabic within the mental health domain. We conduct a comprehensive comparative analysis of few-shot learning with Large Language Models (LLMs) against several fine-tuning strategies using two base models: **CAMeL-BERT** and **AraBERT**. For each base model, we evaluate a single, hyperparameter-optimized model against a 5-fold cross-validation ensemble. Our findings reveal that few-shot learning with a large, specialized LLM and a single, optimized CAMeL-BERT model are the most effective strategies for this task.

---

## 🔑 Key Findings

* **Few-Shot Learning Excels:** Few-shot learning with **Writer/Palmyra-Med-70B-32K** achieved the highest weighted F1-score of **0.605**.
* **Optimized Fine-Tuning is Competitive:** A single, well-tuned **CAMeL-BERT** model was a very close second with a weighted F1-score of **0.597**, significantly outperforming its AraBERTv2 counterpart (0.543).
* **Model Choice is Crucial:** The superior performance of CAMeL-BERT over AraBERTv2 highlights the critical impact of the base model's pre-training on user-generated, dialectal Arabic content.
* **Ensembling Can Be Detrimental:** The k-fold ensemble method proved detrimental for both base models, suggesting that on small, specialized datasets, averaging predictions can degrade performance.

---

## ⚙️ Methodology & System Architecture

Our methodology is a three-pronged comparative analysis:

1.  **Optimized Single Model Fine-Tuning:** We used the **Optuna** framework for automated hyperparameter search to find the best configuration for a single model.

2.  **K-Fold Ensemble Fine-Tuning:** We trained an ensemble of five models using 5-fold cross-validation and averaged their predictions. The training and development sets were combined for this process.

3.  **In-Context Few-Shot Learning with LLMs:** For this strategy, we constructed a prompt that included a task description, category definitions, and 3-5 examples from the training set to guide the LLM in classifying the questions. We utilized several large language models available through the NVIDIA NIM inference microservices API.

<p align="center">
<img width="1183" height="976" alt="Screenshot 2025-08-12 133722" src="https://github.com/user-attachments/assets/78a007fd-c300-4cf1-a13e-3552beb56e03" />

 
  <em>A high-level overview of our data processing pipeline, from input data to the final parsed output.</em>
</p>

---

## 📊 Results

Our experiments revealed a clear performance hierarchy among the different approaches. The Palmyra-Med LLM and the single optimized CAMeL-BERT model are the top performers, significantly outclassing all other approaches.

### Blind Test Set Performance

| Paradigm       | Model Name               | Weighted F1-Score |
| :------------- | :----------------------- | :---------------- |
| **Few-Shot** | **Palmyra-Med-70B** | **0.605** |
| **Fine-Tuning**| **CAMEL-BERT (Optimized)** | 0.597             |
| Few-Shot       | Mixtral-8X22B            | 0.563             |
| Fine-Tuning    | AraBERTv2 (Optimized)    | 0.543             |
| Fine-Tuning    | CAMEL-BERT (K-Fold)      | 0.537             |
| Few-Shot       | Qwen3-235B               | 0.325             |
| Fine-Tuning    | AraBERTv2 (K-Fold)       | 0.328             |
| Few-Shot       | Gpt-Oss-20B              | 0.147             |
| Few-Shot       | Colosseum-355B           | 0.014             |

*Table derived from results presented in the paper.*

---

## 🤖 Pre-trained Models

The pre-trained model weights for our best-performing fine-tuned models can be found on the Hugging Face Hub:

| Model Name           | Link to Model                                  |
|----------------------|------------------------------------------------|
| Sakinah-AI-CAMEL-BERT-Optimized | [HuggingFace](https://huggingface.co/FatimahEmadEldin/Sakinah-AI-CAMEL-BERT-Optimized) |
| Sakinah-AI-AraBERT-Optimized  | [HuggingFace](https://huggingface.co/FatimahEmadEldin/Sakinah-AI-AraBERT-Optimized) |

---

---

## 📜 License & Citation

This project is released under the **MIT License**. For more details, please refer to the `LICENSE` file.

If you use our work in your research, please cite our paper:

```bibtex
@inproceedings{elden2025sakinahai,
      title={{Sakinah-AI at MentalQA: A Comparative Study of Few-Shot, Optimized, and Ensemble Methods for Arabic Mental Health Question Classification}},
      author={Elden, Fatimah Emad and Abukar, Mumina},
      year={2025},
      booktitle={Proceedings of the MentalQA 2025 Shared Task},
      eprint={25XX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
