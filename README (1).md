# Part 3: NLP and Sequence Modeling — Customer Support Sentiment Classification

## Overview

This project builds a complete NLP pipeline to classify customer support messages into three sentiment categories: **negative**, **neutral**, and **positive**.

**Dataset:** Customer Support Text Classification (1,500 records, 6 columns)  
**Source:** [Shared Google Drive Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)

---

## Repository Structure

```
part-3-nlp-sequence-modeling/
├── notebook.ipynb              ← Full Jupyter notebook (all 6 tasks)
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
└── results/
    ├── 01_class_distribution.png    ← Sentiment counts + word length histogram
    ├── 02_tfidf_top_words.png       ← Top TF-IDF keywords per class
    ├── 03_model_comparison.png      ← Baseline model accuracy vs F1 bar chart
    ├── 04_confusion_and_metrics.png ← Confusion matrix + per-class metrics
    ├── 05_lstm_training_curves.png  ← LSTM training/validation curves
    ├── model_evaluation.csv         ← Numeric results for all models
    └── sample_predictions.txt       ← 10 sample predictions with confidence
```

---

## Tasks Completed

| Task | Description |
|------|-------------|
| **Task 1** | Dataset understanding — records, labels, class distribution, average word count |
| **Task 2** | Text preprocessing — lowercase, digit removal, punctuation, tokenization, stopwords |
| **Task 3** | Text vectorization — Bag of Words, TF-IDF (unigrams + bigrams), keyword analysis |
| **Task 4** | Baseline models — Logistic Regression, Naive Bayes, MLP Neural Network |
| **Task 5** | LSTM sequence model — full architecture design + Keras code + simulated training curves |
| **Task 6** | Reflection — RNN limitations, LSTM memory, attention mechanism, transformer importance |

---

## Model Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression (TF-IDF) | 1.00 | 1.00 |
| Naive Bayes (BoW) | 1.00 | 1.00 |
| MLP Neural Net (TF-IDF) | 1.00 | 1.00 |
| LSTM (simulated) | ~0.84 | ~0.82 |

> **Note:** The dataset is synthetic with a consistent vocabulary per sentiment class and a small number of unique template messages (~634 unique). This produces very high accuracy for classical models — expected behaviour on clean, separable synthetic data.

---

## Dataset Summary

- **1,500 rows**, 6 columns: `ticket_id`, `channel`, `customer_message`, `sentiment_label`, `word_count`, `urgent_flag`
- **3 balanced classes:** negative (497), neutral (524), positive (479)
- **Channels:** email, social, phone, chat, app
- **Avg message length:** 12.7 words (range: 7–26)
- **Key finding:** `urgent_flag=1` maps entirely to negative/neutral — zero urgent positive messages

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebook.ipynb
```

Place `customer_support_text_classification.csv` in the same directory and create a `results/` folder before running.

---

## Key Concepts Covered

- **Text preprocessing pipeline** with regex, tokenization, and custom stopword removal
- **TF-IDF vs Bag of Words** — how term weighting improves over raw counts
- **Logistic Regression and Naive Bayes** as strong, interpretable text classifiers
- **LSTM architecture** — embedding → spatial dropout → LSTM → dense → softmax
- **Attention mechanism** and why transformers replaced RNNs in modern NLP
