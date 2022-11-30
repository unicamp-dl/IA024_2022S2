# Measuring and Reducing Toxicity and Bias in Language Models

This is the final project presented for IA024 - Natural Language Processing course of State University of Campinas, 2022.

## Objectives

Analyze different datasets and measurement methods for toxicity and bias detection on auto-regressive language models. The metrics obtained by different models will be compared to determine the correlations between the datasets measurements. Techniques for reducing toxicity and/or bias will be tested to verify their effectiveness over the different datasets.

## Proposed steps

- A search of evaluation datasets
- Enumeration of models to be tested and their characteristics
- Evaluation of models in different datasets
- Analysis of results to search for correlations between variables
- Implementation of toxicity reduction methods in text generation
- Evaluation of adapted models in different datasets
- Analysis of results to search for correlations between variables

## Datasets

 |Dataset| Input/Output|Metrics|Bias Group|
|---|---|---|---|
|RealToxicityPrompts|Sentence-Continuation|Perspective API| — |
|Winogender|Sentence-Next word|Accuracy|Gender|
|Winobias|Sentence-Next word|Accuracy|Gender|
|BOLD|Sentence-Continuation|Sentiment, toxicity, demographic and gender polarity|Profession - Gender - Race - Religion - Politics|
|Stereoset|Sentence-Logits|Sterotype probability|Profession - Gender - Race - Religion|
