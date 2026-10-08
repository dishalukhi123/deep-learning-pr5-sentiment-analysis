# DL PR5 — Sentiment Analysis with RNN, LSTM and GRU

## Project
Deep Learning PR5: sentiment analysis on the Kaggle IMDB Dataset using an order-blind ANN baseline, SimpleRNN, LSTM, Stacked LSTM and GRU.

## Dataset
Download **IMDB Dataset.csv** from the Kaggle dataset specified in the PR5 assignment and place it beside the notebook.

The notebook intentionally does not use `keras.datasets.imdb`.

## Notebook
`DL_PR5_Complete_Answer.ipynb`

## Main work
- Text cleaning, tokenisation and pre-padding
- Train/test split before tokenizer fitting
- ANN baseline
- Manual NumPy RNN forward pass verified against Keras
- SimpleRNN sentiment classification
- BPTT and gradient-flow experiments
- LSTM and GRU theory, parameter counts and models
- Stacked LSTM
- Hidden-unit sensitivity
- ROC curves and model comparison
- Review-length analysis
- Misclassification analysis
- Reusable `predict_sentiment()` inference function

## Plots
The notebook saves generated figures into `plots/`.

## Requirements
Install:

```bash
pip install -r requirements.txt
```

## Video
Add your own 5–10 minute MP4/MOV video link here after recording it. The assignment requires face picture-in-picture and screen sharing.

## GitHub commit examples
1. `Add text cleaning, tokenisation and padding pipeline`
2. `Add manual RNN forward pass and BPTT gradient analysis`
3. `Add LSTM and GRU models with comparison and inference`
