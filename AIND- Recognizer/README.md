# Sign Language Recognizer

## 1. Synopsis

This project is to build word recognizer for American Sign Language (ASL) video using Hidden Markov Model. This project is consisted of 3 parts:
- **Part 1 - Data Exploration and Feature Engineering**: Clean and modify, if necessary, data in ASL dataset then select relevent features set to train sign language model.
- **Part 2 - Model Selection**: Implement model selector for different criteria.
- **Part 3 - Recognizer**: Implement recognizer from ASL data based on different models and features set.

## 2. Repository Structure

- `asl_recognizer.ipynb`: ASL Recognizer in HTML
- `asl_recognizer.ipynb`: ASL Recognizer in Jupyter Notebook
- `my_model_selector.py`: A model selector to select the best model from different model criteria - Log Liklihood with Cross Validation, Baysian Information Criteria, and Discriminative Information Criteria.
- `my_recognizer.py`: A recognizer which returns probability words with respect to sign language data input. The recognizer uses model object from `my_model_selector.py` as model for prediction.

