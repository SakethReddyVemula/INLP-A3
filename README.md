# Training and testing codes
1. svd.py: Train the word embeddings using SVD method and save the word vectors.
2. skip-gram.py: Train the word embeddings using Skip gram method (with negative sampling) and save the word vectors.
3. svd-classification.py: Train any RNN on the classification task using the SVD word vectors
4. skip-gram-classification.py: Train any RNN on the classification task using the Skip-Gram word vectors.
# Pretrained models:
1. svd-word-vectors.pt: Saved word vectors for the entire vocabulary trained using SVD.
2. skip-gram-word-vectors.pt: Saved word vectors for the entire vocabulary trained using Skip-gram (using negative sampling).
3. svd-classification-model.pt: Saved model for the classification task trained using SVD word embeddings.
4. skip-gram-classification-model.pt: Saved model for the classification task trained using Skip-gram word embeddings.
# Report
report.pdf
# Observations and challenges faces:
1. Taking mean of hidden layers and reshaping it to the shape of a single hidden layers worked competitevely wrt taking last hidden layers.
2. Input sentence length is padded, and total length of padded sentence is limited. Otherwise, the model takes too much time and epoch to get trained. The weights of the model doesn't show considerable differences as shown by the loss function. The model outputs single output for all the inputs in test sets.
# Link to pretrained models, evaluation metrics and datasets
Google Drive link: https://drive.google.com/drive/folders/1B-7_9_oNGjonNAh8hoexlQAlN6ZTfdTW?usp=sharing
