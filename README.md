# Document Classification using VGG-16 with Attention Mechanism
# Overview

This project focuses on document classification using the VGG-16 convolutional neural network architecture enhanced with an attention mechanism. The RVL-CDIP dataset is utilized for training and evaluating the model. The aim of this project is to develop an accurate and robust image classification system that can classify different types of documents effectively

# Features

Implementation of VGG-16 architecture: The project employs the VGG-16 convolutional neural network, a widely recognized deep learning architecture for image classification tasks.

Integration of Attention Mechanism: The attention mechanism is incorporated to allow the model to focus on relevant parts of the input images, improving classification accuracy.

RVL-CDIP Dataset: The Ryerson Vision Lab Complex Document Information Processing (RVL-CDIP) dataset is used for training and testing the model. The dataset consists of a diverse collection of documents, making the classification task challenging.

Data Preprocessing: The dataset is preprocessed to ensure compatibility with the model. This includes resizing images to a consistent resolution, data augmentation to enhance model generalization, and normalization for improved convergence during training.

Model Evaluation: The trained model is evaluated using various performance metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and classification reports are generated to provide insights into the model's performance on different document categories.

# Libraries and Frameworks
1. PyTorch
2. NumPy
3. Torchvision
4. scikit-learn

# Data Prepration
  Dataset can be downloaded from https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test
  =>Train/
      letter/
      form/
      email/
      handwritten/
      advertisement/
      scientific report/
      scientific publication/
      specification/
      file folder/
      news article/
      budget/
      invoice/
      presentation/
      questionnaire/
      resume/
      memo/
  =>Test/
      letter/
      form/
      email/
      handwritten/
      advertisement/
      scientific report/
      scientific publication/
      specification/
      file folder/
      news article/
      budget/
      invoice/
      presentation/
      questionnaire/
      resume/
      memo/
