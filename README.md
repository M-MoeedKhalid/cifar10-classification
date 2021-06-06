# cifar10-classification

A machine learning based approach on classifying the famous [cifar-10!](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
The four files are for:
* load_data loads the cifar-10 dataset.
* cifar10_individual_classifiers.py contains code for classification of cifar-10 dataset using Guassian Naive Bayes Algorithm, a KNN (3-NN) algorithm and a Support Vector machine with rbf kernel
* cifar_10_soft_voting uses LDA, Random Forest Classifier and Gaussian Naive Bayes Algorithm to vote and get the best results out of these classifiers.
* cifar_10_plot_image can plot individual images of the dataset to get a better sense of what the dataset is actually about.
