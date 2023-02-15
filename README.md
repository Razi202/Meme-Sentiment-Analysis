# Meme Sentiment Analysis
In this project, I work on classifying the emotion of a meme based on image and text features.
The important aspect that I worked on during this project is the multi-modal architecture of neural networks.
Unlike a more conventional approach, I created a neural network which itself consisted of two more neural network architectures.
One is used for image classification and the other for text classification of the memes.
Normally, we'd train the two neural networks separately and combine there outputs to get a result, but here, in the multi-modal architecture, both text and image features
are used for training at the same time and each of their results effect the outcome since both now share the same optimizers for backward and forward propogations.

For the training, I created a custom dataset loader and trained the model in batches.

I have provided the notebook of the code and also the pdf where you can view the entire architecture diagram of the neural networks (given at the end of the pdf).
