# Reading Comprehension QA system
In this repo, I built the the reading comprehension Question Answering system. I tried two approaches one using machine learning algorithms and another using deep learning NLP techniques. 
For training the model I have used Stanford Question Answering Datatset (https://rajpurkar.github.io/SQuAD-explorer/)

## Technical details
Embeddings: To create the feature vectors I used the Infersent model created by Facebook. The reason was solely based on using the sentence embeddings and infersent have better performance to downstream NLP tasks as experimented in this paper https://arxiv.org/abs/1705.02364. In this they have evaluated number of sentence embedding models and there model has performed better then the rest.

### Machine Learning Model:
I created two features cosine similarity and euclidean distance between the question and each sentence of the document. The target variable is the sentence ID having the correct answer. I have fitted the data with  multinomial logistic regression and random forest. Random forest has performed better. View the model in ml_model.ipynb

### Deep Learning Model: 
To improve on machnine learning model I have implemented the BiDaf model implemented using this https://arxiv.org/abs/1611.01603. I have used this Keras for building the model architecture. The model is trained with document and question vector built using the infersent. The model predicts answerBeginIndex, answerEndIndex in the document containing the answer. View the model in deep_learning_models.ipynb.

To view the demo I have implement a api, run the file api.py . I attached the image below of the request from the postman

## Steps
- Clone the repo 
- Download Infersent - Follow the steps here https://github.com/facebookresearch/InferSent.
- Run api.py to see the demo or play with any of the model

## Future Work: 
Implement state of art model BERT and Albert https://arxiv.org/abs/1909.11942
