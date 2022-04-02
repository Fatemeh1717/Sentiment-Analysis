
# Sentiment Analysis using BERT , CNN, BERT+CNN in Tensorflow

The goal of this project is to classifiy correctly whether movie reviews from IMDB are positive or negative. First,the traditional statiscical approaches called Bag of Words, TF-IDF (Term Frequency - Inverse Document Frequency), and Word2Vec have been investigated. At the second part I am going to apply CNN , BERT, CNN+BERT models to compute vector-space representations of a dataset and figure out how they work differently.

At the **First part** I will: 

*   Loading the IMDB dataset
*   Splitting the dataset between training and test set
*   Data Cleaning and Text Preprocessing
*   Visualization
*   Explaining Bag of Words
*   Explaining TF-IDF
*   Explaining Word2Vec
*   Compareing the approaches by Logistic Regression Model


**Second Part**
*   Build the Convolutional Neural Network Model
*   Train , evaluate the model using Confusion Matrix
*   Save the model 
*   Load a BERT models from TensorFlow Hub 
*   Build and train the BERT models called BERT, small BERT, and Albert 
*   Export the models for inference
*   Evaluate and save the model 
*   Use the pretreined BERT to feed to CNN downsteam Archtectures
*   Evaluate and save the model.
*   Compare the models

The experiment is implemented on the Google Colaboratory (Colab) that provides the Jupyter notebook environment and executes code in Python 3.10.4. The Colab supports the Tesla K80 GPU accelerator. The Keras front end runs on the TensorFlow backend. It enables fast experimentation of deep learning models by running code on the graph processing unit (GPU) and central processing unit (CPU). The classification performance of the models is evaluated by using the sklearn metrics

## Deployment

To deploy this project run

```bash
  !pip install 'tensorflow==2.8.0'
```
```bash
  !pip install 'tf-estimator-nightly==2.8.0.dev2021122109'
```
```bash
  !pip install tensorflow-text
```
```bash
  !pip install tf-models-official==2.7.0
```
And also:

```bash
  import numpy as np
```
```bash
  import pandas as pd
`````
```bash
  from scipy.stats import spearmanr
``````
```bash
  from sklearn.model_selection import GroupKFold
``````
```bash
  from sklearn.model_selection import train_test_split
``````
```bash
  from sklearn.metrics import confusion_matrix
``````
```bash
  import matplotlib.pyplot as plt
``````
```bash
  import tensorflow as tf
``````
```bash
  import tensorflow_hub as hub
``````
```bash
  import tensorflow_text as text
``````
```bash
  from tensorflow import keras
``````
```bash
  from tensorflow.keras import layers
``````
```bash
  from official.modeling import tf_utils
``````
```bash
  from official import nlp
``````
```bash
  from official.nlp import bert
```
```bash
  from sklearn.metrics import confusion_matrix,classification_report
```
```bash
  import seaborn as sns
```



## Dataset
We used the IMDB dataset (Maaset al., 2011) including 50K movie reviews with two labels. These 50k short texts belong to 2 classes which are 1 (positive/good review) and 0 (negative/bad review).

```bash
   review           sentiment        
 Length:50000       Length:50000      
 Class :character   Class :character  
 Mode  :character   Mode  :character  
```

```bash
  df.head()	
```
```bash
                            text            	    label	category
0	One of the other reviewers has mentioned that ...	1	positive
1	A wonderful little production. <br /><br />The...	1	positive
2	I thought this was a wonderful way to spend ti...	1	positive
3	Basically there's a family where a little boy ...	0	negative
4	Petter Mattei's "Love in the Time of Money" is...	1	positive
...	...	...	...
495	"American Nightmare" is officially tied, in my...	0	negative
496	First off, I have to say that I loved the book...	0	negative
497	This movie was extremely boring. I only laughe...	0	negative
498	I was disgusted by this movie. No it wasn't be...	0	negative
499	Such a joyous world has been created for us in...	1	positive
```

## Splitting the data into trainning and test set
```bash
 X_train, X_test, y_train, y_test = train_test_split(
    df['text'],y,
    test_size=0.33, 
    stratify=y   
) 
```
```dash
df_train.head()
``` 
```bash
	                                        text	label
360	The endless bounds of our inhumanity to our ow...	1
499	Such a joyous world has been created for us in...	1
436	This is an excellent film about the characters...	1
290	I saw the movie "Hoot" and then I immediately ...	1
408	I watched this movie based on the good reviews...	0
```
```bash
df_test.head()
```

```bash
                                        text    	label
271	This is a must-see documentary movie for anyon...	1
127	The complaints are valid, to me the biggest pr...	0
270	Clifton Webb is one of my favorites. However, ...	0
64	An unmarried woman named Stella (Bette Midler)...	0
62	So let's begin!)))<br /><br />The movie itself...	1
```

## Initail look at the Data
Understanding how terms are distributed across documents help us to characterize the properties of the algorithms for compressing phrases.

The total Term Frequency of all the 2 sentiment classes for the top 20 words is as following
```bash
    negativ positive total
Terms			
the	218620	227559	446179
and	99101	117707	216808
of	91960	101242	193202
to	91300	87886	179186
is	66472	74666	141138
br	69027	64642	133669
it	64121	63435	127556
in	58361	66060	124421
this54481	46622	101103
that49797	46248	96045
br br34536	32350	66886
was	35118	28801	63919
as	27292	34003	61295
movie33567	25323	58890
for	29236	29606	58842
with28055	30606	58661
but	28571	27069	55640
film24955	28439	53394
ofthe23734	27902	51636
you	23827	22364	46191

```
## Data Cleaning and Text Preprocessing

Removing Stop words: "Stop words" is the frequently occurring words that do not carry much meaning such as "a", "and" , "is", "the". In order to use the data as input for machine learning algorithms, we need to get rid of them. Fortunately, there is a function called stopwords which is already built in NLTK library.

Stemming: Stemming algorithms work by cutting off the end of the word, and in some cases also the beginning while looking for the root. This indiscriminate cutting can be successful in some occasions, but not always, that is why we affirm that this an approach that offers some limitations. ex) studying -> study, studied -> studi

Lemmatization: Lemmatization is the process of converting the words of a sentence to its dictionary form. For example, given the words amusement, amusing, and amused, the lemma for each and all would be amuse. ex) studying -> study, studied -> study. Lemmatization also discerns the meaning of the word by understanding the context of a passage. For example, if a "meet" is used as a noun then it will print out a "meeting"; however, if it is used as a verb then it will print out "meet".

Usually, either one of them is chosen for text-analysis not both. As a side note, Lancaster is the most aggressive stemmer among three major stemming algorithms (Porter, Snowball, Lancaster) and Porter is the least aggressive. The "aggressive algorithms" means how much a working set of words are reduced. The more aggressive the algorithms, the faster it is; however, in some certain circumstances, it will hugely trim down your working set. Therefore, in this project I decide to use snowball since it is slightly faster than Porter and does not trim down too much information as Lancaster does.

Tokenization: Tokenization is the process splitting a sentence or paragraph into the most basic units

```bash
def prep(text):
    
    # Remove HTML tags.
    text = BeautifulSoup(text,'html.parser').get_text()
    
    # Remove non-letters
    text = re.sub("[^a-zA-Z]", " ", text)
    
    # Lower case
    text = text.lower()
    
    # Tokenize to each word.
    tokens = nltk.word_tokenize(text)
    
    # Stemming
    text = [nltk.stem.SnowballStemmer('english').stem(w) for w in tokens]

    # 5. Convert the stopwords list to "set" data type.
    stops = set(nltk.corpus.stopwords.words("english"))
    
    # 6. Remove stop words. 
    words = [w for w in tokens if not w in stops]
    
    # Join the words back into one string separated by space, and return the result.
    return " ".join(text)
```

## Bag of Words
The traditional method uses bag-of-words (BoW) to represent documents. Although the BoW model is effective, this method only regards documents as a collection of words, each word in the text is independent and ignores the variability of word meaning in different linguistic contexts (e.g., polysemy) as well as word order, grammar, and syntactic structure. For instance, for these two sentences: “He deposited his money in this bank.” and “His soldiers were arrayed along the river bank.”, the word bank has different meanings in different contexts.

In order to do that, we use "CountVectorizer" method in sklearn library. As you know already, the number of vocabulary is very large so it is important to limit the size of the feature vectors. In this project, we use the 18000 most frequent words. Also, the other things to notice is that we set min_df = 2 and ngram_range = (1,3). min_df = 2 means in order to include the vocabulary in the matrix, one word must appear in at least two documents. ngram_range means we cut one sentence by number of ngram. Let's say we have one sentence, I am a human. If we cut the sentence by digram (ngram=2) then the sentence would be cut like this ["I am","am a", "a human"]. The result of accuracy can be highly dependent on parameters.

```bash
vectorizer = CountVectorizer(analyzer = "word", 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 18000,
                             min_df = 2,
                             ngram_range = (1,3)
                            )
```
 the matrix is going to be huge so it would be a good idea to use Pipeline for encapsulating and avoiding a data leakage.

 ```bash
 pipe = Pipeline( [('vect', vectorizer)] )
 ```

## Modelling

```bash
kfold = StratifiedKFold(n_splits=5)
lr_CoVec = LogisticRegression()


lr2_param = {
    'penalty':['l2'],
    'dual':[False],
    'C':[0.05],
    'class_weight':['balanced']
    }

lr_CV = GridSearchCV(lr_CoVec, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = -1, verbose = 1)
lr_CV.fit(train_bw, df_train_stat['label'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_
```

```bash
print(lr_CV.best_score_)
```

```bash
0.9283554644687013
```

## TF-IDF (Term Frequency - Inverse Document Frequency)
The second approch is TF-IDF in which instead of CountVectorizer, we will be analyzing the movie reviews by using TF-IDF (Term Frequency - Inverse Document Frequency). Also, we could compare how differently these methods work and the performance of the predictions.

```bash
tv_model = TfidfVectorizer(
                    ngram_range = (1,3),
                    sublinear_tf = True,
                    max_features = 40000)
 ```

 ## Modelling
 ```bash
 lr_TF = LogisticRegression()

lr2_param = {
    'penalty':['l2'],
    'dual':[False],
    'C':[6],
    'class_weight':[{1:1}]
    }

lr_TF_IDF = GridSearchCV(lr_TF, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_TF_IDF.fit(train_tv, df_train_stat['label'])
print(lr_TF_IDF.best_params_)
logi_best = lr_TF_IDF.best_estimator_
 ```   
 ```bash
 print(lr_TF_IDF.best_score_)
 ```                
```bash
0.9421916328803743
```

## Word2Vec
For a bit deeper sentiment analysis, we will be using Word2Vec to train a model. Word2Vec is a neural network based algorithm that creates distributed word vectors. One of the advantages of Word2Vec over Bag Of Words, it understands meaning and semantic relationships among words. Also, it does not require labels in order to create meaningful representations. In addition that in a aspect of speed, it learns quickly compared to other methods. Before we dive into modeling, there is one more step to prepare two different matrices created by using Word2Vec by vector averaging method. We will test them independently and choose the best one among the results predicted by different methods including 'bag of words' and 'TF-IDF' and Word2vec'.

```bash
def preprocess_wordlist(data, stopwords = False):
    
    # Remove HTML tag
    review = BeautifulSoup(data,'html.parser').get_text()
    
    # Remove non-letters
    review = re.sub('[^a-zA-Z]', ' ', review)
    
    # Convert to lower case
    review = review.lower()
    
    # Tokenize
    word = nltk.word_tokenize(review)
    
    # Optional: Remove stop words (false by default)
    if stopwords:
        stops = set(nltk.corpus.stopwords.words("english"))
        
        words = [w for w in word if not w in stops]
    
    return word
   ```

   For creating a model, unlike Bag Of Words method, it is not necessary to remove the stopwords such as "the", "a", "and", etc as the algorithm relies on the broader context of the sentence in order to produce high-quality word vectors.

   ```bash
   def preprocess_sent(data, stopwords = False):
    
    # Split the paragraph into sentences
    
    #raw = tokenizer.tokenize(data.strip())
    raw = nltk.sent_tokenize(data.strip())
    
    # If the length of the sentence is greater than 0, plug the sentence in the function preprocess_wordlist (clean the sentence)
    sentences = [preprocess_wordlist(sent, stopwords) for sent in raw if len(sent) > 0]
    
    return sentences
```

## Combining Labeled and Unlabeled Reviews

```bash
sentence = []

# Append labeled reviews first
for review in df_train_stat['text']:
    sentence += preprocess_sent(review)
    
# Append unlabeled reviews
for review in unlabel_train['review']:
    sentence += preprocess_sent(review)
```

In general, there are two types of architecture options: skip-gram (default) and CBOW (continuous bag of words). Most of time, skip-gram is little bit slower but has more accuracy than CBOW. 

```bash
num_features = 400
min_count = 40
num_processor = 4
context = 10
downsampling = 0.001

Word2vec_model = word2vec.Word2Vec(sentence, workers = num_processor, 
                         size = num_features, min_count = min_count,
                         window = context, sample = downsampling)
```
    
## Vector Averaging
```bash
def makeFeatureVec(review, model, num_features):
    
    featureVec = np.zeros((num_features,), dtype = "float32")
    
    # Unique word set
    word_index = set(model.wv.index2word)
    
    # For division we need to count the number of words
    nword = 0
    
    # Iterate words in a review and if the word is in the unique wordset, add the vector values for each word.
    for word in review:
        if word in word_index:
            nword += 1
            featureVec = np.add(featureVec, model[word])
    
    # Divide the sum of vector values by total number of word in a review.
    featureVec = np.divide(featureVec, nword)        
    
    return featureVec

```

The purpose of this function is to combine all the word2vec vector values of each word in each review if each review is given as input and divide by the total number of words.

```bash
def getAvgFeatureVec(clean_reviews, model, num_features):
    
    # Keep track of the sequence of reviews, create the number "th" variable.
    review_th = 0
    
    # Row: number of total reviews, Column: number of vector spaces (num_features = 250 we set this in Word2Vec step).
    reviewFeatureVecs = np.zeros((len(clean_reviews), num_features), dtype = "float32")
    
    # Iterate over reviews and add the result of makeFeatureVec.
    for review in clean_reviews:
        reviewFeatureVecs[int(review_th)] = makeFeatureVec(review, model, num_features)
        
        # Once the vector values are added, increase the one for the review_th variable.
        review_th += 1
    
    return reviewFeatureVecs
```

While iterating over reviews, add the vector sums of each review from the function "makeFeatureVec" to the predefined vector whose size is the number of total reviews and the number of features in word2vec. The working principle is basically same with "makeFeatureVec" but this is a review basis and makeFeatureVec is word basis (or each word's vector basis)

```bash
clean_train_reviews = []

# Clean the reviews by preprocessing function with stopwords option "on".
for review in df_train_stat["text"]:
    clean_train_reviews.append(preprocess_wordlist(review, stopwords = True))

# Apply "getAvgFeatureVec" function.
trainDataAvg = getAvgFeatureVec(clean_train_reviews, Word2vec_model, num_features)
    
    
# Same steps repeats as we did for train_set.    
clean_test_reviews = []

for review in df_test_stat["text"]:
    clean_test_reviews.append(preprocess_wordlist(review, stopwords = True))

testDataAvg = getAvgFeatureVec(clean_test_reviews, Word2vec_model, num_features)
```
Notice that we use stop word removal, which would just add noise. We will compare the performance of vector averaging method and the next method.

## Modelling

```bash
lr = LogisticRegression()


lr_param2 = {
    'penalty':['l1'],
    'dual':[False],
    'C':[40],
    'class_weight':['balanced'],
    'solver':['saga']
    
}

lr_CV = GridSearchCV(lr, param_grid = [lr_param2], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(trainDataAvg,df_train_stat['label'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_

```

```bash
print(lr_CV.best_score_)
```

```bash
Word2vec: 0.9509191089329472
```

## Conclusion
All of the models have been trained by Logistic Regression and the result obtained from the models are as the following:
```bash
Bag-of-Words: 0.9283554644687013
TF-IDF : 0.9421916328803743 
Word2vec: 0.9509191089329472
```
The best result is 95.09 % performed by Word2vec. 

# Second part

# Convolutional Neural Network(CNN)
Compared with traditional classifiers, neural networks provide greater expressive power and produce better performance. CNN is the popular deep learning technique to solve classification problems. CNNs are inspired by a biological variation of Multi Layer Perceptron (MLPs). They are very similar to ordinary neural networks. In a MLP each neuron has their separate weight vector but neurons in CNN share weights. This sharing of weights helps to reduce the overall number of traininable weight, thus reducing feature dimensionality, hence introducing sparsity.

There are three main types of layers in CNN architecture:

Convolutional Layer: The job of the convolutional layer is feature extraction. It learns to find spatial features in an input image. This layer is produced by applying a series of different image filters to an input image. These filters are known as convolutional kernels. A filter is a small grid of values that slides over the input image pixel by pixel to produce a filtered output image that will be of the same size as the input image.

Pooling Layer: After the convolutional layer comes the pooling layer; the most common type of pooling layer is maxpooling layer. The main goal of the pooling layer is dimensionality reduction, meaning reducing the size of an image by taking the max value from the window. A maxpooling operation breaks an image into smaller patches. A maxpooling layer is defined by a patch size and stride.

Fully Connected Layer : The last layer in CNN is the fully connected layer. Fully connected means that every output that’s produced at the end of the last pooling layer is an input to each node in this fully connected layer.The role of the fully connected layer is to produce a list of class scores and perform classification based on image features that have been extracted by the previous convolutional and pooling layers. So, the last fully connected layer will have as many nodes as there are classes

## Build the CNN Model

```bash
from tensorflow.keras.preprocessing.text import Tokenizer
max_words=10000
tokenizer=Tokenizer(max_words)
tokenizer.fit_on_texts(df_CNN_train)
sequence_train=tokenizer.texts_to_sequences(df_CNN_train)
sequence_test=tokenizer.texts_to_sequences(df_CNN_test)
word2vec=tokenizer.word_index
data_train=pad_sequences(sequence_train)
data_train.shape
data_test=pad_sequences(sequence_test,maxlen=T)
data_test.shape
D=20
i=Input((T,))
x=Embedding(V+1,D)(i)
x=Conv1D(32,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=GlobalMaxPooling1D()(x)
x=Dense(5,activation='softmax')(x)
model=Model(i,x)
model.summary()
```
 ## Model summary
```bash
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 1025)]            0         
                                                                 
 embedding (Embedding)       (None, 1025, 20)          215940    
                                                                 
 conv1d (Conv1D)             (None, 1023, 32)          1952      
                                                                 
 max_pooling1d (MaxPooling1D  (None, 341, 32)          0         
 )                                                               
                                                                 
 conv1d_1 (Conv1D)           (None, 339, 64)           6208      
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 113, 64)          0         
 1D)                                                             
                                                                 
 conv1d_2 (Conv1D)           (None, 111, 128)          24704     
                                                                 
 global_max_pooling1d (Globa  (None, 128)              0         
 lMaxPooling1D)                                                  
                                                                 
 dense (Dense)               (None, 5)                 645       
                                                                 
=================================================================
Total params: 249,449
Trainable params: 249,449
Non-trainable params: 0
```



```bash
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=[metrics])

```
```bash
CNN_history = model.fit(data_train,y_CNN_train,validation_data=(data_test,y_CNN_test),epochs=10,batch_size=100)
```
**Hyper-parameters of the CNN model**
```bash
Hyper-parameters	Values
Loss Function	sparse categorical cross entropy
Optimizer	Adam
Epochs	10
Dense	5
Convolutional size	3× 3
Kernel sizes	2 × 2
Conv1D	32, 64,128
Activation	relu
Bach size 	128

```
## CNN model evaluation
```bash
516/516 [==============================] - 2s 4ms/step - loss: 0.6883 - accuracy: 0.8610
Loss: 0.6882577538490295
Accuracy: 0.8609697222709656
```

```bash
print(classification_report(y_CNN_test,y_CNN_pred))
```

```bash
                precision    recall  f1-score   support

           0       0.85      0.87      0.86      8208
           1       0.87      0.85      0.86      8292

    accuracy                           0.86     16500
   macro avg       0.86      0.86      0.86     16500
weighted avg       0.86      0.86      0.86     16500           
         
```

## Save the model

Now the model will be saved for later use.
```bash
dataset_name = 'CNN_IMDB'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

model.save(saved_model_path, include_optimizer=False)
```


## BERT models
Language models such as word2vec have some limitation when interpreting context and polysemous words. BERT effectively addresses ambiguity, which is the greatest challenge to natural language understanding according to research scientists in the field. It is capable of parsing language with a relatively human-like "common sense".

We use the Tensorflow to build the BERT model.So, the first step is passing the trian and test test to tensorflow.
## Build TensorFlow input
While tf.data tries to propagate shape information, the default settings of Dataset.batch result is an unknown batch size because the last batch may not be full. Note the Nones in the shape.


```bash
train_ds = tf.data.Dataset.from_tensor_slices((df_train.text.values, df_train.label.values))
test_ds = tf.data.Dataset.from_tensor_slices((df_test.text.values, df_test.label.values))
```
## Loading models from TensorFlow Hub
This TF Hub model uses the implementation of BERT from the TensorFlow Models repository on GitHub at tensorflow/models/official/nlp/bert which L is used as a number of hidden layers and H as hidden size of H, and A as attention heads.

Here you can choose which BERT model you will load from TensorFlow Hub and fine-tune. There are multiple BERT models available.

BERT-Base, Uncased and seven more models with trained weights released by the original BERT authors.

The Small BERT models are instances of the original BERT architecture with a smaller number L of layers (i.e., residual blocks) combined with a smaller hidden size H and a matching smaller number A of attention heads.

ALBERT: four different sizes of "A Lite BERT" that reduces model size (but not computation time) by sharing parameters between layers.

```bash
bert_model_name = "albert_en_base" #@param ["bert_en_uncased_L-12_H-768_A-12", "small_bert/bert_en_uncased_L-4_H-512_A-8", "albert_en_base"]

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
```


## Preprocessing the model

Text inputs need to be transformed to numeric token ids and arranged in several Tensors before being input to BERT. TensorFlow Hub provides a matching preprocessing model for each of the BERT models discussed above, which implements this transformation using TF ops from the TF.text library. It is not necessary to run pure Python code outside your TensorFlow model to preprocess text.

So, the preprocessing model will be loaded into a hub.KerasLayer to compose the fine-tuned model.
The BERT models return a map with 3 important keys: pooled_output, sequence_output, encoder_outputs:

pooled_output represents each input sequence as a whole. The shape is [batch_size, H]. You can think of this as an embedding for the entire movie review.

sequence_output represents each input token in the context. The shape is [batch_size, seq_length, H]. You can think of this as a contextual embedding for every token in the movie review.

encoder_outputs are the intermediate activations of the L Transformer blocks. outputs["encoder_outputs"][i] is a Tensor of shape [batch_size, seq_length, 1024] with the outputs of the i-th Transformer block, for 0 <= i < L. The last value of the list is equal to sequence_output

```bash
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
```

## Define the model

The fine-tuned model will be created with the preprocessing model, and the selected BERT model, one Dense and a Dropout layer.

```bash
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)
  ```
  ## Model summary
```bash
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 text (InputLayer)              [(None,)]            0           []                               
                                                                                                  
 preprocessing (KerasLayer)     {'input_mask': (Non  0           ['text[0][0]']                   
                                e, 128),                                                          
                                 'input_word_ids':                                                
                                (None, 128),                                                      
                                 'input_type_ids':                                                
                                (None, 128)}                                                      
                                                                                                  
 BERT_encoder (KerasLayer)      {'encoder_outputs':  28763649    ['preprocessing[0][0]',          
                                 [(None, 128, 512),               'preprocessing[0][1]',          
                                 (None, 128, 512),                'preprocessing[0][2]']          
                                 (None, 128, 512),                                                
                                 (None, 128, 512)],                                               
                                 'sequence_output':                                               
                                 (None, 128, 512),                                                
                                 'pooled_output': (                                               
                                None, 512),                                                       
                                 'default': (None,                                                
                                512)}                                                             
                                                                                                  
 dropout (Dropout)              (None, 512)          0           ['BERT_encoder[0][5]']           
                                                                                                  
 classifier (Dense)             (None, 1)            513         ['dropout[0][0]']                
                                                                                                  
==================================================================================================
Total params: 28,764,162
Trainable params: 28,764,161
Non-trainable params: 1
```
 
  
  ## Model training
  I now have all the pieces to train a model, including the preprocessing module, BERT encoder, data, and classifier.
  Since this is a binary classification problem and the model outputs a probability (a single-unit layer), I'll use losses.BinaryCrossentropy loss function.

  ```bash
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metrics = tf.metrics.BinaryAccuracy()
  ```
## Optomizer

For fine-tuning, let's use the same optimizer that BERT was originally trained with: the "Adaptive Moments" (Adam). This optimizer minimizes the prediction loss and does regularization by weight decay (not using moments), which is also known as AdamW.

For the learning rate (init_lr), I will use the same schedule as BERT pre-training: linear decay of a notional initial learning rate, prefixed with a linear warm-up phase over the first 10% of training steps (num_warmup_steps). In line with the BERT paper, the initial learning rate is smaller for fine-tuning (best of 5e-5, 3e-5, 2e-5).

```bash
epochs = 10
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
```
 **Hyper-parameters of the BERT model**
 
  ```bash
  Hyper-parameters	Values
Learning rate	3e-5
Loss Function	Categorical Cross-entropy
Optimizer	Adam
Batch size	32
Dropout	0.1
Epochs	10
Dense  1  
```
## Evaluate the models

Comparison of the BERT models performance using Accuracy and Loss are shown as follows.


**bert_en_uncased_L-12_H-768_A-12**:
```bash
516/516 [==============================] - 182s 351ms/step - loss: 0.9129 - binary_accuracy: 0.8981
Loss: 0.9128670692443848
Accuracy: 0.8980606198310852
```

**small_bert/bert_en_uncased_L-4_H-512_A-8:**

```bash
516/516 [==============================] - 53s 103ms/step - loss: 0.8582 - binary_accuracy: 0.8698
Loss: 0.8582355380058289
Accuracy: 0.8698182106018066
```
      
**albert_en_base:**

```bash
516/516 [==============================] - 164s 317ms/step - loss: 0.7218 - binary_accuracy: 0.5132
Loss: 0.7217753529548645
Accuracy: 0.5132121443748474
```       
It seems that BERT_uncased has higher accuracy than others.



## Comparison of the prediction's capability among BERT modles
We can try any arbitary sentence to see the score of each models prediction.For example for the following sentences the results would be:
```bash
examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]
```

**Results from the bert en_uncased model:**
```bash
Results from the saved model:
input: this is such an amazing movie! : score: 0.996642
input: The movie was great!           : score: 0.942301
input: The movie was meh.             : score: 0.001411
input: The movie was okish.           : score: 0.012526
input: The movie was terrible...      : score: 0.000074
```

**Results from the small_bert/bert_en_uncased model**


```bash
Results from the saved model:
input: this is such an amazing movie! : score: 0.999300
input: The movie was great!           : score: 0.990146
input: The movie was meh.             : score: 0.189059
input: The movie was okish.           : score: 0.000025
input: The movie was terrible...      : score: 0.000027
```

**Results from the albert_en_base model**
```bash
Results from the saved model:
input: this is such an amazing movie! : score: 0.684056
input: The movie was great!           : score: 0.682430
input: The movie was meh.             : score: 0.672123
input: The movie was okish.           : score: 0.663330
input: The movie was terrible...      : score: 0.689129

```
As you can see among differents type of models BERT_uncased acheived the better result interms performance metrics such as of Accuracy ,Loss and the prediction's capability. So, we will feed the proposed model with the BERT-base model with a hidden size of 768, 12 self-attention heads and 12 Transformer.
## Save the model
Now the fine-tuned model will be saved for later use.

```bash
dataset_name = 'BERT_IMDB'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)
```
# BERT + CNN
In the last part, we will inspect finetuning models by adding simple fully connected Convolutional Neural Network (CNN) layers. BERT takes the final hidden state of the first token [CLS] as the representation of the whole sequence. The most significant difference between our proposed method and the BERT-Base models is that we use the hidden state of all the final outputs of the BERT as the contextualized word vector. Then we use CNN to extract the high-level features, finally, the max-pooling layer retains the most important features for text classification.
```bash
def build_CNN_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    #net = outputs['pooled_output'] # [batch_size, 768].
    net = sequence_output = outputs["sequence_output"] # [batch_size, seq_length, 768]
      
    
    net = tf.keras.layers.Conv1D(32, (2), activation='relu')(net)
    #net = tf.keras.layers.MaxPooling1D(2)(net)
    
    net = tf.keras.layers.Conv1D(64, (2), activation='relu')(net)
    #net = tf.keras.layers.MaxPooling1D(2)(net)
    net = tf.keras.layers.GlobalMaxPool1D()(net)
    
#    net = tf.keras.layers.Flatten()(net)
    
    net = tf.keras.layers.Dense(512, activation="relu")(net)
    
    net = tf.keras.layers.Dropout(0.1)(net)
#   net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    net = tf.keras.layers.Dense(3, activation="softmax", name='classifier')(net)
    
    return tf.keras.Model(text_input, net)
```

ReLU stands for rectified linear unit, and is a type of activation function. Mathematically, it is defined as y = max(0, x). It is linear (identity) for all positive values, and zero for all negative values.

Also, tf.keras.layers.Conv1D creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs.

Tf.keras.layers.GlobalMaxPool1D() downsamples the input representation by taking the maximum value over the time dimension.

The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True). These are all attributes of Dense.

## Model summary

```bash
Model: "model_2"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 text (InputLayer)              [(None,)]            0           []                               
                                                                                                  
 preprocessing (KerasLayer)     {'input_word_ids':   0           ['text[0][0]']                   
                                (None, 128),                                                      
                                 'input_type_ids':                                                
                                (None, 128),                                                      
                                 'input_mask': (Non                                               
                                e, 128)}                                                          
                                                                                                  
 BERT_encoder (KerasLayer)      {'encoder_outputs':  28763649    ['preprocessing[0][0]',          
                                 [(None, 128, 512),               'preprocessing[0][1]',          
                                 (None, 128, 512),                'preprocessing[0][2]']          
                                 (None, 128, 512),                                                
                                 (None, 128, 512)],                                               
                                 'default': (None,                                                
                                512),                                                             
                                 'pooled_output': (                                               
                                None, 512),                                                       
                                 'sequence_output':                                               
                                 (None, 128, 512)}                                                
                                                                                                  
 conv1d_3 (Conv1D)              (None, 127, 32)      32800       ['BERT_encoder[0][6]']           
                                                                                                  
 conv1d_4 (Conv1D)              (None, 126, 64)      4160        ['conv1d_3[0][0]']               
                                                                                                  
 global_max_pooling1d_1 (Global  (None, 64)          0           ['conv1d_4[0][0]']               
 MaxPooling1D)                                                                                    
                                                                                                  
 dense_1 (Dense)                (None, 512)          33280       ['global_max_pooling1d_1[0][0]'] 
                                                                                                  
 dropout_1 (Dropout)            (None, 512)          0           ['dense_1[0][0]']                
                                                                                                  
 classifier (Dense)             (None, 3)            1539        ['dropout_1[0][0]']              
                                                                                                  
==================================================================================================
Total params: 28,835,428
Trainable params: 28,835,427
Non-trainable params: 1
```
## Defining the weights for BERT+CNN the model.
```bash
#This is an balanced dataset.
positive, negative = np.bincount(df['label'])
total = positive + negative 
weight_for_0 = (1 /positive)*(total)/2.0 
weight_for_1 = (1 / negative)*(total)/2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
```

## Build the Model
We define Loss an Mertics as following:
```bash
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#metrics = tf.metrics.CategoricalCrossentropy()
metrics = tf.metrics.Accuracy()
```

```bash
epochs = 10
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

cnn_classifier_model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))
                          
```
**Hyper-parameters of the BERT-CNN model**
```bash
Hyper-parameters	Values
Learning rate	3e-5
Loss Function	Categorical Cross-entropy
Optimizer	Adam
Batch size	768
Dropout	0.1
Convolutional size	2 × 2
Kernel sizes	 [2*2]
Epochs	10
Class_Weight	             {1:1}
Dense	512,3
Conv1d	32,64
Activation	relu

```

## Model Evaluation
```bash
516/516 [==============================] - 182s 351ms/step - loss: 0.8661 - accuracy: 0.9007
Loss: 0.8660560250282288
Accuracy: 0.9006666541099548
```


## Save the model

Now we just save the fine-tuned model for later use

```bash
dataset_name = 'BERTCNN_IMDB'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

cnn_classifier_model.save(saved_model_path, include_optimizer=False)
```




## Models performance comparison
The evaluation result for CNN, BERT, CNN+BERT models are reported as follows. It shows that the proposed model acquired more accuracy.

**CNN** 
```bash
516/516 [==============================] - 2s 4ms/step - loss: 0.6883 - accuracy: 0.8610
Loss: 0.6882577538490295
Accuracy: 0.8609697222709656
```
**BERT**
```bash
516/516 [==============================] - 182s 351ms/step - loss: 0.9129 - binary_accuracy: 0.8981
Loss: 0.9128670692443848
Accuracy: 0.8980606198310852
```

**BERT+CNN**
```bash
516/516 [==============================] - 182s 351ms/step - loss: 0.8661 - accuracy: 0.9007
Loss: 0.8660560250282288
Accuracy: 0.9006666541099548
```
## Conclusion
In this part, we reported experiments with the proposed model in comparison with baseline models. The performance of our model was shown the above section. From the results, we can observe that:
* Some traditional methos discussed in this project achieved a high score accuracy but due to their limitations they cannot be used alone and on the large dataset such as yelp dataset.
* The CNN and BERT models were used as baseline model, while the BERT performs better than CNN and BERT+CNN works better than BERT and CNN.
* Our method achieved significant and consistent improvement as compared to other baselines. The reason is that our models can make full use of the information of contextualized word representations. In our model, after obtaining the rich contextual word representation, the convolutional network automatically extracts higher-level features, and then the most significant semantic features are selected by the max-pooling layer for classification. It verifies that incorporating the rich information of contextualized word representations could help us better correctly classify text.
