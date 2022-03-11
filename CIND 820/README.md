
# Sentiment Analysis using BERT , CNN, BERT+CNN in Tensorflow

 I am going to apply CNN , BERT, CNN+BERT Models to compute vector-space representations of a IMDB dataset which has two class
 
 1- **Positive:1**

 2- **Negative:0**
 
This notebook has three section. At the **First part** I will: 

*   Load the IMDB dataset
*   Load a **BERT models** from TensorFlow Hub 
*   Build the model by combining BERT with a classifier
*   Train the model, fine-tuning BERT as part of that
*   Save the model

**Second Part**
*   Build the Convolutional Neural Network Model
*   Train , evaluate the model using Confusion Matrix
*   Save the model 

**Third part** 
Includes using the pretreined BERT to compute vector-space representations of the IMDB dataset to feed to **CNN** downsteam Archtectures


# BERT

BERT (Devlin et al., 2019), as a contextualized word embedding and pre-trained language model, is a conceptually simple and empirically powerful representation for learning dynamic contextual embedding.  It is designed to pre-train deep bidirectional representations from the unlabeled text by jointly conditioning on both left and right context in all layers. Also, it can be used for a variety of language tasks by adding a small layer to the core model. In the other words, the variants of the BERT model have been developed to cater to different types of NLP-based systems such as classification.

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
    random_state=42, 
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
epochs = 40
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
```

## Evaluate the model

Let's see how the model performs. Two values will be returned. Loss (a number which represents the error, lower values are better), and accuracy.

```bash
loss, accuracy = classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
```

```bash
6/6 [==============================] - 13s 2s/step - loss: 0.4879 - binary_accuracy: 0.7697
Loss: 0.4878818392753601
Accuracy: 0.7696969509124756
```



## Confusion Matrix
```bash
print(classification_report(y_test_classes, y_pred_classes))
```

```bash
                precision    recall  f1-score   support

         0.0       0.67      0.67      0.67       111
         1.0       0.31      0.31      0.31        54

    accuracy                           0.55       165
   macro avg       0.49      0.49      0.49       165
weighted avg       0.55      0.55      0.55       165
```

## Save the model
Now the fine-tuned model will be saved for later use.

```bash
dataset_name = 'BERT_IMDB'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)
```
# Convolutional Neural Network(CNN)

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
CNN_history = model.fit(data_train,y_CNN_train,validation_data=(data_test,y_CNN_test),epochs=20,batch_size=100)
```



```bash
Epoch 1/30
4/4 [==============================] - 3s 521ms/step - loss: 1.5500 - accuracy: 0.4776 - val_loss: 1.4552 - val_accuracy: 0.4848
Epoch 2/30
4/4 [==============================] - 1s 379ms/step - loss: 1.3989 - accuracy: 0.5493 - val_loss: 1.2761 - val_accuracy: 0.4364
Epoch 3/30
4/4 [==============================] - 1s 376ms/step - loss: 1.1941 - accuracy: 0.5433 - val_loss: 1.0312 - val_accuracy: 0.4788
Epoch 4/30
4/4 [==============================] - 2s 383ms/step - loss: 0.9445 - accuracy: 0.7463 - val_loss: 0.8151 - val_accuracy: 0.4848
Epoch 5/30
4/4 [==============================] - 2s 422ms/step - loss: 0.7592 - accuracy: 0.5463 - val_loss: 0.7330 - val_accuracy: 0.4848
Epoch 6/30
4/4 [==============================] - 1s 369ms/step - loss: 0.6926 - accuracy: 0.5463 - val_loss: 0.7044 - val_accuracy: 0.4848
Epoch 7/30
4/4 [==============================] - 2s 453ms/step - loss: 0.6752 - accuracy: 0.5463 - val_loss: 0.7042 - val_accuracy: 0.4848
Epoch 8/30
4/4 [==============================] - 2s 405ms/step - loss: 0.6683 - accuracy: 0.5970 - val_loss: 0.6985 - val_accuracy: 0.4848
Epoch 9/30
4/4 [==============================] - 1s 370ms/step - loss: 0.6582 - accuracy: 0.5463 - val_loss: 0.7180 - val_accuracy: 0.4848
Epoch 10/30
4/4 [==============================] - 2s 378ms/step - loss: 0.6548 - accuracy: 0.5463 - val_loss: 0.7000 - val_accuracy: 0.4848
Epoch 11/30
4/4 [==============================] - 2s 403ms/step - loss: 0.6494 - accuracy: 0.5493 - val_loss: 0.6961 - val_accuracy: 0.4848
Epoch 12/30
4/4 [==============================] - 2s 400ms/step - loss: 0.6413 - accuracy: 0.7672 - val_loss: 0.6920 - val_accuracy: 0.5030
Epoch 13/30
4/4 [==============================] - 2s 387ms/step - loss: 0.6253 - accuracy: 0.8000 - val_loss: 0.7137 - val_accuracy: 0.4848
Epoch 14/30
4/4 [==============================] - 1s 358ms/step - loss: 0.6143 - accuracy: 0.6269 - val_loss: 0.6918 - val_accuracy: 0.5212
Epoch 15/30
4/4 [==============================] - 1s 253ms/step - loss: 0.6039 - accuracy: 0.9910 - val_loss: 0.7023 - val_accuracy: 0.4848
Epoch 16/30
4/4 [==============================] - 1s 255ms/step - loss: 0.5914 - accuracy: 0.5701 - val_loss: 0.6999 - val_accuracy: 0.4848
Epoch 17/30
4/4 [==============================] - 1s 246ms/step - loss: 0.5750 - accuracy: 0.9493 - val_loss: 0.6858 - val_accuracy: 0.5455
Epoch 18/30
4/4 [==============================] - 1s 246ms/step - loss: 0.5569 - accuracy: 0.9940 - val_loss: 0.7105 - val_accuracy: 0.4848
Epoch 19/30
4/4 [==============================] - 1s 247ms/step - loss: 0.5405 - accuracy: 0.6328 - val_loss: 0.6876 - val_accuracy: 0.5152
Epoch 20/30
4/4 [==============================] - 1s 242ms/step - loss: 0.5093 - accuracy: 0.9910 - val_loss: 0.6775 - val_accuracy: 0.6242
Epoch 21/30
4/4 [==============================] - 1s 241ms/step - loss: 0.4674 - accuracy: 0.9851 - val_loss: 0.6975 - val_accuracy: 0.4848
Epoch 22/30
4/4 [==============================] - 1s 239ms/step - loss: 0.4272 - accuracy: 0.9672 - val_loss: 0.6599 - val_accuracy: 0.6182
Epoch 23/30
4/4 [==============================] - 1s 243ms/step - loss: 0.3794 - accuracy: 0.9940 - val_loss: 0.6634 - val_accuracy: 0.5879
Epoch 24/30
4/4 [==============================] - 1s 247ms/step - loss: 0.3103 - accuracy: 0.9881 - val_loss: 0.6427 - val_accuracy: 0.6970
Epoch 25/30
4/4 [==============================] - 1s 249ms/step - loss: 0.2477 - accuracy: 0.9881 - val_loss: 0.6278 - val_accuracy: 0.7152
Epoch 26/30
4/4 [==============================] - 1s 253ms/step - loss: 0.1792 - accuracy: 0.9970 - val_loss: 0.6120 - val_accuracy: 0.7091
Epoch 27/30
4/4 [==============================] - 1s 239ms/step - loss: 0.1224 - accuracy: 1.0000 - val_loss: 0.6224 - val_accuracy: 0.6788
Epoch 28/30
4/4 [==============================] - 1s 256ms/step - loss: 0.0801 - accuracy: 1.0000 - val_loss: 0.5887 - val_accuracy: 0.6909
Epoch 29/30
4/4 [==============================] - 1s 253ms/step - loss: 0.0487 - accuracy: 1.0000 - val_loss: 0.5882 - val_accuracy: 0.6909
Epoch 30/30
4/4 [==============================] - 1s 250ms/step - loss: 0.0286 - accuracy: 1.0000 - val_loss: 0.6048 - val_accuracy: 0.7212
```


## Confusion Matrix

```bash
print(classification_report(y_CNN_test,y_CNN_pred))
```

```bash
                precision    recall  f1-score   support

           0       0.66      0.89      0.76        80
           1       0.84      0.56      0.68        85

    accuracy                           0.72       165
   macro avg       0.75      0.73      0.72       165
weighted avg       0.75      0.72      0.71       165
         
```

## Save the model

Now the model will be saved for later use.
```bash
dataset_name = 'CNN_IMDB'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

model.save(saved_model_path, include_optimizer=False)
```
# BERT + CNN
For this implementation , I am using the sequence_output as input to the convolutional layer. It represents each input token in the context. The shape is [batch_size, seq_length, H]. You can think of this as a contextual embedding for every token in the text. This outputs saves positional information about the inputs, then it would male cense to feed a convolutional layer.

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


## Model Evaluation
```bash
loss, accuracy = cnn_classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
```

```bash
6/6 [==============================] - 13s 2s/step - loss: 0.6346 - accuracy: 0.7333
Loss: 0.6345964074134827
Accuracy: 0.7333333492279053
```




## Confusion Matrix

```bash
print(classification_report(y_CNN_test,y_CNN_pred))
```
```bash
                precision    recall  f1-score   support

           0       0.66      0.89      0.76        80
           1       0.84      0.56      0.68        85

    accuracy                           0.72       165
   macro avg       0.75      0.73      0.72       165
weighted avg       0.75      0.72      0.71       165


```

## Save the model

Now we just save the fine-tuned model for later use

```bash
dataset_name = 'BERTCNN_IMDB'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

cnn_classifier_model.save(saved_model_path, include_optimizer=False)
```




### Export for inference 
 Now we  just saved our models for later use and you can test the model on any sentence you want, just add to the examples variable below.

 ```bash
 def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()


examples = [
    'this is such an amazing movie!', 
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]

 ```

 ```bash


print('Results from the BERT model:')
print_my_examples(examples, BERT_reloaded_results)

print('Results from the BERTCNN model :')
print_my_examples(examples, BERTCNN_reloaded_results)


 ```
 ```bash
 Results from the BERT model:
input: this is such an amazing movie! : score: 0.871721
input: The movie was great!           : score: 0.640763
input: The movie was meh.             : score: 0.660278
input: The movie was okish.           : score: 0.694434
input: The movie was terrible...      : score: 0.372601

Results from the BERTCNN model :
input: this is such an amazing movie! : score: 0.622488
input: The movie was great!           : score: 0.611779
input: The movie was meh.             : score: 0.593441
input: The movie was okish.           : score: 0.605346
input: The movie was terrible...      : score: 0.633656
 ```
 