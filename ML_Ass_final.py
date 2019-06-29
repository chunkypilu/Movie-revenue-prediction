#!/usr/bin/env python
# coding: utf-8

# ML ASSIGNMENT (Priyank Soni)
# Here idea is to use multi input model as there are both text and numeric data
'''
       text
        |
        |
 multi_hot_encoded           
        |
        |
        |
   embedded layer              numerical input
        |                            |                                    |                                     |
        |                            |
        |                            |
         |                          |
          |                        |
           |                      |
            |                    |
             |                  |
              |                |
               |              |
                |            |
                 |          |
                  |        |
                   |      |
                    
                  dense layers
                       |
                       |
                       |
                  output layer
                  
'''
# In[708]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model


# # Reading csv

# In[709]:


p=pd.read_csv("/home/priyank/Downloads/mini_project/data.csv")

#p.head()

keyword= p['Keywords']

overview= p['overview']
overview=(overview.astype(str)).as_matrix()


belongs_to_collection=p['belongs_to_collection']
belongs_to_collection=(belongs_to_collection.astype(str)).as_matrix()




revenue=p['revenue']
revenue=(revenue.astype(float)).as_matrix()
revenue = np.expand_dims(revenue, axis=1)

revenue_train= revenue[:2500]
revenue_test=revenue[2500:]




budget=p['budget']
budget=(budget.astype(float)).as_matrix()
budget = np.expand_dims(budget, axis=1)


genres=p['genres']
genres=(genres.astype(str)).as_matrix()



original_language=p['original_language']
original_language=(original_language.astype(str)).as_matrix()

original_title=p['original_title']
original_title=(original_title.astype(str)).as_matrix()


popularity=p['popularity']
popularity=(popularity.astype(float)).as_matrix()
popularity = np.expand_dims(popularity, axis=1)



runtime=p['runtime']
runtime=(runtime.astype(float)).as_matrix()
runtime = np.expand_dims(runtime, axis=1)

Keywords=p['Keywords']
Keywords=(Keywords.astype(str)).as_matrix()


spoken_languages=p['spoken_languages']
spoken_languages=(spoken_languages.astype(str)).as_matrix()

cast=p['cast']
cast=(cast.astype(str)).as_matrix()


production_companies=p['production_companies']
production_companies=(production_companies.astype(str)).as_matrix()






# In[710]:


revenue_train.shape


# # #########Things related to "overview" column
# 
# Here overview is text so we need to convert it into some meaningfull numeric data, First we are using a embedding layer and the flatten, output after flattening will be used as input for our model, we are using similar methodology for other text inputs, but some variations in layers used (for example embedding is not needed in  case of keywords)

# In[711]:


texts = []

for view in overview:
    texts.append(view)

maxlen = 100 #considering maxlength of a overview
max_words = 10000 #considering only 10000 most common words

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
#labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

data_overview=data.astype(float)


data_overview_train=data_overview[:2500]
data_overview_test=data_overview[2500:]



embedding_dim = 100


# Use Input layers, specify input shape (dimensions except first)
inp_multi_hot = keras.layers.Input(shape=(data_overview_train.shape[1],))


# Bind nulti_hot to embedding layer
emb = keras.layers.Embedding(input_dim=10000, output_dim=100)(inp_multi_hot)
flatten = keras.layers.Flatten()(emb)

print(flatten.shape)



# In[712]:


data_overview_train.shape


# # things related to "belongs_to_collection"
# 

# In[713]:


import re

b_to_c=[]
for i  in belongs_to_collection:
    i
    b_to_c.append(re.findall(r"name': '(.*?)',", i))
    
    
b_to_c_texts=[]
for i  in b_to_c:
    try:
     x= i[0]   
     b_to_c_texts.append(x)
    except:
        b_to_c_texts.append("NA")

       
# Tokenizing the text of the raw data
maxlen = 7  #max length of a sentence
max_words = 2000  #dic size

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(b_to_c_texts)
sequences = tokenizer.texts_to_sequences(b_to_c_texts)
        
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)        
        
    
data_b_to_c=data.astype(float)    

data_b_to_c_train=data_b_to_c[:2500]
data_b_to_c_test=data_b_to_c[2500:]



# Use Input layers, specify input shape (dimensions except first)
inp_multi_hot_b_t_c = keras.layers.Input(shape=(data_b_to_c_train.shape[1],))


# Bind nulti_hot to embedding layer
emb_b_t_c = keras.layers.Embedding(input_dim=10000, output_dim=100)(inp_multi_hot_b_t_c)
flatten_b_t_c = keras.layers.Flatten()(emb_b_t_c)


# In[714]:


data_b_to_c_train.shape


# In[ ]:





# # things related to "genere"

# In[715]:


import re

gen=[]

for i  in genres:
    i
    gen.append(re.findall(r"name': '(.*?)'}", i))
    

#find max len

max_len=0
for g in gen:
    l=len(g)
    if l > max_len:
        max_len=l  
        
        
        
maxlen = max_len
max_words = 20

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(gen)
sequences = tokenizer.texts_to_sequences(gen)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
#labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

data_generes=data.astype(float)


data_generes_train=data_generes[:2500]
data_generes_test=data_generes[2500:]


inp_multi_hot_generes = keras.layers.Input(shape=(data_generes_train.shape[1],))


# Bind nulti_hot to embedding layer
#emb = keras.layers.Embedding(input_dim=10000, output_dim=100)(inp_multi_hot)
#flatten = keras.layers.Flatten()(emb)

        


# In[716]:


inp_multi_hot_generes.shape


# In[ ]:





# In[ ]:





# In[ ]:





# # Original language

# In[717]:


org_lan=[]

for i  in original_language:
    i
    org_lan.append(i)
    
    
maxlen = 1
max_words = 40

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(org_lan)
sequences = tokenizer.texts_to_sequences(org_lan)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
#labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

data_org_lang=data.astype(float)

data_org_lang_train=data_org_lang[:2500]
data_org_lang_test=data_org_lang[2500:]


inp_multi_hot_org_lang = keras.layers.Input(shape=(data_org_lang_train.shape[1],))


# In[ ]:





# In[ ]:





# # Original title

# In[718]:



org_title=[]

for i  in original_title:
    i
    org_title.append(i)
    
    
maxlen = 8
max_words = 3500

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(org_title)
sequences = tokenizer.texts_to_sequences(org_title)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
#labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

data_org_title=data.astype(float)


data_org_title_train=data_org_title[:2500]
data_org_title_test=data_org_title[2500:]

inp_multi_hot_org_title = keras.layers.Input(shape=(data_org_title_train.shape[1],))


# In[ ]:





# # Keywords

# In[719]:


import re

Keywords_list=[]
for i  in Keywords:
    i
    Keywords_list.append(re.findall(r"name': '(.*?)'}", i))
    
    
Keywords_texts=[]
for i  in Keywords_list:
    try:
     x= i[0]   
     Keywords_texts.append(x)
    except:
        Keywords_texts.append("NA")
        
# Tokenizing the text of the raw data
maxlen = 1  #max length of a sentence
max_words = 1000  #dic size

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(Keywords_texts)
sequences = tokenizer.texts_to_sequences(Keywords_texts)
        
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)        
        
    
data_Keywords=data.astype(float)    

data_Keywords_train=data_Keywords[:2500]
data_Keywords_test=data_Keywords[2500:]


inp_multi_hot_Keywords = keras.layers.Input(shape=(data_Keywords_train.shape[1],))


# In[ ]:





# # spoken_languages
# 

# In[720]:


import re

spoken_lang_list=[]
for i  in spoken_languages:
    i
    spoken_lang_list.append(re.findall(r"name': '(.*?)'}", i))
    
    
spoken_lang_texts=[]
for i  in spoken_lang_list:
    try:
     x= i[0]   
     spoken_lang_texts.append(x)
    except:
        spoken_lang_texts.append("NA")

   
# Tokenizing the text of the raw data
maxlen = 1  #max length of a sentence
max_words = 50  #dic size

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(spoken_lang_texts)
sequences = tokenizer.texts_to_sequences(spoken_lang_texts)
        
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)        
        
    
data_spoken_lang=data.astype(float)    

data_spoken_lang_train=data_spoken_lang[:2500]
data_spoken_lang_test=data_spoken_lang[2500:]


inp_multi_hot_spoken_lang = keras.layers.Input(shape=(data_spoken_lang_train.shape[1],))

        


# In[ ]:





# In[ ]:





# # cast
# 

# In[721]:


import re

cast_list=[]
for i  in cast:
    i
    cast_list.append(re.findall(r"name': '(.*?)',", i))
    
    
cast_texts=[]
for i  in cast_list:
    try:
     x= i[0]   
     cast_texts.append(x)
    except:
        cast_texts.append("NA")
# Tokenizing the text of the raw data
maxlen = 2  #max length of a sentence
max_words = 2000  #dic size

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(cast_texts)
sequences = tokenizer.texts_to_sequences(cast_texts)
        
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)        
        
    
data_cast=data.astype(float)    

data_cast_train=data_cast[:2500]
data_cast_test=data_cast[2500:]

inp_multi_hot_cast = keras.layers.Input(shape=(data_cast_train.shape[1],))


# In[ ]:





# # popularity and runtime

# In[722]:


popularity.shape


# In[723]:


runtime.shape


# In[724]:


runtime


# In[725]:


'''
max_run = max(runtime)

print(max_run)
runtime /= max_run
runtime
'''


# In[726]:


runtime.shape


# In[727]:


'''
mean = popularity.mean(axis=0)
popularity -= mean
std = popularity.std(axis=0)
popularity /= std
popularity

'''

'''
for i in range(0,len(runtime)):
    if runtime[i]==0:
        #print('yes')
        runtime[i]=90
'''        


# In[ ]:





# In[728]:


#popl_runtime = np.concatenate((popularity,runtime),axis=1)

popl_runtime = popularity


# In[729]:


popl_runtime.shape


# In[730]:


popl_runtime_train=popl_runtime[:2500]
popl_runtime_test=popl_runtime[2500:]


# In[731]:


popl_runtime_train


# In[732]:


inp_num_data_popl_runtime = keras.layers.Input(shape=(popl_runtime_train.shape[1],))


# In[733]:


inp_num_data_popl_runtime.shape


# In[ ]:





# 

# In[ ]:





# # numeric data (budget)

# In[734]:



budget_train=budget[:2500]
budget_test=budget[2500:]


inp_num_data = keras.layers.Input(shape=(budget_train.shape[1],))


# In[735]:


budget_train.shape


# In[736]:


inp_num_data.shape


# # Concatenation of various inputs and define model

# Here we will be concatinating different inputs and define our model

# In[737]:


conc = keras.layers.Concatenate()([flatten,flatten_b_t_c,inp_multi_hot_generes,inp_multi_hot_org_lang, inp_multi_hot_org_title, inp_multi_hot_Keywords ,inp_multi_hot_spoken_lang,inp_multi_hot_cast, inp_num_data_popl_runtime, inp_num_data])

dense1 = keras.layers.Dense(64, activation=tf.nn.relu, )(conc)

dense2 = keras.layers.Dense(32, activation=tf.nn.relu, )(dense1)

dense3 = keras.layers.Dense(8, activation=tf.nn.relu, )(dense2)

# Creating output layer
out = keras.layers.Dense(1)(dense3)


model = keras.Model(inputs=[inp_multi_hot, inp_multi_hot_b_t_c,inp_multi_hot_generes,inp_multi_hot_org_lang, inp_multi_hot_org_title, inp_multi_hot_Keywords, inp_multi_hot_spoken_lang, inp_multi_hot_cast,inp_num_data_popl_runtime, inp_num_data], outputs=out)


# summarize layers
print(model.summary())
# plot graph
#plot_model(model, to_file='multiple_inputs.png')

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.mean_squared_logarithmic_error,
              metrics=['accuracy'])



#history= model.fit([data_overview_train, data_b_to_c_train,budget_train], revenue_train, epochs=100, validation_data=([data_overview_val, data_b_to_c_val,budget_val], revenue_val))
history= model.fit([data_overview_train, data_b_to_c_train, data_generes_train ,data_org_lang_train, data_org_title_train,data_Keywords_train ,data_spoken_lang_train, data_cast_train, popl_runtime_train, budget_train], revenue_train, epochs=100, validation_split = 0.1)


# In[ ]:





# In[738]:



model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model


# In[739]:


new_model = keras.models.load_model('my_model.h5')
new_model.summary()


# In[740]:


new_model.predict([data_overview_test, data_b_to_c_test, data_generes_test ,data_org_lang_test, data_org_title_test,data_Keywords_test, data_spoken_lang_test, data_cast_test ,popl_runtime_test, budget_test])


# In[741]:


new_model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.mean_squared_logarithmic_error,
              metrics=['accuracy'])


new_model.evaluate([data_overview_test, data_b_to_c_test, data_generes_test ,data_org_lang_test, data_org_title_test,data_Keywords_test, data_spoken_lang_test, data_cast_test,popl_runtime_test, budget_test],revenue_test)


# In[742]:


import matplotlib.pyplot as plt


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#fig1=plt.figure(2,figsize=(7,5))

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




# In[743]:


import matplotlib.pyplot as plt


acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

#fig1=plt.figure(2,figsize=(7,5))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()




# In[ ]:





# # Future scope

# - Some features can be removed or combined (PCA)
# - Pre trained embedding can be used (ex. glove word embeddings) and RNN can also be tried of feature like overview
# - some web scraping can be done for getting more data (information) related to existing features like cast, imdbid, etc.
# - Regularization can be used for reducing overfitting.

# In[ ]:





# # References::
# 
# - https://keras.io/
# - google
# - Deep learning with python - François Chollet

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




