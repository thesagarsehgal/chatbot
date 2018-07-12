
# coding: utf-8

# In[144]:


import pandas as pd
import numpy as np
import re
import time
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import LSTM
tf.__version__


# In[145]:


lines = open('cornell/out.txt').read().split('\n')
len(lines)


# In[146]:


lines[:20]
#lineid, #characterID , #movieID , #chracter name , #text of the utterance


# In[147]:


conv_lines = open('cornell/movie_conversations.txt').read().split('\n')
conv_lines[:10]
#characterid 1, characterid 2 ,movie id, lineids in chronological order


# In[148]:


id2line = {}
for line in lines:
    line_list = line.split(' +++$+++ ')
    if len(line_list) == 5:
        id2line[line_list[0]] = line_list[4]
    else :
        print (line_list)


# In[149]:


print (id2line['L1045'])


# In[150]:


convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","")
    convs.append(_line.split(','))

print (convs[:10])


# In[151]:


input_x = []
output_y = []
for conv in convs:
    for i in range(len(conv) - 1):
        input_x.append(id2line[conv[i]])
        output_y.append(id2line[conv[i+1]])


# In[152]:


for i in range(1,10):
    print(input_x[i])
    print(output_y[i])


# In[153]:


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


# In[154]:


input_x_clean = []
for sent in input_x:
    input_x_clean.append(clean_text(sent))

output_y_clean = []
for sent in output_y:
    output_y_clean.append(clean_text(sent))
    
for i in range(1,10):
    print(input_x_clean[i])
    print(output_y_clean[i])


# In[155]:


print(len(input_x_clean))
print(len(output_y_clean))


# In[156]:


min_length = 2
max_length = 20
final_input = []
final_output = []
for i in range(0,len(input_x_clean)):
    temp_input_list = input_x_clean[i].split(' ')
    temp_output_list = output_y_clean[i].split(' ')
    if len(temp_input_list) > max_length or len(temp_input_list) < min_length  or len(temp_output_list) > max_length or len(temp_output_list) < min_length:
        d  = 1 + 1
    else:
        final_input.append(input_x_clean[i])
        final_output.append(output_y_clean[i])
    


# In[157]:


print (len(final_input))
print (len(final_output))


# In[158]:


print (136053.0/221616.0)


# In[159]:


vocab = {}
for sent in final_input:
    for word in sent.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
for sent in final_output:
    for word in sent.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1


# In[160]:


print (len(vocab))


# In[161]:


word_num = 0
threshold = 10
codes  = ['<EOS>', '<UNK>', '<GO>']
vocab_to_int = {}
vocab_to_int['<PAD>'] = 0
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)
for word, count in vocab.items():
    if count >= threshold:
        vocab_to_int[word] = word_num+4
        word_num += 1
print(len(vocab_to_int))
print(vocab_to_int['<GO>'])


# In[162]:


print(len(vocab_to_int))
int_to_vocab = {v_i : v for v, v_i in vocab_to_int.items()}
print(len(int_to_vocab))
print(int_to_vocab[4])


# In[163]:


for i in range(len(final_output)):
    final_output[i] += ' <EOS>'
for i in range(len(final_output)):
    final_output[i] = '<GO> ' + final_output[i]
print (final_output[0])


# In[164]:


input_int = []
for sent in final_input:
    temp_int = []
    for word in sent.split( ' ' ):
        if word in vocab_to_int:
            temp_int.append(vocab_to_int[word])
        else:
            temp_int.append(vocab_to_int['<UNK>'])
    input_int.append(temp_int)
output_int = []
for sent in final_output:
    temp_int = []
    for word in sent.split( ' ' ):
        if word in vocab_to_int:
            temp_int.append(vocab_to_int[word])
        else:
            temp_int.append(vocab_to_int['<UNK>'])
    output_int.append(temp_int)
    


# In[165]:


print (len(input_int))
print (len(output_int))
print (input_int[1])
print (output_int[1])


# In[166]:


max_length = 0
for i in range (len(input_int)):
    if len(input_int[i]) > max_length:
        max_length = len(input_int[i])
print (max_length)


# In[167]:


import math
min_length = float('inf')
for i in range (len(input_int)):
    if len(input_int[i]) < min_length:
        min_length = len(input_int[i])
        print (input_int[i])
print (min_length)


# In[168]:


sorted_input = []
sorted_output = []
for length in range(min_length, max_length+1):
    for i in enumerate(input_int):
        if len(i[1]) == length:
            sorted_input.append(input_int[i[0]])
            sorted_output.append(output_int[i[0]])
print (len(sorted_input))
print (len(sorted_output))


# In[169]:


for i in range(4):
    print (sorted_output[i])
    
## PAD is 0 , EOS is 1 , UNK is 2 and GO is 3


# In[170]:


np_encoder_input = np.zeros([len(sorted_input), len(max(sorted_input, key = lambda x : len(x)))], dtype = np.int32)
for i,j in enumerate(sorted_input):
    np_encoder_input[i][0:len(j)] = j
np_decoder_input = np.zeros([len(sorted_output), len(max(sorted_output, key = lambda x : len(x)))], dtype = np.int32)
for i,j in enumerate(sorted_output):
    np_decoder_input[i][0:len(j)] = j
print (np_encoder_input.shape)
print (np_decoder_input.shape)


# In[171]:



print(np_encoder_input[0])


# In[172]:


print (np_encoder_input.shape)
np_encoder_input = np.fliplr(np_encoder_input)


# In[173]:


print (np_encoder_input[0])


# In[174]:


print (np_decoder_input[0])


# In[175]:


print(np_encoder_input.shape)


# In[176]:


print(np_decoder_input.shape)


# In[177]:


encoder_input = Input(shape=(None, ), name='encoder_input')
embedding_size = 128
encoder_embedding = Embedding(input_dim=7878,
                              output_dim=embedding_size,
                              name='encoder_embedding')


# In[178]:


state_size = 1024

encoder_lstm1 = LSTM(1024, return_sequences = True, name = 'encoder_lstm1')
encoder_lstm2 = LSTM(1024, return_sequences = True, name = 'encoder_lstm2')
encoder_lstm3 = LSTM(1024, return_sequences = False, name = 'encoder_lstm3', return_state = True)

def connect_encoder():
    net = encoder_input
    net = encoder_embedding(net)
    net = encoder_lstm1(net)
    net = encoder_lstm2(net)
    net = encoder_lstm3(net)

    # This is the output of the encoder.
    encoder_output, state_h, state_c = net
    
    return [state_h, state_c]


# In[179]:


encoder_output = connect_encoder()


# In[180]:


encoder_output[0].shape


# In[181]:


decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_state_input_h = Input(shape=(state_size,))
decoder_state_input_c = Input(shape=(state_size,))
decoder_initial_state = [decoder_state_input_h, decoder_state_input_c]
decoder_embedding = Embedding(input_dim=7878,output_dim=embedding_size,name='decoder_embedding')
decoder_lstm1 = LSTM(1024, return_sequences = True, name = 'decoder_lstm1')
decoder_lstm2 = LSTM(1024, return_sequences = True, name = 'decoder_lstm2')
decoder_lstm3 = LSTM(1024, return_sequences = True, name = 'decoder_lstm3')
decoder_dense = Dense(7878,activation='linear',name='decoder_output')


# In[182]:


def connect_decoder(initial_state):
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_lstm1(net, initial_state=initial_state)
    net = decoder_lstm2(net, initial_state=initial_state)
    net = decoder_lstm3(net, initial_state=initial_state)
    
    decoder_output = decoder_dense(net)
    return decoder_output
    
    


# In[183]:


decoder_output = connect_decoder(initial_state = encoder_output)


# In[184]:


decoder_output


# In[185]:


model_train = Model(inputs = [encoder_input, decoder_input], outputs = [decoder_output])


# In[186]:


model_train.summary()


# In[187]:


model_encoder= Model(inputs = [encoder_input], outputs = encoder_output)
model_encoder.summary()


# In[188]:


decoder_output = connect_decoder(initial_state=decoder_initial_state)
model_decoder = Model(inputs=[decoder_input]+ decoder_initial_state,
                      outputs=[decoder_output])


# In[189]:


model_decoder.summary()


# In[190]:


def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.
    
    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# In[191]:


optimizer = RMSprop(lr=1e-3)


# In[192]:


decoder_target = tf.placeholder(dtype='int32', shape=(None, None))


# In[193]:


model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])


# In[194]:


path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


# In[195]:


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)


# In[196]:


callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard]


# In[197]:


try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# In[198]:


decoder_input_data = np_decoder_input[:, :-1]
#print(decoder_input_data.shape)
decoder_output_data = np_decoder_input[:, 1:]
#print(decoder_output_data.shape)
x_data = {
    'encoder_input': np_encoder_input,
    'decoder_input': decoder_input_data
}
y_data = {
    'decoder_output': decoder_output_data
}
validation_split = 10000.0 / len(np_encoder_input)
validation_split


# In[ ]:


model_train.fit(x=x_data,
                y=y_data,
                batch_size=640,
                epochs=10,
                validation_split=validation_split,
                )


# In[ ]:


def speak(input_text):
    text_to_int = []
    for word in input_text.split(' '):
        if word in vocab_to_int:
            text_to_int.append(vocab_to_int[word])
        else:
            text_to_int.append(vocab_to_int['<UNK>'])
            

    if len(text_to_int) > 20:
        text_to_int = text_to_int[:20]
    while(len(text_to_int)<20):
        text_to_int.append(0)
    final_input_array = np.array(text_to_int, dtype=np.int32)[::-1]
    initial_state = model_encoder.predict(final_input_array)
    max_length_of_output = 22
    shape = [1, max_length_of_output]
    decoder_input_data = np.zeros(shape=shape, dtype = np.int32)
    token_int = 3  # start of sentence
    output_text = ''
    count_tokens = 0
    token_end = 1  #end of sentence
    while token_int != token_end and count_tokens < max_length_of_output :
        decoder_input_data[0,count_tokens] = token_int
        x_data =         {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }
        decoder_output = model_decoder.predict([decoder_input_data] + initial_state)
        print (decoder_output.shape)
        token_onehot = decoder_output[0, count_tokens, :]
        max_index = 0
        max_value = token_onehot[0]
        print (token_onehot)
        for i in range(len(token_onehot)):
            if token_onehot[i] > max_value:
                max_value = token_onehot[i]
                max_index = i
                
        print (max_value)
        #print token_onehot[1]
        token_int = np.argmax(token_onehot)
        #print 
        #print token_int
        sampled_word = int_to_vocab[token_int]
        output_text += " " + sampled_word
        count_tokens += 1
        
        
    print (input_text)
    print(output_text)
    print ()

    
    
    
    


# In[295]:


speak("how are you")


# In[296]:


speak("what are you doing")


# In[293]:


speak("can i see you")

