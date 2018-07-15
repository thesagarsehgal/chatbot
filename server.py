
# coding: utf-8

# In[1]:


from keras.models import Model, load_model
import numpy as np
import tensorflow as tf
import math
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import time
from keras import Model
from keras.layers import Input, Dense, GRU, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[2]:


import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)


# In[3]:


lines = open('movie_lines128.txt').read().split('\n')
len(lines)



# In[4]:


conv_lines = open('movie_conversations128.txt').read().split('\n')
conv_lines[:10]


# In[5]:


id2line = {}
for line in lines:
    line_list = line.split(' +++$+++ ')
    if len(line_list) == 5:
        id2line[line_list[0]] = line_list[4]
    else :
        print(line_list)
        
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","")
    convs.append(_line.split(','))

print(convs[:10])


# In[6]:


mark_start = 'ssss '
mark_end = ' eeee'
input_x = []
output_y = []
for conv in convs:
    for i in range(len(conv) - 1):
        input_x.append(id2line[conv[i]])
        output_y.append(mark_start+id2line[conv[i+1]]+mark_end)


# In[7]:


for i in range(1,10):
    print(input_x[i])
    print(output_y[i])


# In[8]:


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "th-at is", text)
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


# In[9]:


input_x_clean = []
for sent in input_x:
    input_x_clean.append(clean_text(sent))

output_y_clean = []
for sent in output_y:
    output_y_clean.append(clean_text(sent))
    
for i in range(1,10):
    print(input_x_clean[i])
    print(output_y_clean[i])
    
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



# In[10]:


class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]
        
            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens = np.mean(self.num_tokens)                           + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """

        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens


# In[11]:


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "th-at is", text)
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


# In[12]:


num_words = 10000
tokenizer_src = TokenizerWrap(texts = final_input, padding='pre', reverse = True, num_words = num_words)


# In[13]:


tokenizer_dest = TokenizerWrap(texts = final_output, padding='post', reverse = False, num_words = num_words)
tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded
print(tokens_src.shape)
print(tokens_dest.shape)
token_start = tokenizer_dest.word_index[mark_start.strip()]
print(token_start)
token_end = tokenizer_dest.word_index[mark_end.strip()]
print(token_end)
print(tokens_dest[2])
print(tokens_src[2])
print (tokens_dest[5])
print(tokens_src[5])
encoder_input_data = tokens_src[:,:]
decoder_input_data = tokens_dest[:,:-1]
decoder_output_data = tokens_dest[:,1:]
print(decoder_input_data.shape)
print(decoder_output_data.shape)


# In[14]:


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
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# In[15]:


import keras.losses
keras.losses.sparse_cross_entropy = sparse_cross_entropy


# In[16]:


model = load_model('chbot3.h5')


# In[17]:


model.summary()


# In[18]:


print(model.input_names)
print(model.output_names)
print(model.layers)
print (model.layers[0])
print (model.layers[1])
print (model.inputs[1])


# In[19]:


encoder_input = model.input[0]
encoder_output = model.layers[6].output
model_encoder = Model(input=[encoder_input], outputs = [encoder_output])


# In[20]:


model_encoder.summary()


# In[21]:


state_size = 512
decoder_input = model.inputs[1]
decoder_initial_state = Input(shape=(state_size,),name = 'decoder_initial_state')
decoder_embedding = model.layers[5]
decoder_gru1 = model.layers[7]
decoder_gru2 = model.layers[8]
decoder_gru3 = model.layers[9]
decoder_dense = model.layers[10]


# In[22]:


net = decoder_input
net = decoder_embedding(net)
net = decoder_gru1(net, initial_state=decoder_initial_state)


# In[23]:


net = decoder_gru2(net, initial_state=decoder_initial_state)


# In[24]:


net = decoder_gru3(net, initial_state=decoder_initial_state)


# In[25]:


decoder_output = decoder_dense(net)


# In[26]:


model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs = [decoder_output])


# In[27]:


model_decoder.summary()


# In[40]:


def translate(input_text, true_output_text=None):
    """Translate a single text-string."""

    # Convert the input-text to integer-tokens.
    # Note the sequence of tokens has to be reversed.
    # Padding is probably not necessary.
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
    
    # Get the output of the encoder's GRU which will be
    # used as the initial state in the decoder's GRU.
    # This could also have been the encoder's final state
    # but that is really only necessary if the encoder
    # and decoder use the LSTM instead of GRU because
    # the LSTM has two internal states.
    initial_state = model_encoder.predict(input_tokens)

    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_dest.max_tokens

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data =         {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = model_decoder.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]
        
        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer_dest.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]
    
    # Print the input-text.
#     print("Input text:")
#     print(input_text)
#     print()

    # Print the translated output-text.
    out=output_text
#     print("Reply:",end="")
#     print(output_text)
#     print()

    # Optionally print the true translated text.
    if true_output_text is not None:
#         print("True output text:")
#         print(true_output_text)
        out=true_output_text
#         print()
    
    return out


print(translate(input_text="how are you."))
print(translate(input_text="what are you doing"))

# In[42]:


# while(True):
# #     print("YOU:",end="")
#     x=input()
#     print("Bot:",translate(x))

from flask import Flask, request
app = Flask(__name__)

FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'
VERIFY_TOKEN = 'mychatbotapptestpassword'# <paste your verify token here>
PAGE_ACCESS_TOKEN = 'EAAZAyzsSqBHABANREwP81em90L0xJ41OZAZAqtZCTqeabuUeSCZA7Ggh3BGNcmO0v7Ecm3VphAEc2jhZCj8fmWqG9ysSHYYZBXfDUz7TCwSpBpZB0zSgak8oW0b0439cU5EL03PXDc524QVCM6JGZBqDywIzEG6B7R0KnDcrg1SaSYwZDZD'# paste your page access token here>"


def get_bot_response(message):
    """This is just a dummy function, returning a variation of what
    the user said. Replace this function with one connected to chatbot."""
    print("1")
    print("------------------------")
    print(message)
    resp=str(translate(str(message)))
    if(resp.split()[-1]=="eeee"):
        resp=" ".join(resp.split()[:-1])
    print(resp)
    print("---------------------------")
    return resp


def verify_webhook(req):
    print("2")
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    print("3")
    response = get_bot_response(message)
    send_message(sender, response)


def is_user_message(message):
    """Check if the message is a message from the user"""
    print("4")
    return (message.get('message') and
            message['message'].get('text') and
            not message['message'].get("is_echo"))


@app.route("/webhook",methods=['GET','POST'])
def listen():
    """This is the main function flask uses to 
    listen at the `/webhook` endpoint"""
    print(5)
    if request.method == 'GET':
        return verify_webhook(request)

    if request.method == 'POST':
        payload = request.json
        event = payload['entry'][0]['messaging']
        for x in event:
            if is_user_message(x):
                text = x['message']['text']
                sender_id = x['sender']['id']
                respond(sender_id, text)

        return "ok"

import requests

def send_message(recipient_id, text):
    """Send a response to Facebook"""
    print(6)
    payload = {
        'message': {
            'text': text
        },
        'recipient': {
            'id': recipient_id
        },
        'notification_type': 'regular'
    }

    auth = {
        'access_token': PAGE_ACCESS_TOKEN
    }

    response = requests.post(
        FB_API_URL,
        params=auth,
        json=payload
    )

    return response.json()

# In[30]:
if __name__=="__main__":
    app.run(debug=False)




# # In[31]:


# translate(input_text = "Do you love me")


# # In[32]:


# translate(input_text = "what is your name")
# translate(input_text = "how are you")


# # In[33]:


# translate(input_text = "Do you like me")


# # In[34]:


# translate(input_text = "where do you live")


# # In[35]:


# translate(input_text = "lets go out today")


# # In[36]:


# translate(input_text = "can i visit you today")


# # In[37]:


# translate(input_text="what will you eat today")

