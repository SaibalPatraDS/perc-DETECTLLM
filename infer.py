"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

from model import GPT2PPLV2 as GPT2PPL

# initialize the model
model = GPT2PPL()

# sentence = "We load the pre-trained multilingual BERT tokenizer and model. We tokenize and encode the sample multilingual dataset using the tokenizer."

sentence = "Let me tell you something. Develop an advanced Intent Detection system for hotel booking interactions with a focus on multilingual support.\
        The system should accurately classify user messages into various intents across languages, including standard actions like booking rooms,\
         canceling bookings, checking availability, modifying bookings, as well as nuanced intents like requesting special offers, inquiring about local attractions, and resolving complaints. "

model(sentence, 100, "v1.1")
