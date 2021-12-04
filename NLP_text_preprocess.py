# Standard libraries
import os
import re
import string
import logging
import csv
from pathlib import Path
from functools import wraps
from unicodedata import normalize
from typing import List, Optional, Union, Callable

# Third party libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


class PreProcessing:
  def __init__(self, to_lower = False,remove_url=False,remove_time=False,expand_contraction=False,remove_special_character=False, remove_punctuation=False,
               remove_whitespace=False,check_spelling=False,remove_stopword=False,lemmatize_word=False):
    
    self.to_lower = to_lower
    self.remove_url=remove_url
    self.remove_time=remove_time
    self.expand_contraction = expand_contraction
    self.remove_special_character=remove_special_character
    self.remove_punctuation=remove_punctuation
    self.remove_whitespace=remove_whitespace
    self.check_spelling=check_spelling
    self.remove_stopword=remove_stopword
    self.lemmatize_word=lemmatize_word

  
  def preprocess(self, input_text):

    if self.to_lower:
      input_text = self.to_lower_method(input_text)

    if self.remove_url:
      input_text = self.remove_url_method(input_text)

    if self.remove_time:
      input_text = self.remove_time_method(input_text)
      
     if self.expand_contraction:
      input_text = self.expand_contraction_method(input_text)     

    if self.remove_special_character:
      input_text = self.remove_special_character_method(input_text)
    
    if self.remove_punctuation:
      input_text = self.remove_punctuation_method(input_text)

    if self.remove_whitespace:
      input_text = self.remove_whitespace_method(input_text)
      
    if self.check_spelling:
      input_text = self.check_spelling_method(input_text)


    if self.remove_stopword:
      input_text = self.remove_stopword_method(input_text)
            
    if self.lemmatize_word:
      input_text = self.lemmatize_word_method(input_text)

    if isinstance(input_text, str):
        processed_text = input_text
    else:
        processed_text = ' '.join(input_text)
    return processed_text

  def to_lower_method(self,input_text:str)->str:
    """ Convert input text to lower case """
    return input_text.lower()

  def remove_url_method(self,input_text:str)-> str:
    """ Remove url in the input text """
    return re.sub('(www|http)\S+', '', input_text)

  def remove_time_method(self,input_text:str)-> str:
    """ Remove url in the input text """
    return re.sub('(1[0-2]|0?[1-9]):([0-5][0-9]) ?([AaPp].?[Mm])', '', input_text)
  
  def expand_contraction_method(self,input_text: str) -> str:
    """ Expand contractions in input text """
    return contractions.fix(input_text)

  def remove_punctuation_method(self,input_text:str)-> str:
    """
    Removes all punctuations from a string, as defined by string.punctuation or a custom list.
    For reference, Python's string.punctuation is equivalent to '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~'
    """
    punctuations = string.punctuation
    processed_text = input_text.translate(str.maketrans('', '', punctuations))
    return processed_text

  def remove_special_character_method(self,input_text:str)-> str:
    """ Removes special characters """
    special_characters = 'å¼«¥ª°©ð±§µæ¹¢³¿®ä£'
    processed_text = input_text.translate(str.maketrans('', '', special_characters))
    return processed_text

  def keep_alpha_numeric_method(self,input_text:str):
    """ Remove any character except alphanumeric characters """
    return ''.join(c for c in input_text if c.isalnum())

  def remove_whitespace_method(self,input_text:str, remove_duplicate_whitespace: bool = True)-> str:
    """ Removes leading, trailing, and (optionally) duplicated whitespace """
    if remove_duplicate_whitespace:
        return ' '.join(re.split('\s+', input_text.strip(), flags=re.UNICODE))
    return input_text.strip()

  def remove_stopword_method(self,input_text_or_list: Union[str, List[str]])-> List[str]:
    """ Remove stop words """

    stop_words = set(stopwords.words('english'))
    not_stop_words = ["not","nor", "after","before","above", "below","between"]
    stop_words = stop_words.difference(not_stop_words)
    if isinstance(stop_words, list):
        stop_words = set(stop_words)
    if isinstance(input_text_or_list, str):
        tokens = word_tokenize(input_text_or_list)
        processed_tokens = [token for token in tokens if token not in stop_words]
    else:
        processed_tokens = [token for token in input_text_or_list
                            if (token not in stop_words and token is not None)]
    return processed_tokens
     
  def lemmatize_word_method(self,input_text_or_list: Union[str, List[str]])-> List[str]:
    """ Lemmatize each token in a text by finding its base form """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    if isinstance(input_text_or_list, str):
        tokens = word_tokenize(input_text_or_list)
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    else:
        processed_tokens = [lemmatizer.lemmatize(token) for token in input_text_or_list if token is not None]
    return processed_tokens
     

  def check_spelling_method(self,input_text_or_list: Union[str, List[str]], lang='en') -> str:
    """ Check and correct spellings of the text list """
    if input_text_or_list is None:
        return ''
    spelling_checker = SpellChecker()
    # TODO: add acronyms into spell checker to ignore auto correction specified by _IGNORE_SPELLCHECK_WORD_FILE_PATH
    #spelling_checker.word_frequency.load_text_file(ignore_word_file_path)
    if isinstance(input_text_or_list, str):
        if not input_text_or_list.islower():
            input_text_or_list = input_text_or_list.lower()
        tokens = word_tokenize(input_text_or_list)
    else:
        tokens = [token.lower() for token in input_text_or_list if token is not None]
    misspelled = spelling_checker.unknown(tokens)
    for word in misspelled:
        tokens[tokens.index(word)] = spelling_checker.correction(word)
    return ' '.join(tokens).strip()
