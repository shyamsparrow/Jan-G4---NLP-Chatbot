{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Text_Preprocessing.py",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/shyamsparrow/Jan-G4---NLP-Chatbot/blob/main/Text_Preprocessing_py.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6DqqzSMEsAnw",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1b79f0e4-e905-4542-cf92-4807edfb1e57"
      },
      "source": [
        "# Standard libraries\n",
        "!pip install pyspellchecker\n",
        "import os\n",
        "import re\n",
        "import string\n",
        "import logging\n",
        "import csv\n",
        "from pathlib import Path\n",
        "from functools import wraps\n",
        "from unicodedata import normalize\n",
        "from typing import List, Optional, Union, Callable\n",
        "\n",
        "# Third party libraries\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize, PunktSentenceTokenizer\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from spellchecker import SpellChecker\n",
        "\n",
        "nltk.download('stopwords', quiet=True)\n",
        "nltk.download('wordnet', quiet=True)\n",
        "nltk.download('punkt', quiet=True)\n",
        "\n",
        "class PreProcessing:\n",
        "  def __init__(self, to_lower = False,remove_url=False,remove_special_character=False, remove_punctuation=False,\n",
        "               remove_whitespace=False,check_spelling=False,remove_stopword=False,lemmatize_word=False):\n",
        "    \n",
        "    self.to_lower = to_lower\n",
        "    self.remove_url=remove_url\n",
        "    self.remove_special_character=remove_special_character\n",
        "    self.remove_punctuation=remove_punctuation\n",
        "    self.remove_whitespace=remove_whitespace\n",
        "    self.check_spelling=check_spelling\n",
        "    self.remove_stopword=remove_stopword\n",
        "    self.lemmatize_word=lemmatize_word\n",
        "\n",
        "  \n",
        "  def preprocess(self, input_text):\n",
        "\n",
        "    if self.to_lower:\n",
        "      input_text = self.to_lower_method(input_text)\n",
        "\n",
        "    if self.remove_url:\n",
        "      input_text = self.remove_url_method(input_text)\n",
        "    \n",
        "    if self.remove_special_character:\n",
        "      input_text = self.remove_special_character_method(input_text)\n",
        "    \n",
        "    if self.remove_punctuation:\n",
        "      input_text = self.remove_punctuation_method(input_text)\n",
        "\n",
        "    if self.remove_whitespace:\n",
        "      input_text = self.remove_whitespace_method(input_text)\n",
        "      \n",
        "    if self.check_spelling:\n",
        "      input_text = self.check_spelling_method(input_text)\n",
        "      \n",
        "    if self.remove_stopword:\n",
        "      input_text = self.remove_stopword_method(input_text)\n",
        "            \n",
        "    if self.lemmatize_word:\n",
        "      input_text = self.lemmatize_word_method(input_text)\n",
        "\n",
        "    if isinstance(input_text, str):\n",
        "        processed_text = input_text\n",
        "    else:\n",
        "        processed_text = ' '.join(input_text)\n",
        "    return processed_text\n",
        "\n",
        "  def to_lower_method(self,input_text:str)->str:\n",
        "    \"\"\" Convert input text to lower case \"\"\"\n",
        "    return input_text.lower()\n",
        "\n",
        "  def remove_url_method(self,input_text:str)-> str:\n",
        "    \"\"\" Remove url in the input text \"\"\"\n",
        "    return re.sub('(www|http)\\S+', '', input_text)\n",
        "\n",
        "  \n",
        "  def remove_punctuation_method(self,input_text:str)-> str:\n",
        "    \"\"\"\n",
        "    Removes all punctuations from a string, as defined by string.punctuation or a custom list.\n",
        "    For reference, Python's string.punctuation is equivalent to '!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_{|}~'\n",
        "    \"\"\"\n",
        "    punctuations = string.punctuation\n",
        "    processed_text = input_text.translate(str.maketrans('', '', punctuations))\n",
        "    return processed_text\n",
        "\n",
        "  def remove_special_character_method(self,input_text:str)-> str:\n",
        "    \"\"\" Removes special characters \"\"\"\n",
        "    special_characters = 'å¼«¥ª°©ð±§µæ¹¢³¿®ä£'\n",
        "    processed_text = input_text.translate(str.maketrans('', '', special_characters))\n",
        "    return processed_text\n",
        "\n",
        "  def keep_alpha_numeric_method(self,input_text:str):\n",
        "    \"\"\" Remove any character except alphanumeric characters \"\"\"\n",
        "    return ''.join(c for c in input_text if c.isalnum())\n",
        "\n",
        "  def remove_whitespace_method(self,input_text:str, remove_duplicate_whitespace: bool = True)-> str:\n",
        "    \"\"\" Removes leading, trailing, and (optionally) duplicated whitespace \"\"\"\n",
        "    if remove_duplicate_whitespace:\n",
        "        return ' '.join(re.split('\\s+', input_text.strip(), flags=re.UNICODE))\n",
        "    return input_text.strip()\n",
        "\n",
        "  def remove_stopword_method(self,input_text_or_list: Union[str, List[str]])-> List[str]:\n",
        "    \"\"\" Remove stop words \"\"\"\n",
        "\n",
        "    stop_words = set(stopwords.words('english'))\n",
        "    if isinstance(stop_words, list):\n",
        "        stop_words = set(stop_words)\n",
        "    if isinstance(input_text_or_list, str):\n",
        "        tokens = word_tokenize(input_text_or_list)\n",
        "        processed_tokens = [token for token in tokens if token not in stop_words]\n",
        "    else:\n",
        "        processed_tokens = [token for token in input_text_or_list\n",
        "                            if (token not in stop_words and token is not None)]\n",
        "    return processed_tokens\n",
        "     \n",
        "  def lemmatize_word_method(self,input_text_or_list: Union[str, List[str]])-> List[str]:\n",
        "    \"\"\" Lemmatize each token in a text by finding its base form \"\"\"\n",
        "    lemmatizer = WordNetLemmatizer()\n",
        "    stop_words = set(stopwords.words('english'))\n",
        "\n",
        "    if isinstance(input_text_or_list, str):\n",
        "        tokens = word_tokenize(input_text_or_list)\n",
        "        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]\n",
        "    else:\n",
        "        processed_tokens = [lemmatizer.lemmatize(token) for token in input_text_or_list if token is not None]\n",
        "    return processed_tokens\n",
        "     \n",
        "\n",
        "  def check_spelling_method(self,input_text_or_list: Union[str, List[str]], lang='en') -> str:\n",
        "    \"\"\" Check and correct spellings of the text list \"\"\"\n",
        "    if input_text_or_list is None:\n",
        "        return ''\n",
        "    spelling_checker = SpellChecker()\n",
        "    # TODO: add acronyms into spell checker to ignore auto correction specified by _IGNORE_SPELLCHECK_WORD_FILE_PATH\n",
        "    #spelling_checker.word_frequency.load_text_file(ignore_word_file_path)\n",
        "    if isinstance(input_text_or_list, str):\n",
        "        if not input_text_or_list.islower():\n",
        "            input_text_or_list = input_text_or_list.lower()\n",
        "        tokens = word_tokenize(input_text_or_list)\n",
        "    else:\n",
        "        tokens = [token.lower() for token in input_text_or_list if token is not None]\n",
        "    misspelled = spelling_checker.unknown(tokens)\n",
        "    for word in misspelled:\n",
        "        tokens[tokens.index(word)] = spelling_checker.correction(word)\n",
        "    return ' '.join(tokens).strip()"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pyspellchecker in /usr/local/lib/python3.7/dist-packages (0.6.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P0hlcsqnsn7W",
        "outputId": "85b4771b-9f2c-4d74-bc9b-817b1bd53269"
      },
      "source": [
        "%%writefile config.py\n",
        "to_lower = True\n",
        "remove_url=True\n",
        "remove_special_character=True\n",
        "remove_punctuation=True\n",
        "remove_whitespace=True\n",
        "check_spelling=True\n",
        "remove_stopword=True\n",
        "lemmatize_word=True"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting config.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "FKsNOEvatCcb",
        "outputId": "838b3993-b585-4042-bc98-755cbcccdf91"
      },
      "source": [
        "pp = PreProcessing(to_lower = config.to_lower, remove_url=config.remove_url, remove_special_character=config.remove_special_character, \n",
        "                   remove_punctuation=config.remove_punctuation, remove_whitespace=config.remove_whitespace,\n",
        "                   check_spelling = config.check_spelling, remove_stopword=config.remove_stopword, \n",
        "                   lemmatize_word=config.lemmatize_word)\n",
        "\n",
        "text_to_process = 'Helllo,    I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'\n",
        "\n",
        "pp.preprocess(text_to_process)"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'hello john doe email johndoeemailcom visit website'"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lYt7b0vrcXF1"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
