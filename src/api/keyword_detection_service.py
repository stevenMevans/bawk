from wordhoard import Homophones
import itertools
from nltk.util import ngrams
from spellchecker import SpellChecker
import re
from Levenshtein import setratio


class KeywordDetectionService:
    """
    A class to find stop words in a text. Intended for audio snippets transcribed to words but any text will work.

    ...

    Attributes
    ----------
    stop_words : str
        Phrase used as stop word 
    expanded_stop_words : set(tuple)
        Represents possible combinations of stop words
    stop_word_length : int
        Number of words in stop word

    Methods
    -------
    tokenize(phrase)
        Tokenizes text into words.
    expand_stop_words(self)
        Expands all possible combinations of homophones
    process_homophones(homophone):
        Strips sentence returned by homophone library
    check_text(self, text):
        Attempts to locate stop words in a string


    Example Usage
    -------
    stop_words = 'Hay Foufth Brain' #homophone and spelling mistake
    sp = KeywordDetectionService(stop_words)
    phrase = '''the quick brown fox jumped over the lazy dog'''
    phrase2 = phrase + ' hey fourth brain'
    
    print(sp.check_text(phrase)) -> False
    print(sp.check_text(phrase2)) -> True
    """

    def __init__(self, stop_words):
        """
        Parameters
        ----------
        stop_words : str
            Phrase to be used as stop word
        """

        self.stop_words = stop_words
        self.expanded_stop_words = self.expand_stop_words()
        self.expanded_stop_words = KeywordDetectionService.spellcheck(KeywordDetectionService.tokenize(self.stop_words), self.expanded_stop_words)
        self.stop_word_length = len(list(self.expanded_stop_words)[0])

    def tokenize(phrase):
        """ Tokenizes text into words.
        
        Parameters
        ----------
        phrase : str
            The phrase to break up into words
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)

        Returns
        -------
        list
            a list of strings. Each string in list is approxiamately a word.
        """

        #Built-in split functions handles contractions better than ngrams tokenizer
        #remove common punctuation except for apostrophe and hypen 
        return re.sub('[\.:,\"\?\+\*\!\$\@%^;()]+', ' ', phrase.lower()).split()

    def process_homophones(homophone):
        """ Strips sentence returned by homophone library; returning just the homophone.
        
        Parameters
        ----------
        homophone : unknown
            Potential homophone

        Returns
        -------
        list
            homophones if any that were found
        """

        # library returns a string instead of empty array for non-homophones
        if type(homophone) == str or homophone is None:
            return []

        # library returns a sentence instead of single word for homophones
        if type(homophone) == list:
            return [x.split()[-1] for x in homophone]

    def expand_stop_words(self):
        """ Expands all possible combinations of homophones so that set intersection can find the correct match. Ex. Stop word of 'hi there' would be expanded to ('hi','there'), ('hi','their'),('hi','they're),('high','there'), ('high','their'),('high','they're)
        
        Returns
        -------
        list
            a list of strings. Each string in list is approxiamately a word.
        """

        # break down into words
        tokenized_phrase = KeywordDetectionService.tokenize(self.stop_words)
        phrase = []

        for word in tokenized_phrase:
            homophone = Homophones(word)
            homophone_results = homophone.find_homophones()
            phrase.append(KeywordDetectionService.process_homophones(
                homophone_results) + [word])

        # return cartesian product of homophones
        return set([x for x in itertools.product(*phrase)])

    def check_text(self, text, spellcheck = False, levenshtein_threshold = .9):
        """ Attempts to locate stop words in a string
        Parameters
        ----------
        text : str
            Sample text that may contain stop words
        spellcheck: bool
            Flag that runs spellchecker if True, Levenshtein distance if false
        levenshtein_threshold: float
            Minimum string set similarity necessary to be considered a match

        Returns
        -------
        bool
            True if stop word was detected in the phrase. False otherwise.
        """

        text = KeywordDetectionService.tokenize(text)
        ngram_set = set(ngrams(text, self.stop_word_length))

        #use spellchecker and set intersection
        if spellcheck:
            #Fixing spelling is slow. Iterates through every word then every tuple for the misspelled words.
            ngram_set = KeywordDetectionService.spellcheck(text, ngram_set)
       
            intersection = self.expanded_stop_words.intersection(ngram_set)
            return len(intersection) > 0

        #use levenshtein distance
        else:
            for word1 in self.expanded_stop_words:
                for word2 in ngram_set:
                    if setratio(word1,word2) > levenshtein_threshold:
                        return True

            return False

    def spellcheck(text,ngrams):
        """ Checks spelling of words in tuples and adds corrected tuples to set.
        Parameters
        ----------
        text : str
            Words that need spell checking
        ngrams : set(tuple)
            text variable split into tuples

        Returns
        -------
        set(bool)
            Tuples with corrected spellings appended
        """

        #find all misspelled words
        spell = SpellChecker()
        misspelled = spell.unknown(text)

        #leave original text in place but add alternative set for misspelled words
        for word in misspelled:
            tuples = []
            if(spell.correction(word) != word):
                for tup in ngrams:
                    if word in tup:
                        tuples.append(tuple([spell.correction(word) if x == word else x for x in tup]))
                ngrams.update(tuples)

        return ngrams