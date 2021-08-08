from wordhoard import Homophones
import itertools
from nltk.util import ngrams
#from spellchecker import SpellChecker


class WordDetectionService:

    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.expanded_stop_words = self.expand_stop_words()
        self.stop_word_length = len(list(self.expanded_stop_words)[0])

    def tokenize(phrase):
        return phrase.lower().split()

    def process_homophones(homophone):
        # library returns a string instead of empty array for non-homophones
        if type(homophone) == str or homophone is None:
            return []

        # library returns a sentence instead of single word for homophones
        if type(homophone) == list:
            return [x.split()[-1] for x in homophone]

    def expand_stop_words(self):
        # break down into words
        tokenized_phrase = WordDetectionService.tokenize(self.stop_words)
        phrase = []

        for word in tokenized_phrase:
            homophone = Homophones(word)
            homophone_results = homophone.find_homophones()
            phrase.append(WordDetectionService.process_homophones(
                homophone_results)+[word])

        # return cartesian product of homophones
        return set([x for x in itertools.product(*phrase)])

    def check_text(self, text):
        text = WordDetectionService.tokenize(text)
        ngram_list = list(ngrams(text, self.stop_word_length))
        intersection = self.expanded_stop_words.intersection(ngram_list)
        return len(intersection) > 0


# Example test code
#stop_words = 'Hey Fourth Brain'
#sp = stopwords_parser(stop_words)
#print(sp.check_text('ejnjen nwknwkenj wjndjnwej jwnejnwjen jwnejnwjn wnejnwejnw hae forth brain snjdsjn anjanjsba'))
