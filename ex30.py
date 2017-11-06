# Sentence tokenization
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import brown

text = "Are you curious about tokenization? Let's see how it " \
       "works! We need to analyze a couple of sentences with punctuations " \
       "to see it in action."

sent_tokenize_list = sent_tokenize(text)
print "\nSentence tokenizer:"
print sent_tokenize_list

print "\nWord tokenizer:"
print word_tokenize(text)

treebank_word_tokenizer = TreebankWordTokenizer()
print "\nTreebank word tokenizer:"
print treebank_word_tokenizer.tokenize(text)

word_punct_tokenizer = WordPunctTokenizer()
print "\nWord punct tokenizer:"
print word_punct_tokenizer.tokenize(text)

words = ['table', 'probably', 'wolves', 'playing', 'is',
         'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']
# Compare different stemmers
stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL']
stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmer_snowball = SnowballStemmer('english')
formatted_row = '{:>16}' * (len(stemmers) + 1)
print '\n', formatted_row.format('WORD', *stemmers), '\n'

for word in words:
       stemmed_words = [stemmer_porter.stem(word),stemmer_lancaster.stem(word),stemmer_snowball.stem(word)]
       print formatted_row.format(word, *stemmed_words)

# Compare different lemmatizers
lemmatizers = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
lemmatizer_wordnet = WordNetLemmatizer()
formatted_row = '{:>24}' * (len(lemmatizers) + 1)
print '\n', formatted_row.format('WORD', *lemmatizers), '\n'
for word in words:
       lemmatized_words = [lemmatizer_wordnet.lemmatize(word,pos='n'),lemmatizer_wordnet.lemmatize(word, pos='v')]
       print formatted_row.format(word, *lemmatized_words)

# Split a text into chunks
def splitter(data, num_words):
       words = data.split(' ')
       output = []
       cur_count = 0
       cur_words = []
       for word in words:
              cur_words.append(word)
              cur_count += 1
              if cur_count == num_words:
                     output.append(' '.join(cur_words))
                     cur_words = []
                     cur_count = 0
       output.append(' '.join(cur_words))
       return output
if __name__=='__main__':
       # Read the data from the Brown corpus
       data = ' '.join(brown.words()[:10000])
       # Number of words in each chunk
       num_words = 1700
       chunks = []
       counter = 0
       text_chunks = splitter(data, num_words)
       print "Number of text chunks =", len(text_chunks)

       # Number of words in each chunk
       num_words = 2000
       chunks = []
       counter = 0
       text_chunks1 = splitter(data, num_words)
       for text in text_chunks1:
              chunk = {'index': counter, 'text': text}
              chunks.append(chunk)
              counter += 1
       vectorizer = CountVectorizer(min_df=5, max_df=.95)
       doc_term_matrix = vectorizer.fit_transform([chunk['text'] for chunk in chunks])
       vocab = np.array(vectorizer.get_feature_names())
       print "\nVocabulary:"
       print vocab
       print "\nDocument term matrix:"
       chunk_names = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3','Chunk-4']
       formatted_row = '{:>12}' * (len(chunk_names) + 1)
       print '\n', formatted_row.format('Word', *chunk_names), '\n'
       for word, item in zip(vocab, doc_term_matrix.T):
              # 'item' is a 'csr_matrix' data structure
              output = [str(x) for x in item.data]
              print formatted_row.format(word, *output)