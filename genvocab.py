import word2phrase
from textblob import TextBlob
from mycounter import Counter
import pickle

def get_book():
	salida={}
	for line in open('./train/bigquizlemma.txt'):
		salida[line]=None
	for line in open('./train/bigquizlemma2.txt'):
		salida[line]=None
	for line in open('./train/bigquizlemma3.txt'):
		salida[line]=None
	for line in open('./train/CK12lemma.txt'):
		salida[line]=None
	return [m.split(' ') for m in salida.keys()]


def main():
	book_sentences = get_book()
	phrased1 = word2phrase.train_model(book_sentences, min_count=3)
	phrased2 = word2phrase.train_model(phrased1, min_count=3)
	two_word_counter = Counter()
	three_word_counter = Counter()
	for sentence in phrased2:
		for word in sentence:
			if word.count('_') == 1:
				two_word_counter[word] += 1
			if word.count('_') == 2:
				three_word_counter[word] += 1
	pickle.dump([two_word_counter,three_word_counter],open('quizletvocab3.pick','wb+'))
	print '=' * 60
	print 'Top 20 Two Word Phrases'
	for phrase, count in two_word_counter.most_common(20):
		print '%56s %6d' % (phrase, count)

	print
	print '=' * 60
	print 'Top 10 Three Word Phrases'
	for phrase, count in three_word_counter.most_common(10):
		print '%56s %6d' % (phrase, count)


if __name__ == '__main__':
	main()
