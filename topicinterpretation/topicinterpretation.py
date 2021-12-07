import numpy as np
import math
import sys


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path, topic_path=''):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.topic_path = topic_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        
        file = open(self.documents_path, "r")
        while True: 
            line = file.readline()
            if not line:
                break
            words = line.split()
            self.documents.append(words)
        file.close()
        self.number_of_documents = len(self.documents)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        
        self.vocabulary = list(set(word for document in self.documents for word in document))
        self.vocabulary_size = len(self.vocabulary)

        if self.topic_path != '':
            self.load_topics()

        print(self.topic_word_prob)

    def load_topics(self):
        total_topic = 0
        file = open(self.topic_path, "r")
        while True: 
            line = file.readline()
            words = line.split()
            if len(words) == 0:
                total_topic += 1
            if not line:
                break
        file.close()

        self.topic_word_prob = np.zeros((total_topic, self.vocabulary_size))
        self.topic_word_prob.fill(0.0000001)
        file = open(self.topic_path, "r")
        topic_counter = 0
        while True: 
            line = file.readline()
            words = line.split()
            if len(words) == 0:
                topic_counter += 1
            else:
                if words[0] in self.vocabulary:
                    index = self.vocabulary.index(words[0])
                    self.topic_word_prob[topic_counter][index] = float(words[1]) + self.topic_word_prob[topic_counter][index]
            if not line:
                break
        file.close()

        self.topic_word_prob = normalize(self.topic_word_prob)
        print(self.topic_word_prob)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        
        self.term_doc_matrix = []
        for i in range(self.number_of_documents):
            term_doc_list = []
            for j in range(self.vocabulary_size):
                term_doc_list.append(self.documents[i].count(self.vocabulary[j]))
            self.term_doc_matrix.append(term_doc_list)
        self.term_doc_matrix = np.asarray(self.term_doc_matrix)

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################

        self.document_topic_prob = np.random.random((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        if self.topic_path == '':
            self.topic_word_prob = np.random.random((number_of_topics, len(self.vocabulary)))
            self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        if self.topic_path == '':
            self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
            self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        # ############################

        for doc_index, pi_arr in enumerate(self.document_topic_prob):
            for word_index in range(self.vocabulary_size):
                word_topics_arr = [row[word_index] for row in self.topic_word_prob]
                normalizer = np.dot(pi_arr, word_topics_arr)
                
                for topic_index, pi_value in enumerate(pi_arr):           
                    self.topic_prob[doc_index][topic_index][word_index] = pi_value * self.topic_word_prob[topic_index][word_index] / normalizer

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")

        # update P(w | z)
        
        # ############################
        # your code here
        # ############################

        for doc_index in range(self.number_of_documents):
            z_matrix = self.topic_prob[doc_index]
            term_count_arr = self.term_doc_matrix[doc_index]
            
            denominator = 0
            for topic_index in range(number_of_topics):
                denominator += np.dot(term_count_arr, z_matrix[topic_index])
                
            for topic_index in range(number_of_topics):
                numerator = np.dot(term_count_arr, z_matrix[topic_index])
                self.document_topic_prob[doc_index][topic_index] = numerator / denominator

        # update P(z | d)

        # ############################
        # your code here
        # ############################
        
        if self.topic_path == '':
            for topic_index in range(number_of_topics):
                denominator = 0
                for word_index in range(self.vocabulary_size):
                    for doc_index in range(self.number_of_documents):
                        denominator += self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
                    
                for word_index in range(self.vocabulary_size):    
                    numerator = 0
                    for doc_index in range(self.number_of_documents):
                        numerator += self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
                
                    self.topic_word_prob[topic_index][word_index] = numerator / denominator 

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        
        total = 0
        for doc_index in range(self.number_of_documents):
            w_sum = 0
            for word_index in range(self.vocabulary_size):
                # P(w | z) for all the z
                word_topics_arr = [row[word_index] for row in self.topic_word_prob]
                temp = np.dot(self.document_topic_prob[doc_index], word_topics_arr)
                w_sum += self.term_doc_matrix[doc_index][word_index] * math.log(temp)
            
            total += w_sum
        
        self.likelihoods.append(total)
        return total

    def em(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################

            self.expectation_step()
            self.maximization_step(number_of_topics)
            next_likelihood = self.calculate_likelihood(number_of_topics)
            print next_likelihood
            if (current_likelihood != 0.0) and (next_likelihood - current_likelihood < epsilon):
                break
            current_likelihood = next_likelihood

    def print_topics(self):
        print("A total of " + str(len(self.topic_word_prob)) + " topics")
        print
        for i in range(len(self.topic_word_prob)):
            print("topic " + str(i+1))
            for j in range(len(self.topic_word_prob[i])):
                print(self.vocabulary[j] + ": "+ str(self.topic_word_prob[i][j]))
            print

    def print_topic_distribution(self):
        for i in range(self.number_of_documents):
            print("document " + str(i+1))
            for j in range(len(self.document_topic_prob[i])):
                print("topic " + str(j+1) + ": "+ str(self.document_topic_prob[i][j]))
            print

def main():
    documents_path = sys.argv[1]
    if (len(sys.argv) == 3):
        topic_path = sys.argv[2]
        corpus = Corpus(documents_path, topic_path)
    else:
        corpus = Corpus(documents_path)
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    max_iterations = 10
    number_of_topics = 2
    epsilon = 0.001
    if (len(sys.argv) == 3):
        corpus.em(len(corpus.topic_word_prob), max_iterations, epsilon)
    else:
        corpus.em(number_of_topics, max_iterations, epsilon)

if __name__ == '__main__':
    main()
