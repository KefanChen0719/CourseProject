import numpy as np
import math


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

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
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

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

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

        document_word_normalizer = np.matmul(self.document_topic_prob, self.topic_word_prob)
        self.topic_prob = []
        for i in range(self.number_of_documents):
            document_prob_list = []
            for j in range(self.vocabulary_size):
                document_word_prob_list = []
                for k in range(len(self.topic_word_prob)):
                    document_word_prob_list.append(self.document_topic_prob[i][k] * self.topic_word_prob[k][j] / document_word_normalizer[i][j])
                document_prob_list.append(document_word_prob_list)
            self.topic_prob.append(document_prob_list)
        self.topic_prob = np.asarray(self.topic_prob)

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")

        # update P(z | d)

        # ############################
        # your code here
        # ############################
        
        doc_topic_vocabulary = self.topic_prob.transpose(0, 2, 1)
        self.document_topic_prob = []
        for i in range(self.number_of_documents):
            normalizer = np.sum(np.inner(doc_topic_vocabulary[i], self.term_doc_matrix[i]))
            topic_prob_list = []
            for j in range(number_of_topics):
                val = np.inner(self.term_doc_matrix[i], doc_topic_vocabulary[i][j])
                val = val / normalizer
                topic_prob_list.append(val)
            self.document_topic_prob.append(topic_prob_list)
        self.document_topic_prob = np.asarray(self.document_topic_prob)

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        
        likelihoods = 0.0
        prob = np.matmul(self.document_topic_prob, self.topic_word_prob)
        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                likelihoods = likelihoods + math.log(prob[i][j] + 0.0000001, 2) * self.term_doc_matrix[i][j]
        self.likelihoods.append(likelihoods)
        return likelihoods

    def plsa(self, number_of_topics, topic_word_prob, max_iter, epsilon):

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
        self.topic_word_prob = topic_word_prob

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

def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 10
    epsilon = 0.001
    topic_word_prob = load_topics()
    corpus.plsa(number_of_topics, topic_word_prob, max_iterations, epsilon)


def load_topics():
    return [[0.25, 0, 0.25, 0, 0, 0.25, 0, 0.25], [0, 0.25, 0, 0.25, 0.25, 0, 0.25, 0]]

if __name__ == '__main__':
    main()
