import numpy as np
import math
from collections import Counter
from collections import defaultdict


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
        
        with open(self.documents_path) as f:
            for line in f:
                word_list = line.split()
                # Remove the '0' or '1' label at the beginning of a document
                if word_list[0] in ['0','1']:
                    word_list = word_list[1:]
                    
                self.documents.append(word_list)
                self.number_of_documents += 1
                

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        
        # Flatten self.documents to a 1D list, which can then be easily converted to a set to remove duplicated vocabulary
        flat_list = sum(self.documents, [])
        
        self.vocabulary = list(set(flat_list))
        self.vocabulary.sort() #sort the vocabulary words
        self.vocabulary_size = len(self.vocabulary)
        
        

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        
        # Initialize the 2D matrix
        rows, cols = (self.number_of_documents, self.vocabulary_size)
        self.term_doc_matrix = [[0 for j in range(cols)] for i in range(rows)]
        
        for i, doc in enumerate(self.documents):
            c = Counter(doc)
            # Write term counts in matrix
            for j, word in enumerate(self.vocabulary):
                self.term_doc_matrix[i][j] = c[word]
                  
        

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        
        arr = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(arr)
        
        arr = np.random.random_sample((number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(arr)
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    
    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        #print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)
            

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        #print("E step:")
        
        for doc_index, pi_arr in enumerate(self.document_topic_prob):
            for word_index in range(self.vocabulary_size):
                # P(w | z) for all the z
                word_topics_arr = [row[word_index] for row in self.topic_word_prob]
                normalizer = np.dot(pi_arr, word_topics_arr)
                
                for topic_index, pi_value in enumerate(pi_arr):           
                    # update P(z | d, w)
                    self.topic_prob[doc_index][topic_index][word_index] = pi_value * self.topic_word_prob[topic_index][word_index] / normalizer
                         
                        

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        #print("M step:")

        
        # update P(z | d)
        
        for doc_index in range(self.number_of_documents):
            z_matrix = self.topic_prob[doc_index]
            term_count_arr = self.term_doc_matrix[doc_index]
            
            denominator = 0
            for topic_index in range(number_of_topics):
                denominator += np.dot(term_count_arr, z_matrix[topic_index])
                
            for topic_index in range(number_of_topics):
                numerator = np.dot(term_count_arr, z_matrix[topic_index])
                self.document_topic_prob[doc_index][topic_index] = numerator / denominator
        
        # update P(w | z)
        
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
                    
    
    def maximization_with_prior(self, number_of_topics, pseudo_count, conjugate_prior):
        """ The M-step updates P(w | z)
        """
        #print("M step:")

        
        # update P(z | d)
        
        for doc_index in range(self.number_of_documents):
            z_matrix = self.topic_prob[doc_index]
            term_count_arr = self.term_doc_matrix[doc_index]
            
            denominator = 0
            for topic_index in range(number_of_topics):
                denominator += np.dot(term_count_arr, z_matrix[topic_index])
                
            for topic_index in range(number_of_topics):
                numerator = np.dot(term_count_arr, z_matrix[topic_index])
                self.document_topic_prob[doc_index][topic_index] = numerator / denominator
        
        # update P(w | z)
        
        for topic_index in range(number_of_topics):
            # Force topic 0 to incorporate prior
            denominator = pseudo_count if topic_index == 1 else 0
            for word_index in range(self.vocabulary_size):
                for doc_index in range(self.number_of_documents):
                    denominator += self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
                    
            for word_index, word in enumerate(self.vocabulary):    
                numerator = pseudo_count * conjugate_prior[word] if topic_index == 1 else 0
                for doc_index in range(self.number_of_documents):
                    numerator += self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
                    
                self.topic_word_prob[topic_index][word_index] = numerator / denominator

                    

    def calculate_likelihood(self):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
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
        
        return
    

    def plsa(self, number_of_topics, max_iter, epsilon):

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
            #print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood()
            
        print("\n----------Likelihoods-----------")
        print(self.likelihoods)
        
        
        
    def plsa_with_prior(self, number_of_topics, max_iter, epsilon, pseudo_count, conjugate_prior):

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
            #print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_with_prior(number_of_topics, pseudo_count, conjugate_prior)
            self.calculate_likelihood()
            
        print("\n----------Likelihoods----------- ")
        print(self.likelihoods)
        

def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    
    pseudo_count = 10000
    conjugate_prior = {'seattle':0.5, 'rainier':0.5}
    conjugate_prior = defaultdict(lambda:0, conjugate_prior)  #set default value to 0
    
    #corpus.plsa(number_of_topics, max_iterations, epsilon)
    corpus.plsa_with_prior(number_of_topics, max_iterations, epsilon, pseudo_count, conjugate_prior)
    
    np_array1 = np.array(corpus.document_topic_prob)
    np_array2 = np.array(corpus.topic_word_prob)
    #np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    np.set_printoptions(suppress=True)
    print('\n--------------Document-Topic-Prob---------------')
    print(np_array1)
    print('\n--------------Topic-Word_Prob---------------')
    print(np_array2)
    


if __name__ == '__main__':
    main()
