import math
import statistics
import warnings
import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Initilization
        best_score = float('inf')
        best_num_states = None

        try:

            for num_states in range(self.min_n_components, self.max_n_components+1):

                # Initilization
                BIC = None
                logL = None

                model = GaussianHMM(n_components = num_states, n_iter = 1000).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)

                # calculate p for BIC calculation
                parameters = num_states * num_states + 2 * num_states * len(self.X[0]) - 1
                # BIC Calculation
                BIC = (-2 * logL) + (parameters * np.log(len(self.X)))

                if BIC < best_score:
                    best_score = BIC
                    best_num_states = num_states

        except Exception as e:

            # keep log to selector-bic.log
            logging.basicConfig(filename='selector-bic.log', level=logging.DEBUG)
            logging.debug(str(e))
            # print("Failed\n")
            pass

        # Return no. of state which gives best model from CV
        # Use ModelSelect.base_model method to return model
        # If CV is failed, return 3 state model (default value)
        return self.base_model(best_num_states) if best_num_states is not None else self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Initilization
        best_score = float('-inf')
        best_num_states = None
        # Get word for DIC calculation
        M = len(self.words.keys())

        try:

            for num_states in range(self.min_n_components, self.max_n_components+1):

                # Initilization
                DIC = None
                logL = None
                sum_logL = 0

                model = GaussianHMM(n_components = num_states, n_iter = 1000).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)

                for each_word in self.hwords.keys():

                    X_each_word, lengths_each_word = self.hwords[each_word]

                    # Calculate score for each word.
                    # If can not get score, assign 0 instead
                    try:

                        sum_logL += model.score(X_each_word, lengths_each_word)

                    except:

                        sum_logL +=0

                # DIC Calculation
                DIC = logL - (1/(M-1)) * (sum_logL - logL)

                if DIC > best_score:
                    best_score = DIC
                    best_num_states = num_states

        except Exception as e:

            # keep log to selector-dic.log
            logging.basicConfig(filename='selector-dic.log', level=logging.DEBUG)
            logging.debug(str(e))
            pass

        # Return no. of state which gives best model from CV
        # Use ModelSelect.base_model method to return model
        # If CV is failed, return 3 state model (default value)
        return self.base_model(best_num_states) if best_num_states is not None else self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Ignore Runtime Warning occurs in Recognizer Part 3
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Initialisation
        # Minimum model score and no. of state in HMM
        best_score = float('-inf')
        best_num_states = None
        # Check wheter input word has enough samples to CV
        # Defaul split in KFold.split = 3
        n_splits = min(len(self.sequences), 3)

        # Run through min to max state in ModelSelector class
        for num_states in range(self.min_n_components, self.max_n_components+1):

            # Initialisation for each state simulation
            scores = []
            logL = None

            # If something goes wrong, try-except with catch the error
            try:

                # Part 3 has some word with only one example
                # Bypass CV if there is only example
                try:

                    split_method = KFold(n_splits = n_splits)
                    for cv_trian_idx, cv_test_idx in split_method.split(self.sequences):

                        # Get train/test from KFold for CV
                        X_train, lengths_train = combine_sequences(cv_trian_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                except:

                    X_train, X_test = self.X
                    lengths_train, lengths_test = self.lengths

                    # Model fitting
                    model = GaussianHMM(n_components = num_states, n_iter = 1000).fit(X_train, lengths_train)

                    # Get Model Score and add it to list
                    logL = model.score(X_test, lengths_test)
                    scores.append(logL)

                # Get Average score for this simulation
                score = np.mean(scores)

                # Keep record of best socre, model, and no. of state
                if score > best_score:
                    best_score = score
                    best_num_states = num_states

            except Exception as e:

                # keep log to selector-cv.log
                logging.basicConfig(filename='selector-cv.log',level=logging.DEBUG)
                logging.debug(str(e))
                pass

        # Return no. of state which gives best model from CV
        # Use ModelSelect.base_model method to return model
        # If CV is failed, return 3 state model (default value)
        return self.base_model(best_num_states) if best_num_states is not None else self.base_model(self.n_constant)
