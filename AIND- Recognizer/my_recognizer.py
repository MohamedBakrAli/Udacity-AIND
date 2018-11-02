import warnings
import logging
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set
   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # Get words from test_set
    hwords = test_set.get_all_Xlengths()

    # In case lists setup fail, go to except and kepp log
    try:

        # Create word_id to match with train data
        for word_id in range(0, len(test_set.get_all_sequences())):

            # Initialization
            words_prob = {}
            best_score = float('-inf')
            guess_word = None

            #Get X nad Lengths from test_set
            X_predict, lenghts_predict = hwords[word_id]

            # match trained model with test_set data by word
            for word, model in models.items():

                # if can not get model score, assign -inf to model's score
                try:

                    score = model.score(X_predict, lenghts_predict)

                except:

                    score = float('-inf')

                words_prob[word] = score

                if score > best_score:
                    guess_word = word
                    best_score = score

            # add result to output lists
            probabilities.append(words_prob)
            guesses.append(guess_word)

    except Exception as e:

            # keep log to recognizer.log
            logging.basicConfig(filename='recognizer.log', level=logging.DEBUG)
            logging.debug(str(e))
            pass

    return probabilities, guesses
