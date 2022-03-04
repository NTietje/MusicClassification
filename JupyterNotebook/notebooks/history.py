import pickle


class History_trained_model(object):
    """ Creates an History_trained_model out of the a keras history object.
    The created model can be saved as JSON.

    Parameters
    ----------
    object : history, epoch and params of the keras history object.

    """
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params


def load_history(filename):
    """Loads and returns the JSON object as History_trained_model.
    The given filename have to match the JSON object name.

    Parameters
    ----------
    filename : str

    Returns
    -------
    history : history object
    """
    with open('histories/' + filename + '.json', 'rb') as file:
        history = pickle.load(file)
        # print('load history')
    return history


def save_history(history, filename):
    """Saves the models history as History_trained_model.

    Parameters
    ----------
    history : history objects
    filename : str
    """
    with open('histories/' + filename + '.json', 'wb') as file:
        model_history = History_trained_model(history.history, history.epoch, history.params)
        pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)
        # print('saved history')

