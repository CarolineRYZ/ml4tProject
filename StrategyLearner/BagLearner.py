from scipy import stats
import numpy as np

class BagLearner(object):

    # constructor
    def __init__(self, learner, kwargs = {"leaf_size": 5}, bags = 20, boost = False, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        # instantiate several learners
        self.learner_ensemble = []
        for i in range(bags):
            self.learner_ensemble.append(learner(**kwargs))


    # my Georgia Tech username
    def author(self):
        return "CarolineRYZ"

    # add training data to bags
    def add_evidence(self, data_x, data_y):
        for learner in self.learner_ensemble:
            index = np.random.choice(data_x.shape[0], data_x.shape[0])
            xtrain = data_x[index]
            ytrain = data_y[index]
            learner.add_evidence(xtrain, ytrain)

    # estimate the test output y
    def query(self, points):
        ytest = np.zeros(points.shape[0])

        for learner in self.learner_ensemble:
            ytest += learner.query(points)

        output = ytest / self.bags

        return output


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
