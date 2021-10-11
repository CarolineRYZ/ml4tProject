import numpy as np

class RTLearner(object):

    # constructor
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None


    # my Georgia Tech username
    def author(self):
        return "CarolineRYZ"


    # A Cutler's algorithm for random tree
    def build_tree(self, data_x, data_y):

        # recursion stopping criteria
        # only one sample left
        if data_x.shape[0] == 1:
            return np.array([["leaf", np.mean(data_y), np.nan, np.nan]])

        # all data_y same
        if len(np.unique(data_y)) == 1:
            return np.array([["leaf", np.mean(data_y), np.nan, np.nan]])

        # sample size less than leaf size
        if data_x.shape[0] <= self.leaf_size:
            return np.array([["leaf", np.mean(data_y), np.nan, np.nan]])

        else:
            # determine random feature to split on
            index = np.random.randint(0, data_x.shape[1]-1)
            split_value = np.median(data_x[:, index])

            # corner case: no right tree
            if split_value == max(data_x[:, index]):
                return np.array([["leaf", np.mean(data_y), np.nan, np.nan]])


            # build left and right trees recursively
            left_index = data_x[:, index] <= split_value
            left_tree = self.build_tree(data_x[left_index], data_y[left_index])

            right_index = data_x[:, index] > split_value
            right_tree = self.build_tree(data_x[right_index], data_y[right_index])

            root = np.array([[index, split_value, 1, left_tree.shape[0]+1]])

            return np.vstack((root, left_tree, right_tree))


    # add training data to DTLearner
    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)


    # estimate the test output y
    def query(self, points):
        test_y = []
        for i in range(points.shape[0]):
            j = 0
            while(self.tree[j, 0] != "leaf"):
                split_value = self.tree[j, 1]
                if points[i, int(float(self.tree[j, 0]))] <= float(split_value):
                    j += int(float(self.tree[j, 2]))

                else:
                    j += int(float(self.tree[j, 3]))

            y = self.tree[j, 1]
            test_y.append(float(y))

        return test_y


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
