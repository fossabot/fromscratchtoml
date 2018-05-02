import torch as ch
import numpy as np
from fromscratchtoml.models.statistics import Statistics


class GaussianNaiveBayesClassifier(object):
    def __init__(self):
        self.__use_cuda = False
        self.__classes = list()
        self.__num_attributes = 0
        self.__model = dict()

    def fit(self, x, y):
        if isinstance(x, np.ndarray):
            x = ch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = ch.from_numpy(y)
        if self.__use_cuda:
            x = x.cuda()
            y = y.cuda()

        self.__classes = ch.from_numpy(np.unique(y.numpy()))
        self.__num_attributes = x.size()[-1]

        for attribute_index in range(self.__num_attributes):
            for class_label in self.__classes:
                indices = (y == class_label).nonzero().squeeze()
                temp_x = ch.index_select(x, 0, indices)[:, attribute_index]
                temp_x = temp_x.type(ch.FloatTensor)
                mean = temp_x.mean()
                sd = Statistics.standard_deviation(temp_x.unsqueeze(dim=1))
                self.__model[(attribute_index, class_label)] = (mean, sd)

    def predict(self, x):
        result = dict()
        for class_label in self.__classes:
            probability = 0
            for attribute_index in range(self.__num_attributes):
                mean, sd = self.__model[(attribute_index, class_label)]
                prob = Statistics.gaussian_probability(x[attribute_index], mean, sd)
                probability += ch.log(prob)
                result[class_label] = probability.numpy()

        result = sorted(result.items(), key=lambda t: t[1], reverse=True)
        return result[0][0]
