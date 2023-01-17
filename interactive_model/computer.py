import tensorflow as tf
import numpy as np


class computer():
    def __init__(self, train_set, train_categories, test_set=None, test_categories=None):
        self.train_set = train_set
        self.train_categories = train_categories

        if test_set:
            self.test_set = test_set
        if test_categories:
            self.test_categories = test_categories

