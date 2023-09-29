# __init__
'''
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import tensorflow.python.util.deprecation as deprecation
import glob as glob
import random
import keras.backend as K
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import copy
import argparse
import keras
import warnings

from sklearn.linear_model import LogisticRegression
from numpy import load
from scipy.special import logit
from scipy.stats import bootstrap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed # for parallel processing
from keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.compat.v1.keras.utils import to_categorical
from keras.layers import Layer, Input, Dense, Concatenate, BatchNormalization, Dropout, Softmax
from tensorflow.compat.v1.keras.utils import register_keras_serializable
from keras.models import Model,load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from lightgbm import LGBMRegressor, LGBMClassifier
from IPython.display import clear_output'''

# __init__

#import sys
#!{sys.executable} -m pip install scipy==1.7.1

import pandas as pd
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import glob as glob
import random
import keras.backend as K
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import copy
import argparse
import keras
import warnings


from numpy import load
from joblib import Parallel, delayed # for parallel processing
from scipy.special import logit
from scipy.stats import bootstrap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed # for parallel processing
from keras.metrics import binary_accuracy, categorical_accuracy
#from keras.optimizer_v1 import rmsprop as RMSprop, sgd as SGD, Adam
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.compat.v1.keras.utils import to_categorical
from keras.layers import Layer, Input, Dense, Concatenate, BatchNormalization, Dropout, Softmax
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from lightgbm import LGBMRegressor
from IPython.display import clear_output

