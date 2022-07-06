# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from __future__ import print_function
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader
from django.urls import reverse

import os
import shap
import warnings
import logging
# import tensorflow.python.compat.v2_compat as K
from tensorflow.python.keras.models import Model
import numpy as np

import tensorflow as tf
np.random.seed(1337)  # for reproducibility


from keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Lambda
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)

from sklearn.preprocessing import Normalizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
import h5py
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.layers import LSTM, GRU, SimpleRNN
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

class model:
    def __init__(self, X_train, X_test, y_train, y_test, X, Y):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = X
        self.Y = Y

def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))

def capture(request):
    # Nhấn nút start capture
    if request.method == 'GET':
        os.system("cicflowmeter -i Wi-Fi -c Hehe.csv")
        return render(request, 'capture.html')



# hàm explain file capture bắt được 
def explain(request, file, model):
    if(request.method == 'GET'):
        warnings.filterwarnings("ignore")
        logger = logging.getLogger('shap')
        logger.disabled = True

        explainer = shap.GradientExplainer(model, file)
        shap_values = explainer.shap_values(file)

        shap_values = np.asarray(shap_values)
        file = np.reshape(file, (file.shape[0],file.shape[1]))
        shap_values = np.reshape(shap_values, (shap_values.shape[0],shap_values.shape[1], shap_values.shape[2]))
        file = pd.DataFrame(file)

        # benign: 1; dos: 2; probe: 3; ddos: 4; brute_force: 5; botnet: 6; web_attack: 7 
        class_name = ['benign', 'dos', 'probe', 'ddos', 'brute_force', 'botnet', 'web_attack']
        shap_values = list(shap_values)
        shap.summary_plot(shap_values, file.values, plot_type='bar', class_names=class_name, feature_names=model.X.columns)

def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
