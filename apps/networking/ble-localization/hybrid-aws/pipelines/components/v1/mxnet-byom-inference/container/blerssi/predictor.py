# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
import pickle

from six import BytesIO
import flask
import pandas as pd
import mxnet as mx
from mxnet import nd, gluon
import numpy as np

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

def _npy_loads(data):
    """
    Deserializes npy-formatted bytes into a numpy array
    """
    stream = BytesIO(data)
    return np.load(stream)


def _npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                deserialized_net = gluon.nn.SymbolBlock.imports(model_path+"/blenet-symbol.json", ['data'], model_path+"/blenet-0100.params", ctx=ctx)
                cls.model = deserialized_net

        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    if flask.request.content_type == 'application/x-npy':
        # Decode data from request here
        data = _npy_loads(flask.request.data)
        print('Request invoked with content_type application/x-npy')
    else:
        return flask.Response(response='This predictor only supports numpy data', status=415, mimetype='text/plain')

    pred = nd.array(data)
    pred_data = gluon.data.DataLoader(gluon.data.ArrayDataset(pred),batch_size=1)
    for i, data in enumerate(pred_data):
        data = data.as_in_context(model_ctx)
        output = ScoringService.predict(data)
        predictions = nd.argmax(output, axis=1)

    result_tuple = output, predictions

    # Do the prediction
    # predictions = ScoringService.predict(data)

    # Convert from tuple to serialized data
    result = pickle.dumps(result_tuple, protocol=0)

    return flask.Response(response=result, status=200, mimetype='application/json')
