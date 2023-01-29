# import the necessary packages
import logging
from logging.handlers import RotatingFileHandler
import os
import pickle   # Or can use joblib
import time

import pandas as pd

import flask

from model_transforms import NumberTaker, ExperienceTransformer, NumpyToDataFrame


def load_model(model_path):
    # load the pre-trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    return model


# Logging
logfile = 'model_api.log'
handler = RotatingFileHandler(filename=logfile, maxBytes=1048576, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Initialize Flask app
app = flask.Flask(__name__)

# Load the model
model_file = 'app/models/ctb_clf.pkl'
model = load_model(model_file)


@app.route('/', methods=['GET'])
def general():
    return """Welcome to employee job change prediction API!"""


@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the view
    response = {'success': False}
    curr_time = time.strftime('[%Y-%b-%d %H:%M:%S]')
    
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == 'POST':
        request_json = flask.request.get_json()
        
        input_data = pd.DataFrame({
            'enrollee_id': None,
            'city': request_json.get('city', ''),
            'city_development_index': float(request_json.get('city_development_index', '')),
            'gender': request_json.get('gender', ''),
            'relevent_experience': request_json.get('relevent_experience', ''),
            'enrolled_university': request_json.get('enrolled_university', ''),
            'education_level': request_json.get('education_level', ''),
            'major_discipline': request_json.get('major_discipline', ''),
            'experience': request_json.get('experience', ''),
            'company_size': request_json.get('company_size', ''),
            'company_type': request_json.get('company_type', ''),
            'last_new_job': request_json.get('last_new_job', ''),
            'training_hours': int(request_json.get('training_hours', '')),
        }, index=[0])
        logger.info(f'{curr_time} Data: {input_data.to_dict()}')
        
        try:
            # Predict the result
            preds = model.predict_proba(input_data)
            response['predictions'] = round(preds[:, 1][0], 5)
            # Request successful
            response['success'] = True
        except AttributeError as e:
            logger.warning(f'{curr_time} Exception: {str(e)}')
            response['predictions'] = str(e)
            # Request unsuccessful
            response['success'] = False
    
    # return the data dictionary as a JSON response
    return flask.jsonify(response)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    port = int(os.environ.get('FLASK_SERVER_PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
