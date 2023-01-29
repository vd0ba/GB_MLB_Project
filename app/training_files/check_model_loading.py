import pickle

import pandas as pd

from sklearn.metrics import roc_auc_score

from model_transforms import NumberTaker, ExperienceTransformer, NumpyToDataFrame


if __name__ == '__main__':
    # Read data
    data = pd.read_csv("data/aug_train.csv")
    
    # Load model
    with open('../models/ctb_clf.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Check model is working with prediction
    preds = model.predict(data.drop(columns=['target']))
    roc = roc_auc_score(preds, data['target'])
    print(roc)
