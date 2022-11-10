import argparse
import json

import numpy as np
import requests
from keras.applications import inception_v3
from keras.preprocessing import image


payload = {
    "instances": [{'input_image': img.tolist()}]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9000/v1/models/ImageClassifier:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))

# Decoding the response
# decode_predictions(preds, top=5) by default gives top 5 results
# You can pass "top=10" to get top 10 predicitons
print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))