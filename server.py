from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import json
import base64
from flask_cors import cross_origin

'''
模拟调用推理服务
'''
def aqi_predict(data):
    model_path = 'aqi_model_pm25_medium_quality'
    test_data = pd.DataFrame(data)
    predictor = TabularPredictor.load(model_path)
    y_pred = predictor.predict(test_data)
    return y_pred.values.tolist()

from flask import Flask, request, jsonify
app = Flask(__name__)

'''
模拟调用bedrock服务
'''
def gen_image(image_path='beijing.jpg'):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        encoded_string = base64.b64encode(image_data).decode('utf-8')
        return encoded_string

'''
模拟调用noaa api,根据位置获取天气信息
'''
def get_whether(data):
    return {"DEWP":{"1":38.3},"WDSP":{"1":0.4},"MAX":{"1":70.0},"MIN":{"1":45.0},"PRCP":{"1":0.0},"TEMP":{"1":55.0},"MXSPD":{"1":2.9}}

'''
lambda handler 函数
'''
def predict(data):
    wather_data = get_whether(data)
    aqi = aqi_predict(wather_data)
    image = gen_image()
    return {"image": image, "aqi":aqi}

'''
模拟 api gateway service
'''
@app.route('/predict', methods=['POST'])
@cross_origin()
def flask_predict():
    data = request.json
    #logging.info(data)
    
    try:
        predictions = predict(data)
        #logging.info(json.dumps(predictions))
        return jsonify(json.dumps(predictions))
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
