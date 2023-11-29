# Flask : Flask server 
# jsonify : JSON응답데이터를 만들어 주는 메서드 
from flask import Flask,jsonify,request
from os.path import join
import os 
import json
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import ResNet50
from PIL import Image
import json
from flask_cors import CORS


# Flask 객체를 app에 할당 
app = Flask(__name__)
CORS(app)

##### API routing #####

@app.route("/image_predict",methods=['POST'])
def prdict_image():
    
    # 이미지 받기 
    img_file = request.files['img']

    # # 이미지를 base64로 디코딩
    # img_data = base64.b64decode(img.read())
    
    # # BytesIO 객체로 이미지 읽기
    # img = Image.open(BytesIO(img_data))
    
    # 변경할 사이즈 
    image_size = 224
    # 모델 저장위치 
    # model_weight_path = '../AI-models/AI_test/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    # 예측할 사진 위치 
    # img_path = img_file
    # class 저장 파일 위치 
    # class_list_path = '../AI-models/AI_test/imagenet_class_index.json'
    model_weight_path = os.getenv('MODEL_WEIGHT_PATH', './AI-models/AI_test/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    class_list_path = os.getenv('CLASS_LIST_PATH', './AI-models/AI_test/imagenet_class_index.json')
    
    # kaggle learntools 라이브러리 내의 decode_predictions 함수 
    # https://github.com/Kaggle/learntools
    def decode_predictions(preds, top=5, class_list_path='../input/resnet50/imagenet_class_index.json'):
        """Decodes the prediction of an ImageNet model.
        Arguments:
            preds: Numpy tensor encoding a batch of predictions.
            top: integer, how many top-guesses to return.
            class_list_path: Path to the canonical imagenet_class_index.json file
        Returns:
            A list of lists of top class prediction tuples
            `(class_name, class_description, score)`.
            One list of tuples per sample in batch input.
        Raises:
            ValueError: in case of invalid shape of the `pred` array
                (must be 2D).
        """
        if len(preds.shape) != 2 or preds.shape[1] != 1000:
            raise ValueError('`decode_predictions` expects '
                            'a batch of predictions '
                            '(i.e. a 2D array of shape (samples, 1000)). '
                            'Found array with shape: ' + str(preds.shape))
        CLASS_INDEX = json.load(open(class_list_path))
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            results.append(result)
        return results


    # 이미지 사이즈 변경 함수 
    def read_and_prep_image(img_file, img_height=image_size, img_width=image_size):
        img = Image.open(img_file.stream)
        img = img.resize((img_height, img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 모델에 맞게 차원 확장
        img_array = preprocess_input(img_array)
        return img_array


    # predict 및 결과 출력 함수 
    def model_predict (model_weight_path, img_file, class_list_path):
        my_model = ResNet50(weights=model_weight_path)
        test_data = read_and_prep_image(img_file)
        preds = my_model.predict(test_data)
        most_likely_labels = decode_predictions(preds, top=3, class_list_path=class_list_path)
        return most_likely_labels[0][0][1],most_likely_labels[0][1][1]
    
    result1, result2 = model_predict(model_weight_path, img_file, class_list_path)
    # print(result)
    data = {'result1':result1,'result2':result2}
    return jsonify(data)

###### API routing end #######

# 메인 모듈로 실행될 때 플라스크 서버구동 
if __name__ == "__main__":              
    app.run(host="0.0.0.0", port="5000")