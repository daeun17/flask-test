pipeline {
    agent any

    environment {
        FLASK_APP = './flask/AiService.py'
        FLASK_ENV = 'production'
        PYTHON_VERSION = '3.6'
        MODEL_WEIGHT_PATH = './AI-models/AI_test/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        CLASS_LIST_PATH = './AI-models/AI_test/imagenet_class_index.json'
    }

    stages {
        stage('DockerSize') {
            steps {
                sh '''
                    docker stop flask || true
                    docker rm flask || true
                    docker rmi flask || true
                    docker build -t flask .
                    echo "flask: build success"
                '''
            }
        }
        stage('Deploy') {
            steps {
                sh '''
                docker run -e MODEL_WEIGHT_PATH="${MODEL_WEIGHT_PATH}" -e FLASK_APP="${FLASK_APP}" -e FLASK_ENV="${FLASK_ENV}" -d --name flask --network gentledog -p 5000:5000 flask
                echo "flask: run success"
                '''
                }
        }
    }
}