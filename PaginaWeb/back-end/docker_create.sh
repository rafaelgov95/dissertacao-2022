docker pull tensorflow/serving:latest
cd cnnwebv2
source venv/bin/activate
export FLASK_APP=./autoapp.py
export DATABASE_URL=mongodb://localhost/flask
waitress-serve --call 'app:create_app'

pip install git+https://github.com/aaren/wavelets

pip install opencv-python
sudo apt update && sudo apt install -y libsm6 libxext6
sudo apt-get install -y libxrender-dev


iptables -A INPUT -i ens4 -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -i ens4 -p tcp --dport 8080 -j ACCEPT
iptables -A PREROUTING -t nat -i ens4 -p tcp --dport 80 -j REDIRECT --to-port 8080

    
docker run -p 8501:8501 \
    --rm --mount type=bind,source=$(pwd),target=$(pwd) \
    --name vfc \
    -e MODEL_BASE_PATH=/home/user/cnnwebv2/back-end/tensorflow_models \
    -e MODEL_NAME=vfc -t docker.io/tensorflow/serving:latest 
 
docker run --rm -p 8502:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    --name covid19 \
    -e MODEL_BASE_PATH=/home/user/cnnwebv2/tensorflow_models \
    -e MODEL_NAME=covid19 -t docker.io/tensorflow/serving:latest


docker run --rm -p 8502:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    --name covid19 \
    -e MODEL_BASE_PATH=/home/$(USER)/Documentos/Projetos/Meus/CNN/cnnwebv2/tensorflow_models \
    -e MODEL_NAME=covid19 -t tensorflow/serving:latest
    
docker run --rm -p 8501:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    --name vfc \
    -e MODEL_BASE_PATH=/home/rafael/Documentos/Projetos/Meus/CNN/cnnwebv2/back-end/tensorflow_models \
    -e MODEL_NAME=vfc -t tensorflow/serving:latest

docker run --rm -p 8502:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    --name covid19 \
    -e MODEL_BASE_PATH=/home/rafael/Documentos/Projetos/Meus/CNN/cnnwebv2/back-end/tensorflow_models \
    -e MODEL_NAME=covid19 -t tensorflow/serving:latest

docker run --rm -p 8503:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    --name vfc \
    -e MODEL_BASE_PATH=/home/rafael/Documentos/Projetos/CNN/cnnwebv2/back-end/tensorflow_models \
    -e MODEL_NAME=vfc-rv -t tensorflow/serving:latest