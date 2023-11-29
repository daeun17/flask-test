# from
FROM ubuntu:latest

# apt init
ENV LANG=C.UTF-8
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata g++ git curl


# python stuff
RUN apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    /usr/local/bin/python -m pip install --upgrade pip

# apt cleanse
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# copy resources
COPY . .

# timezone
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# workspace
RUN mkdir -p /workspace
WORKDIR /workspace

FROM python:3.6

# venv 생성 
RUN python -m venv gentledog
RUN chmod +x gentledog/bin/activate
RUN gentledog/bin/activate

COPY . /app

RUN pip3 install flask

WORKDIR /app

# pip install 실행
RUN pip install -r requirements.txt


CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port", "5000"]