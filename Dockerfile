FROM python:3.8-slim-buster

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD ["streamlitApp.py"]
