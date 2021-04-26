FROM digi0ps/python-opencv-dlib

RUN pip install -U pip wheel cmake
RUN pip install opencv-python dlib

WORKDIR /app
COPY . /app

CMD [ "python", "main.py" ]