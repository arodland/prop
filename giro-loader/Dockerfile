FROM python:latest

WORKDIR /usr/local/tick

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY * ./

CMD [ "python", "./tick.py" ]
