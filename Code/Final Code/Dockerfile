FROM python:3

ADD requirements.txt /

RUN pip install -r requirements.txt

COPY . .

CMD [ "start.sh" ]
