FROM python:3.9

ENV VIDEO_APP app.py

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000
CMD [ "python3", "app.py" ]