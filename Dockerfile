 FROM python:3
 ENV PYTHONUNBUFFERED 1
 RUN mkdir /strax
 WORKDIR /strax
 ADD . /strax/
 RUN pip install -r requirements.txt
 RUN python setup.py install
