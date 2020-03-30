FROM python:3
RUN apt-get update
RUN apt-get install gettext swig libssl-dev dpkg-dev netcat -y
ENV PYTHONUNBUFFERED 1

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8000
CMD ["gunicorn", "-w", "10", "-b", ":8000", "object_detection_app.wsgi:application"]