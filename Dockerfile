FROM python:3.8.6-slim-buster

LABEL maintainer "Toby Hall, tobyhall658@gmail.com"
LABEL maintainer2 "Matt Swinnock, matthew.swinnock@ons.gov.uk"

# set working directory in container
WORKDIR /usr/src/app

# Copy and install packages
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# Copy app folder to app folder in container
COPY app.py /usr/src/app/
COPY /assets /usr/src/app/
COPY /data /usr/src/app/

# Changing to non-root user
RUN useradd -m appUser
USER appUser

# Run locally on port 8050
CMD gunicorn --bind 0.0.0.0:8050 app:server