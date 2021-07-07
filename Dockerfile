FROM python:3.7

COPY . /app
RUN pip install pip==20.0.1 \
    && pip install --no-cache-dir --upgrade -r requirements.txt \
    && pip install --no-cache-dir --upgrade -r requirements-dev.txt
