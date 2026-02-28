FROM python:3.12

WORKDIR /app

# Install Google Cloud SDK + BigTable emulator
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
      gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y google-cloud-cli google-cloud-cli-bigtable-emulator && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir --upgrade -r requirements-dev.txt
RUN pip install -e .
