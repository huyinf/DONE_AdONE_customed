# Use an official Python base image with Python 3.5
FROM python:3.5-slim

# Set the working directory inside the container
WORKDIR /src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements into the container
COPY requirements.txt ./requirements.txt

# Install the Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project code into the container
COPY . .

# Create a shell script to run both Python commands
RUN echo '#!/bin/bash\npython run_done.py --config config_done\npython run_adone.py --config config_adone' > run_all.sh \
    && chmod +x run_all.sh

# Specify the command to run your application
CMD ["bash", "./run_all.sh"]
