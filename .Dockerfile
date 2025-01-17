# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create volume for database persistence
VOLUME ["/app/data"]

# Create script for database updates
COPY scripts/update_db.sh /app/scripts/
RUN chmod +x /app/scripts/update_db.sh

# Expose Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "src/pypi_lens/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# scripts/update_db.sh
#!/bin/bash
# Script to update PyPI package database
python -m src.pypi_lens.update_index

# azure-pipelines.yml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  imageRepository: 'pypi-lens'
  containerRegistry: 'yourregistry.azurecr.io'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  jobs:
  - job: Build
    steps:
    - task: Docker@2
      inputs:
        containerRegistry: 'ACR'
        repository: '$(imageRepository)'
        command: 'buildAndPush'
        Dockerfile: '$(dockerfilePath)'
        tags: |
          $(tag)
          latest
