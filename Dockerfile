FROM python:3.13-slim

# System deps for PyMuPDF and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install petey from GitHub + benchmark deps
RUN pip install --no-cache-dir \
    git+https://github.com/afriedman412/petey.git \
    python-dotenv pandas scikit-learn google-cloud-storage \
    unstructured-client==0.42.12

# Copy benchmark runner and data
COPY benchmark.py .
COPY evaluate_claude_data.py .
COPY data/ data/

# Default: dry run (override CMD at deploy time with actual args)
ENTRYPOINT ["python", "benchmark.py"]
CMD ["--dry-run"]
