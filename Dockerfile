# Use Python 3.11 for better performance and security
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies with optimizations for production
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create necessary directories
RUN mkdir -p vector_db

# Run as non-root user for security
RUN useradd -m -u 1001 streamlit && \
    chown -R streamlit:streamlit /app
USER streamlit

# Expose the port that Streamlit will run on
EXPOSE 8501

# Add health check for better container management
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set the command to run the Streamlit app with production settings
CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
