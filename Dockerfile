FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy all the application files
COPY --chown=user . /app

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Switch to the backend directory and run gunicorn
WORKDIR /app/backend
# Hugging Face Spaces sets port to 7860 by default
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
