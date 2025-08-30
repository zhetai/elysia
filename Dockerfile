FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Optional: Set environment variables here or in docker-compose.yml
# ENV WCD_URL=...
# ENV WCD_API_KEY=...

# Install pip dependencies if you have requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Start your Python script
CMD ["python", "elysia/util/client.py"]
