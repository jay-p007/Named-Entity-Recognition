# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Copy the pre-downloaded wheels directory
COPY wheels /wheels

# Install dependencies from local wheels first, then install other dependencies
RUN pip install --no-cache-dir /wheels/* && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the required port (if your FastAPI/Flask app runs on port 8000)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
