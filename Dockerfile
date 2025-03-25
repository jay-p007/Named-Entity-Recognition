# Use the official Python 3.10.16 image
FROM python:3.10.16

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the required port (if your FastAPI/Flask app runs on port 10000)
EXPOSE 10000

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "10000"]
