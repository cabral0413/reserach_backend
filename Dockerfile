# Use the official Python 3.10 slim base image
FROM python:3.10-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port the app will run on (5000 default or any other via $PORT)
EXPOSE 5000

# Set the environment variable for PORT (Optional, if not already set in the platform)
ENV PORT 5000

# Command to run the app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "--timeout", "90", "app:app"]

