# Use NVIDIA's PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:20.03-py3

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the rest of the application files to the container
COPY . .

# Expose the port for the flask API
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]
