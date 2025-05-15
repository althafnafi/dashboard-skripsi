# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create a non-root user and switch to it
RUN useradd -m streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Run streamlit.py when the container launches
CMD ["streamlit", "run", "streamlit.py", "--server.address=0.0.0.0"] 