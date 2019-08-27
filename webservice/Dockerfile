# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /webservice
WORKDIR /webservice

# Copy the current directory contents into the container at /webservice
COPY . /webservice

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]