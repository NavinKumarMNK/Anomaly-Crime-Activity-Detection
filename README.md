# Anomaly Detection & Recognition System

- Completed : Training & Bugs Pending

## Pytorch video recognition API with TensorRT support in a Docker container

### Building the image
- To build the image, navigate to the directory where the Dockerfile is located, and run the following command:
```
docker build -t pytorch-video-recognition-flask-tensorrt
```
This will build the image and tag it with the name "pytorch-video-recognition-flask-tensorrt".

### Running the container
- Once the image is built, you can use it to run the container by using the following command:
```
docker run -p 5000:5000 pytorch-video-recognition-flask-tensorrt
```
This command will start the container and run the video recognition script on it, the container will listen on port 5000, and you can access the API through `http://localhost:5000`

### Pushing the image to DockerHub
- To share the image with others, you can push it to a container registry like DockerHub. First, you will need to create an account on DockerHub and then you can use the following commands to log in and push the image:
```
docker login
docker push pytorch-video-recognition-flask-tensorrt
```
### Pulling the image and running it
- Once the image is pushed to DockerHub, others can use the following command to pull the image and run the container:
```
docker pull pytorch-video-recognition-flask-tensorrt
docker run -p 5000:5000 pytorch-video-recognition-flask-tensorrt
```

**Note:** Make sure that the host machine has the required dependencies, such as NVIDIA drivers and CUDA, to run the container properly.

### Using the API
- The API has two endpoints, one for uploading a video file and the other for getting the results of the video recognition process.
- To upload a video file, you can use the following command:
```
curl -F "file=@/path/to/video/file" http://localhost:5000/predict
```

- Returns images of persons, with prediction of crime-activity