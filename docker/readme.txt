# This work build upon the Docker container created in this example:
# https://towardsdatascience.com/tensorflow-object-detection-with-docker-from-scratch-5e015b639b0b

1. Manual download dependencies needed for the container
# Download TensorFlow models and examples
git clone https://github.com/tensorflow/models.git

# Download EdjeElectronics tutorial
git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-#Tutorial-Train-Multiple-Objects-Windows-10.git

# Download (and unzip) inference_graph into folder inference_graph
https://www.dropbox.com/s/va9ob6wcucusse1/inference_graph.zip?dl=0

Your directory should now look like this:

+ edje-tutorial
+ inference_graph
+ ts_models (contains models/research/... folders)
DockerFile
faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
readme.txt

2. Build and run the docker container
Download and install Docker if yoiu havent already, make sure your Docker is set to "Linux Containers"
Now open up a command line and navigate to this directory

docker build -t ts-api .
< = Should finish without any major errors

Now start the container:
docker run -it --name ts-api -p 8888:8888 ts-api /bin/bash
If you donw want interactive container, simply run
docker run --name ts-api -p 8888:8888 ts-api

If everything is ok, you should be able to browse http://localhost:8888 and see the Jupyter login prompt


