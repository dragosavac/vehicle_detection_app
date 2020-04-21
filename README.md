# Vehicle detection app



## Machine Learning Concepts

### Machine Learning Models

In machine learning, a **model** is umbrella term that, in the broadest sense,
represents a mathematical function that receives input data, and produces some
output data. For example:

- regression models predict output based on input, e.g. ambient temperature for
  tomorrow based on historical measurements
- classification models assign a label denoting some category to input data,
  e.g. given an image of some fruit, model decides what kind of fruit it is -
  apple, orange, banana etc.
- object-detection models detect objects on input images, e.g. given an image
  of a crowd of people, model draws a rectangle - also called a _bounding box_ -
  around each person in the image
- any combination of the above and much, much more

In this task, you are provided with an **object-detection model** that
detects trucks on input images.

### Model Training and Inference

_Learning_ in machine learning stands for the ability of models to "learn"
causalities and dependencies between input data and output data, thus getting
better - improving accuracy of their output - over time. This happens
during a process called **training**.

When you initially create a model, its performance is very bad - it will output
pretty much random results for any input given. During training, model's
performance slowly improves. Training is often times computationally intensive
and lasts for a long time - it is not uncommon for training to last for days or
weeks, and utilize multiple CPUs and GPUs.

Once model reaches satisfactory level of performance, usually measured in
accuracy, we can then actually use the model. We give the model novel data it
has not seen before, and we let it compute the output. This process is called
**inference**.

To summarize, in order to solve a machine learning problem, you:
1. Build a model (designing)
2. Train it (training)
3. Run it (inference)

In this assignment, model you are provided with has already been trained, and
you will be dealing with the last part, **inference**.

### Infer Model

Infer model you can see is an object-detection model that detects
trucks on specific camera images. The camera is fixed, and it monitors arrival
of trucks to cargo delivery area around the clock. These trucks orderly wait in
multiple lanes to deliver their cargo.

Here are a couple of examples of input images:

![input1](./images/port_elizibeth_webcam_2018-10-15T10-04-02.jpg)

![input2](./images/port_elizibeth_webcam_2018-10-15T12-36-02.jpg)

![input3](./images/port_elizibeth_webcam_2018-10-15T13-58-01.jpg)

For each object detected, the model outputs 4 bounding box coordinates, a class
label and a confidence score in `[0, 1.0]` range. If we were to draw bounding boxes on
top of original image, we would get something like this:

![input1-pred](./images/port_elizibeth_webcam_2018-10-15T10-04-02-pred.jpg)

![input2-pred](./images/port_elizibeth_webcam_2018-10-15T12-36-02-pred.jpg)

![input3-pred](./images/port_elizibeth_webcam_2018-10-15T13-58-01-pred.jpg)

To recap:
- model input is a 320x240 JPEG image
- model output is a JSON dictionary, containing the following fields:
    - `boxes` - a list of bounding box coordinates for each detected object
    - `scores` - a list of confidence scores for each detected object
    - `classes` - a list of class labels for each detected object (`1` is `truck`)

The model is provided as a frozen TensorFlow v1.x graph with weights.
[TensorFlow](https://www.tensorflow.org/) is one of the most prominent machine
learning frameworks. It is written in C++ and Python, with latter being the main API.
You are also given a Python script that can be used to run
inference on a single image - take a look at `infer.py`. This script can be
invoked with:
```bash
pip install -r requirements.txt
./infer.py --threshold 0.6 --o <output_image> <input_image>
```

Object detection app is  **inference service** that will make use of
this model to **count the number of trucks in each lane**. Additionally,
service is also calculating traffic load statistics per lane for custom
date/time ranges.

Inference service is a web app - backend only, no frontend - that serves
requests. Your service should be backed by a database. The service you are building
must have the following two endpoints:

1. **Inference endpoint**. Each request to this endpoint contains a single
   input image, and time when the image was taken. Steps this endpoint is passing through:
   - parse incoming request, deserialize the image and pass it to the model as input
   - run the model and collect model output
   - count number of trucks in each lane - there are 10 visible lanes in
     input images, so you need to come up with 10 numbers, one per lane
   - store truck count in each lane in the database, along with image timestamp
   - finally, send back response containing number of trucks per lane
2. **Statistics endpoint**. Requests to this endpoint contain a date/time range,
   e.g. from `2020-03-01 12:00:00` up to `2020-03-05 15:30:35`. Steps this endpoint is passing through:
   - load relevant data from the database
   - calculate maximum, minimum, median, mean and standard deviation for truck
     counts for each lane for provided time range
   - send back response containing this data
   
## Directories overview
```
    
    ├── infer model
    │    ├── images      
    │    ├── model     
    │    ├── infer.py        
    │    └── PortElizabethWebCam.zip  
    ├── inference_app                 contains Models, Services, Serializers and Views
    │    ├── migrations      
    │    ├── admin.py     
    │    ├── apps.py        
    │    ├── inference_service.py
    │    ├── models.py
    │    ├── serializers.py
    │    ├── tests.py
    │    └── views.py
    │
    ├── object_detection_app
    │    ├── settings                 contains settings for different environments
    │        ├── base.py        
    │        ├── local.py   
    │        ├── production.py
    │        └── test.py
    │    ├── init.py  
    │    ├── wsgi.py
    │    └── urls.py
    │
    ├── training_model.py
         ├── Train_model.ipnb.zip     contains explained calculation for predicting position of truck in the lane
    ├── utilities
         ├── constants.py             contains constans
    │    ├── count_statistics.py      contains calculation function for stats per lane
    │    └── predict_vehicle_lane.py  contains functions for predicting number of vehicles per lane
    ├── .dockerignore
    ├── .env
    ├── .gitignore
    ├── docker-compose.yml
    ├── Dockerfile
    ├── Makefile
    ├── manage.py  
    ├── README.md
    └── requirements.txt
   
```
   



#### Example of usage:

This is an object detection app created with Python/Django/Docker/Postgres stack. 

#### Requirements:
You'll need to have installed Docker and Postgres on your local machine.

###### Settings 
for different environments are stored in object_detection app folder, along with urls.py(endpoints) and wsgi.py files.

###### Inference app 
is the folder where I've stored Models, Services, Serializers and Views.

##### Training_model file
is the jupyter notebook file where you can see explained calculation for predicting position of truck in the lane

##### Utilities folder
is the folder where  you can see functions for predicting vehicle lane, along with constants.


TESTING API:
I suggest you to use Postman for testing API-es. First you'll need to fill database with Inference endpoint by sending an
POST with uploaded image and valid format date(example:2019-07-29 18:18:14) on:
 
```
127.0.0.1:8000/api/inference/
```


After that, you'll be able to test Statistics endpoint by hitting a GET endpoint on url like this: 


```
127.0.0.1:8000/api/stats?start_date=2020-03-25 15:15:14.123&end_date=2020-03-30 15:15:16.123
```

Examples for testing API could be found documented on the link [here](https://documenter.getpostman.com/view/4800952/SzYbzdP5?version=latest)


Steps for running the project:

```
docker-compose build
```
 

Do migrations and runserver
```
make migrate
make runserver
```
will start the development server. 

To access the admin portal, run `make createsuperuser` and the command line will guide you through creation of user with admin privileges

Alternatively, Makefile is added to templates with provided commands:
* `runserver`
* `migrations`
* `migrate`
* `createsuperuser`
* `createuser`
* `static` - collectstatic management command
* `translate` - compilemessages management command
* `shell`
* `test`
* `teardown`
