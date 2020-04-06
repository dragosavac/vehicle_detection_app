# object_detection_app



Example of usage:

This is an object detection app created with Python/Django/Docker/Postgres stack. 

Requirements:
You'll need to have installed Docker and Postgres on your machine.

Settings for different environments are stored in object_detection app folder, along with urls.py(endpoints) and wsgi.py files.

Inference app is the folder where I've stored Models, Services, Serializers and Views.

After creating a superuser, you'll be able to see all values stored in database on localhost:8000/admin

In training_model file, you can see explained jupyter notebook calculation for predicting position of truck in the lane

In utilities folder, you can see functions for predicting vehicle lane, along with constants.


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

Examples for testing API could be found documented on the link below:

```
https://documenter.getpostman.com/view/4800952/SzYbzdP5?version=latest
```


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
