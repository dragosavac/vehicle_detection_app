"""object_detection_app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
# from inference_app.views import health_check, InferenceView, FileUploadView
from inference_app.views import health_check, InferenceView, StatsView


urlpatterns = [
    url(r'^admin/', admin.site.urls,),
    url(r'^api/healthcheck', health_check),
    url(r'^api/inference/', InferenceView.as_view(), name='inference'),
    url(r'^api/stats', StatsView.as_view(), name='stats')

]
