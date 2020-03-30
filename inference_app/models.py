from django.db import models

# Create your models here.


class Camera(models.Model):
    name = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.name


class InferenceModel(models.Model):
    path = models.FileField(upload_to='post_image', blank=True)  #upload through admin
    version = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.version


class InferenceInstance(models.Model):
    calculation_model = models.ForeignKey('InferenceModel', on_delete=models.CASCADE, null=True)
    calculation_datetime = models.DateTimeField(null=False)
    value = models.CharField(max_length=10000, null=False, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "{}".format(self.calculation_datetime, self.value)
