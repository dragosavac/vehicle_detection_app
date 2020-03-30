from django.contrib import admin
from .models import InferenceInstance, InferenceModel, Camera

# Register your models here.
admin.site.register(InferenceModel)
admin.site.register(InferenceInstance)
admin.site.register(Camera)








