# Generated by Django 2.2.11 on 2020-03-28 11:46

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('inference_app', '0005_auto_20200328_0959'),
    ]

    operations = [
        migrations.AlterField(
            model_name='file',
            name='date',
            field=models.DateTimeField(default=datetime.datetime(2020, 3, 28, 11, 46, 4, 420038, tzinfo=utc)),
        ),
    ]
