# Generated by Django 2.2.11 on 2020-03-28 17:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inference_app', '0011_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='file',
            name='calculation',
            field=models.CharField(default={}, max_length=120),
            preserve_default=False,
        ),
    ]
