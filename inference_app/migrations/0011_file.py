# Generated by Django 2.2.11 on 2020-03-28 17:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inference_app', '0010_delete_file'),
    ]

    operations = [
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='')),
                ('date', models.DateTimeField()),
            ],
        ),
    ]
