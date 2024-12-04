from django.db import models

class MemeTemplate(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='memes/')
    keywords = models.CharField(max_length=255)  # store comma-separated keywords

    def __str__(self):
        return self.name

from django.db import models

class MemeImage(models.Model):
    image_file = models.ImageField(upload_to='memes/')  # Adjust the upload_to parameter as needed
    # Add other fields as necessary

    def __str__(self):
        return self.image_file.name
