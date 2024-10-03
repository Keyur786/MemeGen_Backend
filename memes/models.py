from django.db import models

class MemeTemplate(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='memes/')
    keywords = models.CharField(max_length=255)  # store comma-separated keywords

    def __str__(self):
        return self.name
