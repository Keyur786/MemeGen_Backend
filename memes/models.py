from django.db import models

class MemeImage(models.Model):
    image_file = models.ImageField(upload_to='memes/')  # Adjust the upload_to parameter as needed
    # Add other fields as necessary

    def __str__(self):
        return self.image_file.name
