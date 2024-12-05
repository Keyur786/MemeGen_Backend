from django.contrib import admin
from .models import MemeImage



class MemeImageAdmin(admin.ModelAdmin):
    list_display = ['image_file']

admin.site.register(MemeImage, MemeImageAdmin)

