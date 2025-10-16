from django.contrib import admin
from .models import RiceInfo, RiceModel

@admin.register(RiceInfo)
class RiceInfoAdmin(admin.ModelAdmin):
    list_display = ['variety_name', 'created_at', 'updated_at']
    search_fields = ['variety_name']
    list_filter = ['created_at']
    ordering = ['variety_name']

@admin.register(RiceModel)
class RiceModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active', 'created_at', 'updated_at']
    search_fields = ['name']
    list_filter = ['is_active', 'created_at']
    ordering = ['-created_at']

    def get_fieldsets(self, request, obj=None):
        if obj and 'VGG16' in obj.name:
            return [
                ('Basic Information', {'fields': ['name']}),
                ('VGG16 Files', {'fields': ['model_file', 'tflite_file']}),
                ('Status', {'fields': ['is_active']}),
            ]
        elif obj and 'Ensemble' in obj.name:
            return [
                ('Basic Information', {'fields': ['name']}),
                ('Ensemble Files', {'fields': ['vgg_weights_file', 'mobilenet_weights_file', 'xgb_model_file']}),
                ('Status', {'fields': ['is_active']}),
            ]
        else:
            return [
                ('Basic Information', {'fields': ['name']}),
                ('VGG16 Files', {'fields': ['model_file', 'tflite_file']}),
                ('Ensemble Files', {'fields': ['vgg_weights_file', 'mobilenet_weights_file', 'xgb_model_file']}),
                ('Status', {'fields': ['is_active']}),
            ]
