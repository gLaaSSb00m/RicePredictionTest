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
    list_display = ['name', 'model_type', 'is_active', 'created_at', 'updated_at']
    search_fields = ['name']
    list_filter = ['model_type', 'is_active', 'created_at']
    ordering = ['-created_at']

    def get_fieldsets(self, request, obj=None):
        if obj and obj.model_type == 'vgg':
            return [
                ('Basic Information', {'fields': ['name', 'model_type']}),
                ('VGG16 Files', {'fields': ['model_file', 'tflite_file']}),
                ('Status', {'fields': ['is_active']}),
            ]
        elif obj and obj.model_type == 'ensemble':
            return [
                ('Basic Information', {'fields': ['name', 'model_type']}),
                ('Ensemble Files', {'fields': ['vgg_weights_file', 'mobilenet_weights_file', 'xgb_model_file']}),
                ('Status', {'fields': ['is_active']}),
            ]
        else:
            # Default for add form
            return [
                ('Basic Information', {'fields': ['name', 'model_type']}),
                ('VGG16 Files', {'fields': ['model_file', 'tflite_file']}),
                ('Ensemble Files', {'fields': ['vgg_weights_file', 'mobilenet_weights_file', 'xgb_model_file']}),
                ('Status', {'fields': ['is_active']}),
            ]
