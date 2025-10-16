from django.db import models

class RiceInfo(models.Model):
    """Model to store information about different rice varieties."""

    variety_name = models.CharField(
        max_length=100,
        unique=True,
        help_text="Name of the rice variety"
    )

    info = models.TextField(
        default="info isn't available",
        help_text="Detailed information about this rice variety"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Rice Information"
        verbose_name_plural = "Rice Information"
        ordering = ['variety_name']

    def __str__(self):
        return f"{self.variety_name}"

class RiceModel(models.Model):
    """Model to store machine learning models for rice prediction."""

    name = models.CharField(
        max_length=100,
        unique=True,
        help_text="Name of the model"
    )

    # For VGG16
    model_file = models.FileField(
        upload_to='models/',
        null=True,
        blank=True,
        help_text="Path to the .h5 model file (for VGG16)"
    )

    tflite_file = models.FileField(
        upload_to='models/',
        null=True,
        blank=True,
        help_text="Path to the .tflite model file (for VGG16)"
    )



    is_active = models.BooleanField(
        default=False,
        help_text="Whether this model is currently active"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Rice Model"
        verbose_name_plural = "Rice Models"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({'Active' if self.is_active else 'Inactive'})"
