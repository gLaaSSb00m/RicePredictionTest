# Rice Classification System

A comprehensive rice variety classification system with both web and mobile applications. Uses a VGG16 deep learning model to classify 62 different types of rice with high accuracy.

## 🚀 Features

### Web Application
- **Image Upload**: Users can upload rice grain images for classification
- **Real-time Prediction**: Instant classification results using VGG16 model
- **Confidence Scores**: Displays prediction confidence for each rice type
- **62 Rice Types**: Supports classification of 62 different rice varieties
- **Responsive Design**: Works on desktop and mobile devices
- **REST API**: Provides API endpoints for mobile app integration

### Mobile Application (Flutter)
- **Camera Integration**: Take photos directly from camera
- **Gallery Access**: Select images from device gallery
- **Real-time Prediction**: Instant classification using backend API
- **Cross-platform**: Works on both Android and iOS
- **Offline Support**: Basic functionality works without internet
- **Modern UI**: Material Design 3 with intuitive interface

## Installation

### Web Application

1. Clone the repository:
```bash
git clone <repository-url>
cd Rice_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py migrate
```

4. Start the development server:
```bash
python manage.py runserver
```

### Mobile Application

1. Navigate to the mobile app directory:
```bash
cd rice_prediction_mobile
```

2. Get Flutter dependencies:
```bash
flutter pub get
```

3. Run the app on an emulator or device:
```bash
flutter run
```

## Usage

### Web

1. Open your browser and navigate to `http://localhost:8000`
2. Click on "Predict Rice Type"
3. Upload a rice grain image
4. View the classification results with confidence scores

### Mobile

1. Launch the app on your device or emulator
2. Use the camera or gallery to select a rice grain image
3. Tap "Predict Rice Variety"
4. View the prediction results and rice information

## Model Information

- **Architecture**: VGG16
- **Training Dataset**: 62 rice varieties
- **Accuracy**: High accuracy on test dataset
- **Model File**: `models/best_VGG16_stage2.weights.h5`

## Supported Rice Types

The application can classify the following rice varieties:
- 1_Subol_Lota
- 2_Bashmoti
- 3_Ganjiya
- 4_Shampakatari
- 5_Katarivog
- 6_BR28
- 7_BR29
- 8_Paijam
- 9_Bashful
- 10_Lal_Aush
- 11_Jirashail
- 12_Gutisharna
- 13_Red_Cargo
- 14_Najirshail
- 15_Katari_Polao
- 16_Lal_Biroi
- 17_Chinigura_Polao
- 18_Amondhan
- 19_Shorna5
- 20_Lal_Binni
- 21_Arborio
- 22_Turkish_Basmati
- 23_Ipsala
- 24_Jasmine
- 25_Karacadag
- 26_BD30
- 27_BD33
- 28_BD39
- 29_BD49
- 30_BD51
- 31_BD52
- 32_BD56
- 33_BD57
- 34_BD70
- 35_BD72
- 36_BD75
- 37_BD76
- 38_BD79
- 39_BD85
- 40_BD87
- 41_BD91
- 42_BD93
- 43_BD95
- 44_Binadhan7
- 45_Binadhan8
- 46_Binadhan10
- 47_Binadhan11
- 48_Binadhan12
- 49_Binadhan14
- 50_Binadhan16
- 51_Binadhan17
- 52_Binadhan19
- 53_Binadhan21
- 54_Binadhan23
- 55_Binadhan24
- 56_Binadhan25
- 57_Binadhan26
- 58_BR22
- 59_BR23
- 60_BRRI67
- 61_BRRI74
- 62_BRRI102

## Technical Stack

- **Backend**: Django 5.x, Django REST Framework
- **Frontend**: HTML5, CSS3, JavaScript
- **Mobile**: Flutter (Dart)
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: PIL/Pillow
- **Database**: SQLite (default)

## Project Structure

```
Rice_Prediction/
├── rice_prediction/          # Django project settings
├── prediction/               # Main app for rice classification
│   ├── views.py             # Prediction logic (web)
│   ├── api_views.py         # API views for mobile app
│   ├── serializers.py       # DRF serializers
│   ├── models.py            # Database models
│   └── urls.py              # URL configurations
├── templates/               # HTML templates
├── static/                  # CSS, JS, images
├── rice_prediction_mobile/  # Flutter mobile app
│   ├── lib/                 # Flutter source code
│   ├── android/             # Android platform code
│   └── ios/                 # iOS platform code
├── requirements.txt         # Python dependencies
├── manage.py                # Django management script
└── README.md
```

## Development

### Running Tests
```bash
python manage.py test prediction
```

### Adding New Rice Types
To add support for new rice types, you'll need to:
1. Retrain the model with new data
2. Update the class mapping in `views.py` and `api_views.py`
3. Update the frontend and mobile app to display new types

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
