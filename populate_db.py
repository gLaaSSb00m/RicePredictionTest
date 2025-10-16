import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rice_prediction.settings')
django.setup()

from prediction.models import RiceInfo, RiceModel

# List of rice varieties provided by user
rice_varieties = sorted([
    "10_Lal_Aush",
    "11_Jirashail",
    "12_Gutisharna",
    "13_Red_Cargo",
    "14_Najirshail",
    "15_Katari_Polao",
    "16_Lal_Biroi",
    "17_Chinigura_Polao",
    "18_Amon",
    "19_Shorna5",
    "1_Subol_Lota",
    "20_Lal_Binni",
    "21_Arborio",
    "22_Turkish_Basmati",
    "23_Ipsala",
    "24_Jasmine",
    "25_Karacadag",
    "26_BD30",
    "27_BD33",
    "28_BD39",
    "29_BD49",
    "2_Bashmoti",
    "30_BD51",
    "31_BD52",
    "32_BD56",
    "33_BD57",
    "34_BD70",
    "35_BD72",
    "36_BD75",
    "37_BD76",
    "38_BD79",
    "39_BD85",
    "3_Ganjiya",
    "40_BD87",
    "41_BD91",
    "42_BD93",
    "43_BD95",
    "44_Binadhan7",
    "45_Binadhan8",
    "46_Binadhan10",
    "47_Binadhan11",
    "48_Binadhan12",
    "49_Binadhan14",
    "4_Shampakatari",
    "50_Binadhan16",
    "51_Binadhan17",
    "52_Binadhan19",
    "53_Binadhan21",
    "54_Binadhan23",
    "55_Binadhan24",
    "56_Binadhan25",
    "57_Binadhan26",
    "58_BR22",
    "59_BR23",
    "5_Katarivog",
    "60_BRRI67",
    "61_BRRI74",
    "62_BRRI102",
    "6_BR28",
    "7_BR29",
    "8_Paijam",
    "9_Bashful"
])

def populate_rice_info():
    # Delete all existing RiceInfo entries
    RiceInfo.objects.all().delete()
    print("Deleted all existing rice varieties.")

    for variety in rice_varieties:
        # Use the full variety name as is
        variety_name = variety

        # Create new RiceInfo entry
        RiceInfo.objects.create(
            variety_name=variety_name,
            info=f"Information about {variety_name} rice variety."
        )
    print(f"Populated {len(rice_varieties)} rice varieties into RiceInfo.")

def populate_rice_model():
    # Update or create VGG16 model entry
    vgg_model, created = RiceModel.objects.get_or_create(
        name="VGG16 Model",
        defaults={
            'is_active': True,
            'model_file': None,
            'tflite_file': None
        }
    )

    # Update or create Ensemble model entry
    ensemble_model, created = RiceModel.objects.get_or_create(
        name="Ensemble Model",
        defaults={
            'is_active': False,
            'vgg_weights_file': None,
            'mobilenet_weights_file': None,
            'xgb_model_file': None
        }
    )

    print("Updated VGG16 and Ensemble model entries.")

if __name__ == "__main__":
    populate_rice_info()
    populate_rice_model()
    print("Database populated successfully.")
