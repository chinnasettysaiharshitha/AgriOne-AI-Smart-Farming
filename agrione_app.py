# ============================================================
# AGRIONE AI
# MULTILINGUAL SMART FARMING ASSISTANT
# ============================================================

import os
import io
import json
import tempfile

import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st

from PIL import Image
from tensorflow.keras.models import load_model

# Voice
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# Ollama
import ollama


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AgriOne AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AgriOne AI")
st.caption("Multilingual Smart Farming Assistant")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "models")

CROP_MODEL_PATH = os.path.join(
    BASE_DIR,
    "crop_recommendation_model.pkl"
)

FERTILIZER_MODEL_PATH = os.path.join(
    BASE_DIR,
    "fertilizer_recommendation_model.pkl"
)

SOIL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "soil_nutrient_model.pkl"
)

SOIL_ENCODERS_PATH = os.path.join(
    BASE_DIR,
    "soil_nutrient_encoders.pkl"
)

PLANT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model.keras"
)

PLANT_CLASSES_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_classes.json"
)


# ============================================================
# FFMPEG
# ============================================================

FFMPEG_BIN = (
    r"C:\Users\vijay\Downloads"
    r"\ffmpeg-9.0.2-essentials_build"
    r"\ffmpeg-9.0.2-essentials_build"
    r"\bin"
)

FFMPEG_EXE = os.path.join(
    FFMPEG_BIN,
    "ffmpeg.exe"
)

FFPROBE_EXE = os.path.join(
    FFMPEG_BIN,
    "ffprobe.exe"
)

if os.path.exists(FFMPEG_EXE):

    AudioSegment.converter = FFMPEG_EXE

if os.path.exists(FFPROBE_EXE):

    AudioSegment.ffprobe = FFPROBE_EXE


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {

    "English": {
        "speech": "en-IN",
        "tts": "en"
    },

    "Telugu": {
        "speech": "te-IN",
        "tts": "te"
    },

    "Hindi": {
        "speech": "hi-IN",
        "tts": "hi"
    },

    "Tamil": {
        "speech": "ta-IN",
        "tts": "ta"
    }
}


# ============================================================
# LOAD CROP MODEL
# ============================================================

@st.cache_resource
def load_crop_model():

    if not os.path.exists(CROP_MODEL_PATH):
        return None

    return joblib.load(CROP_MODEL_PATH)


crop_model = load_crop_model()


# ============================================================
# LOAD FERTILIZER MODEL
# ============================================================

@st.cache_resource
def load_fertilizer_model():

    if not os.path.exists(FERTILIZER_MODEL_PATH):
        return None

    return joblib.load(FERTILIZER_MODEL_PATH)


fertilizer_bundle = load_fertilizer_model()


# ============================================================
# LOAD SOIL MODEL
# ============================================================

@st.cache_resource
def load_soil_models():

    if not os.path.exists(SOIL_MODEL_PATH):
        return None, None

    if not os.path.exists(SOIL_ENCODERS_PATH):
        return None, None

    model = joblib.load(
        SOIL_MODEL_PATH
    )

    encoders = joblib.load(
        SOIL_ENCODERS_PATH
    )

    return model, encoders


soil_model, soil_encoders = load_soil_models()


# ============================================================
# LOAD PLANT DISEASE MODEL
# ============================================================

@st.cache_resource
def load_plant_model():

    if not os.path.exists(PLANT_MODEL_PATH):
        return None

    return load_model(
        PLANT_MODEL_PATH,
        compile=False
    )


plant_model = load_plant_model()


# ============================================================
# LOAD PLANT DISEASE CLASSES
# ============================================================

@st.cache_resource
def load_plant_classes():

    if not os.path.exists(
        PLANT_CLASSES_PATH
    ):
        return None

    with open(
        PLANT_CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


plant_classes = load_plant_classes()


# ============================================================
# WEATHER DESCRIPTION
# ============================================================

def weather_description(code):

    descriptions = {

        0: "Clear sky",

        1: "Mainly clear",

        2: "Partly cloudy",

        3: "Overcast",

        45: "Fog",

        48: "Depositing rime fog",

        51: "Light drizzle",

        53: "Moderate drizzle",

        55: "Dense drizzle",

        61: "Slight rain",

        63: "Moderate rain",

        65: "Heavy rain",

        71: "Slight snowfall",

        73: "Moderate snowfall",

        75: "Heavy snowfall",

        80: "Slight rain showers",

        81: "Moderate rain showers",

        82: "Violent rain showers",

        95: "Thunderstorm",

        96: "Thunderstorm with hail",

        99: "Thunderstorm with heavy hail"
    }

    return descriptions.get(
        code,
        "Unknown weather"
    )


# ============================================================
# LOCATION SEARCH
# ============================================================

def search_location(location):

    try:

        url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        params = {

            "name": location.strip(),

            "count": 10,

            "language": "en",

            "format": "json",

            "countryCode": "IN"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return []

        data = response.json()

        results = data.get(
            "results",
            []
        )

        locations = []

        for item in results:

            name = item.get(
                "name",
                ""
            )

            district = item.get(
                "admin2",
                ""
            )

            state = item.get(
                "admin1",
                ""
            )

            display_name = ", ".join(
                x
                for x in
                [name, district, state]
                if x
            )

            locations.append({

                "display":
                    display_name,

                "latitude":
                    item.get("latitude"),

                "longitude":
                    item.get("longitude")
            })

        return locations

    except Exception:

        return []


# ============================================================
# WEATHER FUNCTION
# ============================================================

def get_weather(
    latitude,
    longitude
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current":
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "weather_code,"
            "wind_speed_10m",

        "daily":
            "weather_code,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum",

        "forecast_days": 7,

        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# CROP PREDICTION
# ============================================================

def predict_crop(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):

    if crop_model is None:
        return None

    values = np.array([[
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    ]])

    prediction = crop_model.predict(
        values
    )

    return prediction[0]


# ============================================================
# FERTILIZER OPTIONS
# ============================================================

def get_fertilizer_options():

    if fertilizer_bundle is None:
        return [], []

    expected_columns = (
        fertilizer_bundle["columns"]
    )

    soil_types = []

    crop_types = []

    for column in expected_columns:

        if column.startswith(
            "Soil Type_"
        ):

            soil_types.append(
                column.replace(
                    "Soil Type_",
                    ""
                )
            )

        if column.startswith(
            "Crop Type_"
        ):

            crop_types.append(
                column.replace(
                    "Crop Type_",
                    ""
                )
            )

    return (
        sorted(soil_types),
        sorted(crop_types)
    )


# ============================================================
# FERTILIZER PREDICTION
# ============================================================

def predict_fertilizer(
    temperature,
    humidity,
    moisture,
    soil_type,
    crop_type,
    nitrogen,
    potassium,
    phosphorous
):

    if fertilizer_bundle is None:
        return None

    model = fertilizer_bundle["model"]

    expected_columns = (
        fertilizer_bundle["columns"]
    )

    data = pd.DataFrame([{

        "Temparature":
            temperature,

        "Humidity ":
            humidity,

        "Moisture":
            moisture,

        "Soil Type":
            soil_type,

        "Crop Type":
            crop_type,

        "Nitrogen":
            nitrogen,

        "Potassium":
            potassium,

        "Phosphorous":
            phosphorous
    }])

    data = pd.get_dummies(
        data,
        columns=[
            "Soil Type",
            "Crop Type"
        ]
    )

    data = data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    prediction = model.predict(
        data
    )

    return prediction[0]


# ============================================================
# SOIL NUTRIENT PREDICTION
# ============================================================

def predict_soil_nutrient(
    state,
    district,
    nutrient,
    value
):

    if (
        soil_model is None
        or soil_encoders is None
    ):
        return None

    try:

        state_encoder = (
            soil_encoders["state_name"]
        )

        district_encoder = (
            soil_encoders["district_name"]
        )

        nutrient_encoder = (
            soil_encoders["nutrient_name"]
        )

        level_encoder = (
            soil_encoders["nutrient_level"]
        )

        state_encoded = (
            state_encoder
            .transform([state])[0]
        )

        district_encoded = (
            district_encoder
            .transform([district])[0]
        )

        nutrient_encoded = (
            nutrient_encoder
            .transform([nutrient])[0]
        )

        input_data = np.array([[
            state_encoded,
            district_encoded,
            nutrient_encoded,
            value
        ]])

        prediction = soil_model.predict(
            input_data
        )

        result = (
            level_encoder
            .inverse_transform(
                prediction
            )[0]
        )

        return result

    except Exception:

        return None


# ============================================================
# PLANT DISEASE PREDICTION
# ============================================================

def predict_plant_disease(
    image
):

    if (
        plant_model is None
        or plant_classes is None
    ):
        return None, None

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (160, 160)
    )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = plant_model.predict(
        image_array,
        verbose=0
    )[0]

    index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[index]
    ) * 100

    if isinstance(
        plant_classes,
        dict
    ):

        if str(index) in plant_classes:

            disease = plant_classes[
                str(index)
            ]

        elif index in plant_classes:

            disease = plant_classes[
                index
            ]

        else:

            disease = (
                f"Class {index}"
            )

    elif isinstance(
        plant_classes,
        list
    ):

        if index < len(
            plant_classes
        ):

            disease = plant_classes[
                index
            ]

        else:

            disease = (
                f"Class {index}"
            )

    else:

        disease = (
            f"Class {index}"
        )

    return (
        disease,
        confidence
    )


# ============================================================
# OLLAMA AI
# ============================================================

def ask_agri_ai(
    question,
    language="English",
    extra_context=""
):

    system_prompt = f"""

You are AgriOne AI, a multilingual smart farming
assistant for farmers.

Answer completely in {language}. This includes every heading, bullet point, explanation, warning, and recommendation. Do not answer in English unless the selected language is English. Keep crop/disease scientific names and official scheme names unchanged when translation would make them unclear.

You help farmers with:

- crop selection
- soil
- nutrients
- fertilizer
- irrigation
- plant diseases
- pests
- weather
- government schemes
- market prices
- general farming practices

Important rules:

1. Use simple farmer-friendly language.
2. Give practical suggestions.
3. Do not invent facts.
4. Do not claim 100 percent certainty.
5. Do not provide dangerous or unsupported advice.
6. Do not give exact fertilizer doses unless they are
   provided by a verified soil recommendation.
7. Encourage soil testing when necessary.
8. For plant disease predictions, explain that AI prediction
   is not a guaranteed diagnosis.
9. For market prices, tell the user to verify current prices
   through official market sources.
10. If important information is missing, say what information
    is needed.
11. Keep the answer reasonably short and easy to understand.

Additional information:

{extra_context}
"""

    try:

        response = ollama.chat(

            model="llama3.2:3b",

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return (
            "❌ AI assistant could not connect to Ollama.\n\n"
            f"Error: {e}\n\n"
            "Please make sure Ollama is running."
        )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak_answer(
    text,
    language
):

    try:

        lang_code = LANGUAGES[
            language
        ]["tts"]

        tts = gTTS(
            text=text,
            lang=lang_code
        )

        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        audio_file.close()

        tts.save(
            audio_file.name
        )

        return audio_file.name

    except Exception:

        return None


# ============================================================
# OFFLINE MULTILINGUAL UI
# ============================================================

UI_TRANSLATIONS = {
    "Telugu": {
        "🌱 Select Feature":"🌱 ఫీచర్‌ను ఎంచుకోండి", "🌐 Language":"🌐 భాష", "Home":"హోమ్",
        "Plant Disease Detection":"మొక్కల వ్యాధి గుర్తింపు", "Crop Recommendation":"పంట సిఫార్సు",
        "Fertilizer Recommendation":"ఎరువుల సిఫార్సు", "Soil Nutrient Analysis":"నేల పోషకాల విశ్లేషణ",
        "🌦️ Weather Information":"🌦️ వాతావరణ సమాచారం", "Weather Information":"వాతావరణ సమాచారం",
        "Government Schemes":"ప్రభుత్వ పథకాలు", "Market Prices":"మార్కెట్ ధరలు",
        "AI Agriculture Assistant":"AI వ్యవసాయ సహాయకుడు", "Voice Assistant":"వాయిస్ సహాయకుడు",
        "🌱 Welcome to AgriOne AI":"🌱 AgriOne AI కి స్వాగతం", "Multilingual Smart Farming Assistant":"బహుభాషా స్మార్ట్ వ్యవసాయ సహాయకుడు",
        "Select Feature":"ఫీచర్‌ను ఎంచుకోండి", "Enter village / town / district":"గ్రామం / పట్టణం / జిల్లాను నమోదు చేయండి",
        "Select location":"ప్రాంతాన్ని ఎంచుకోండి", "Submit":"సమర్పించండి", "Predict":"అంచనా వేయండి",
        "Analyze":"విశ్లేషించండి", "Search":"వెతకండి", "Ask":"అడగండి", "Clear":"తొలగించండి",
        "Download Report":"నివేదికను డౌన్‌లోడ్ చేయండి", "📍 Enter location":"📍 ప్రాంతాన్ని నమోదు చేయండి",
        "🌦️ Get Weather":"🌦️ వాతావరణాన్ని పొందండి", "🤖 Ask AgriOne AI":"🤖 AgriOne AI ని అడగండి",
        "🤖 Ask AI":"🤖 AI ని అడగండి", "🌾 Official Market Price Sources":"🌾 అధికారిక మార్కెట్ ధరల వనరులు",
        "📈 Agricultural Market Prices":"📈 వ్యవసాయ మార్కెట్ ధరలు", "Current language":"ప్రస్తుత భాష",
        "Model Confidence":"మోడల్ విశ్వాస స్థాయి", "Recommended Fertilizer":"సిఫార్సు చేసిన ఎరువు",
        "AI Disease Explanation":"AI వ్యాధి వివరణ", "AI Fertilizer Explanation":"AI ఎరువు వివరణ",
        "AI Soil Suggestions":"AI నేల సూచనలు", "AI Farming Suggestions":"AI వ్యవసాయ సూచనలు",
        "🌾 Main Features":"🌾 ప్రధాన ఫీచర్లు", "🌐 Supported Languages":"🌐 మద్దతు ఉన్న భాషలు",
        "English":"ఆంగ్లం", "Telugu":"తెలుగు", "Hindi":"హిందీ", "Tamil":"తమిళం",
        "What should the farmer do now":"రైతు ఇప్పుడు ఏమి చేయాలి", "Please enter a question":"దయచేసి ప్రశ్న నమోదు చేయండి",
    },
    "Hindi": {
        "🌱 Select Feature":"🌱 सुविधा चुनें", "🌐 Language":"🌐 भाषा", "Home":"होम",
        "Plant Disease Detection":"पौधों की बीमारी की पहचान", "Crop Recommendation":"फसल की सिफारिश",
        "Fertilizer Recommendation":"उर्वरक की सिफारिश", "Soil Nutrient Analysis":"मिट्टी के पोषक तत्वों का विश्लेषण",
        "🌦️ Weather Information":"🌦️ मौसम की जानकारी", "Weather Information":"मौसम की जानकारी",
        "Government Schemes":"सरकारी योजनाएँ", "Market Prices":"बाज़ार की कीमतें",
        "AI Agriculture Assistant":"AI कृषि सहायक", "Voice Assistant":"वॉयस सहायक",
        "🌱 Welcome to AgriOne AI":"🌱 AgriOne AI में आपका स्वागत है", "Multilingual Smart Farming Assistant":"बहुभाषी स्मार्ट कृषि सहायक",
        "Select Feature":"सुविधा चुनें", "Enter village / town / district":"गाँव / शहर / ज़िला दर्ज करें",
        "Select location":"स्थान चुनें", "Submit":"जमा करें", "Predict":"अनुमान लगाएँ",
        "Analyze":"विश्लेषण करें", "Search":"खोजें", "Ask":"पूछें", "Clear":"साफ़ करें",
        "Download Report":"रिपोर्ट डाउनलोड करें", "📍 Enter location":"📍 स्थान दर्ज करें",
        "🌦️ Get Weather":"🌦️ मौसम प्राप्त करें", "🤖 Ask AgriOne AI":"🤖 AgriOne AI से पूछें",
        "🤖 Ask AI":"🤖 AI से पूछें", "🌾 Official Market Price Sources":"🌾 आधिकारिक बाज़ार मूल्य स्रोत",
        "📈 Agricultural Market Prices":"📈 कृषि बाज़ार कीमतें", "Current language":"वर्तमान भाषा",
        "Model Confidence":"मॉडल का विश्वास स्तर", "Recommended Fertilizer":"अनुशंसित उर्वरक",
        "AI Disease Explanation":"AI रोग विवरण", "AI Fertilizer Explanation":"AI उर्वरक विवरण",
        "AI Soil Suggestions":"AI मिट्टी सुझाव", "AI Farming Suggestions":"AI खेती सुझाव",
        "🌾 Main Features":"🌾 मुख्य सुविधाएँ", "🌐 Supported Languages":"🌐 समर्थित भाषाएँ",
        "English":"अंग्रेज़ी", "Telugu":"तेलुगु", "Hindi":"हिंदी", "Tamil":"तमिल",
        "Please enter a question":"कृपया प्रश्न दर्ज करें",
    },
    "Tamil": {
        "🌱 Select Feature":"🌱 அம்சத்தைத் தேர்ந்தெடுக்கவும்", "🌐 Language":"🌐 மொழி", "Home":"முகப்பு",
        "Plant Disease Detection":"தாவர நோய் கண்டறிதல்", "Crop Recommendation":"பயிர் பரிந்துரை",
        "Fertilizer Recommendation":"உரப் பரிந்துரை", "Soil Nutrient Analysis":"மண் ஊட்டச்சத்து பகுப்பாய்வு",
        "🌦️ Weather Information":"🌦️ வானிலை தகவல்", "Weather Information":"வானிலை தகவல்",
        "Government Schemes":"அரசுத் திட்டங்கள்", "Market Prices":"சந்தை விலைகள்",
        "AI Agriculture Assistant":"AI வேளாண்மை உதவியாளர்", "Voice Assistant":"குரல் உதவியாளர்",
        "🌱 Welcome to AgriOne AI":"🌱 AgriOne AI-க்கு வரவேற்கிறோம்", "Multilingual Smart Farming Assistant":"பலமொழி ஸ்மார்ட் விவசாய உதவியாளர்",
        "Select Feature":"அம்சத்தைத் தேர்ந்தெடுக்கவும்", "Enter village / town / district":"கிராமம் / நகரம் / மாவட்டத்தை உள்ளிடவும்",
        "Select location":"இடத்தைத் தேர்ந்தெடுக்கவும்", "Submit":"சமர்ப்பிக்கவும்", "Predict":"கணிக்கவும்",
        "Analyze":"பகுப்பாய்வு செய்யவும்", "Search":"தேடவும்", "Ask":"கேட்கவும்", "Clear":"அழிக்கவும்",
        "Download Report":"அறிக்கையைப் பதிவிறக்கவும்", "📍 Enter location":"📍 இடத்தை உள்ளிடவும்",
        "🌦️ Get Weather":"🌦️ வானிலையைப் பெறவும்", "🤖 Ask AgriOne AI":"🤖 AgriOne AI-யிடம் கேட்கவும்",
        "🤖 Ask AI":"🤖 AI-யிடம் கேட்கவும்", "🌾 Official Market Price Sources":"🌾 அதிகாரப்பூர்வ சந்தை விலை ஆதாரங்கள்",
        "📈 Agricultural Market Prices":"📈 வேளாண் சந்தை விலைகள்", "Current language":"தற்போதைய மொழி",
        "Model Confidence":"மாதிரி நம்பிக்கை நிலை", "Recommended Fertilizer":"பரிந்துரைக்கப்பட்ட உரம்",
        "AI Disease Explanation":"AI நோய் விளக்கம்", "AI Fertilizer Explanation":"AI உர விளக்கம்",
        "AI Soil Suggestions":"AI மண் பரிந்துரைகள்", "AI Farming Suggestions":"AI விவசாய பரிந்துரைகள்",
        "🌾 Main Features":"🌾 முக்கிய அம்சங்கள்", "🌐 Supported Languages":"🌐 ஆதரிக்கப்படும் மொழிகள்",
        "English":"ஆங்கிலம்", "Telugu":"தெலுங்கு", "Hindi":"இந்தி", "Tamil":"தமிழ்",
        "Please enter a question":"தயவுசெய்து கேள்வியை உள்ளிடவும்",
    }
}

def tr(text):
    if not isinstance(text, str):
        return text
    language = st.session_state.get("global_language", "English")
    translations = UI_TRANSLATIONS.get(language, {})
    if text in translations:
        return translations[text]
    # Translate known labels inside longer Markdown or messages.
    result = text
    for source_text, translated_text in sorted(translations.items(), key=lambda item: len(item[0]), reverse=True):
        if source_text and source_text in result:
            result = result.replace(source_text, translated_text)
    return result

# Wrap common Streamlit UI methods to translate headings, messages and labels locally.
# This uses only the dictionary above; it does not call Google or another translation API.
def _wrap_streamlit_method(obj, name):
    original = getattr(obj, name, None)
    if original is None or getattr(original, "_agrione_tr_wrapped", False):
        return
    def wrapped(*args, **kwargs):
        args = list(args)
        if args and isinstance(args[0], str):
            args[0] = tr(args[0])
        for key in ("label", "placeholder", "help", "caption"):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = tr(kwargs[key])
        # Preserve internal English values used by if/elif logic; translate display only.
        if len(args) > 1 and isinstance(args[1], (list, tuple)) and all(isinstance(item, str) for item in args[1]):
            kwargs.setdefault("format_func", tr)
        return original(*args, **kwargs)
    wrapped._agrione_tr_wrapped = True
    setattr(obj, name, wrapped)

for _name in ("title", "header", "subheader", "caption", "write", "markdown", "text", "info", "success", "warning", "error", "button", "download_button", "checkbox", "toggle", "radio", "selectbox", "multiselect", "slider", "text_input", "text_area", "number_input", "file_uploader", "camera_input", "form_submit_button", "metric", "link_button"):
    _wrap_streamlit_method(st, _name)
    _wrap_streamlit_method(st.sidebar, _name)

# Language selection is now global, so every module uses the same selection.
selected_language = st.sidebar.selectbox(
    "🌐 Language",
    list(LANGUAGES.keys()),
    key="global_language",
    format_func=tr,
)
st.title(tr("🌱 AgriOne AI"))
st.caption(tr("Multilingual Smart Farming Assistant"))

menu = st.sidebar.radio(
    "🌱 Select Feature",
    [
        "Home", "Plant Disease Detection", "Crop Recommendation",
        "Fertilizer Recommendation", "Soil Nutrient Analysis",
        "🌦️ Weather Information", "Government Schemes", "Market Prices",
        "AI Agriculture Assistant", "Voice Assistant"
    ],
    key="feature_menu",
    format_func=tr,
)


# ============================================================
# HOME
# ============================================================

if menu == "Home":

    st.header(
        "🌱 Welcome to AgriOne AI"
    )

    st.write(
        "A multilingual smart farming assistant "
        "designed to support farmers using AI, "
        "machine learning and weather information."
    )

    st.markdown(
        """
### 🌾 Main Features

- 🌿 Plant Disease Detection
- 🌾 Crop Recommendation
- 🧪 Fertilizer Recommendation
- 🧪 Soil Nutrient Analysis
- 🌦️ Weather Information
- 🏛️ Government Schemes
- 📈 Market Price Information
- 🤖 AI Agriculture Assistant
- 🎤 Voice Assistant

### 🌐 Supported Languages

- English
- Telugu
- Hindi
- Tamil
"""
    )


# ============================================================
# PLANT DISEASE DETECTION
# ============================================================

elif menu == "Plant Disease Detection":

    st.header(
        "🌿 Plant Disease Detection"
    )

    st.write(
        "Upload a plant leaf image to detect a possible "
        "plant disease."
    )

    uploaded_image = st.file_uploader(

        "📷 Upload Plant Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_image:

        image = Image.open(
            uploaded_image
        )

        st.image(
            image,
            caption="Uploaded Plant Image",
            width=400
        )

        if st.button(
            "🔍 Detect Disease",
            key="detect_disease"
        ):

            with st.spinner(
                "🌿 Analyzing plant image..."
            ):

                disease, confidence = (
                    predict_plant_disease(
                        image
                    )
                )

            if disease:

                st.success(
                    f"🌿 Predicted Disease: "
                    f"**{disease}**"
                )

                st.info(
                    f"Model Confidence: "
                    f"**{confidence:.2f}%**"
                )

                # AI explanation

                context = f"""

Plant Disease Model Result:

Predicted Disease:
{disease}

Model Confidence:
{confidence:.2f}%
"""

                question = f"""

The plant disease model predicted:

{disease}

with a confidence of
{confidence:.2f}%.

Explain this result to a farmer.

Explain:

1. What the disease means
2. Common symptoms
3. Possible causes
4. How it may spread
5. What the farmer should do now
6. Prevention methods
7. What to monitor

Do not claim this is a guaranteed diagnosis.

Do not give unsupported pesticide doses.
"""

                st.subheader(
                    "🤖 AI Disease Explanation"
                )

                with st.spinner(
                    "🤖 Preparing explanation..."
                ):

                    ai_answer = ask_agri_ai(
                        question,
                        selected_language,
                        context
                    )

                st.write(
                    ai_answer
                )

                st.warning(
                    "⚠️ This is an AI-based image prediction "
                    "and is not a guaranteed plant disease "
                    "diagnosis. Consult an agricultural expert "
                    "for confirmation."
                )


# ============================================================
# CROP RECOMMENDATION
# ============================================================

elif menu == "Crop Recommendation":

    st.header(
        "🌾 Crop Recommendation"
    )

    st.write(
        "Use soil-test values and automatically obtained "
        "weather information to get a crop recommendation."
    )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    st.subheader(
        "📍 Step 1: Select Location"
    )

    location = st.text_input(
        "Enter village / town / district",
        placeholder="Example: Chilakaluripet",
        key="crop_location"
    )

    selected_location = None

    if location.strip():

        locations = search_location(
            location
        )

        if locations:

            selected_name = st.selectbox(
                "Select the correct location",
                [
                    x["display"]
                    for x in locations
                ],
                key="crop_location_select"
            )

            selected_location = next(
                x for x in locations
                if x["display"] == selected_name
            )

        else:

            st.warning(
                "⚠️ Location not found."
            )

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    temperature = None
    humidity = None
    rainfall = None

    if selected_location:

        try:

            weather = get_weather(

                selected_location[
                    "latitude"
                ],

                selected_location[
                    "longitude"
                ]
            )

            current = weather[
                "current"
            ]

            temperature = current[
                "temperature_2m"
            ]

            humidity = current[
                "relative_humidity_2m"
            ]

            daily_rain = weather[
                "daily"
            ][
                "precipitation_sum"
            ]

            rainfall = sum(
                x or 0
                for x in daily_rain
            )

            st.subheader(
                "🌦️ Step 2: Automatic Weather"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "🌡️ Temperature",
                    f"{temperature} °C"
                )

            with c2:

                st.metric(
                    "💧 Humidity",
                    f"{humidity} %"
                )

            with c3:

                st.metric(
                    "🌧️ 7-Day Rainfall",
                    f"{rainfall:.1f} mm"
                )

        except Exception as e:

            st.error(
                f"❌ Weather error: {e}"
            )

    # --------------------------------------------------------
    # SOIL
    # --------------------------------------------------------

    st.subheader(
        "🧪 Step 3: Soil-Test Values"
    )

    st.info(
        "Enter N, P, K and pH from the farmer's "
        "soil-test report. Do not guess these values."
    )

    c1, c2 = st.columns(2)

    with c1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=0.0,
            key="crop_n_new"
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=0.0,
            key="crop_p_new"
        )

    with c2:

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=0.0,
            key="crop_k_new"
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=0.0,
            key="crop_ph_new"
        )

    # --------------------------------------------------------
    # RECOMMEND
    # --------------------------------------------------------

    if st.button(
        "🌾 Recommend Crop",
        key="crop_recommend_button"
    ):

        if selected_location is None:

            st.warning(
                "⚠️ Please select your location."
            )

        elif (
            nitrogen <= 0
            or phosphorus <= 0
            or potassium <= 0
            or ph <= 0
        ):

            st.warning(
                "⚠️ Please enter valid soil-test values."
            )

        elif (
            temperature is None
            or humidity is None
            or rainfall is None
        ):

            st.warning(
                "⚠️ Weather information is unavailable."
            )

        else:

            with st.spinner(
                "🌾 Finding suitable crop..."
            ):

                crop = predict_crop(

                    nitrogen,
                    phosphorus,
                    potassium,
                    temperature,
                    humidity,
                    ph,
                    rainfall
                )

            if crop:

                st.success(
                    f"🌾 Recommended Crop: **{crop}**"
                )

                st.subheader(
                    "📋 Information Used"
                )

                st.write(
                    f"📍 Location: "
                    f"{selected_location['display']}"
                )

                st.write(
                    f"🧪 N: {nitrogen}"
                )

                st.write(
                    f"🧪 P: {phosphorus}"
                )

                st.write(
                    f"🧪 K: {potassium}"
                )

                st.write(
                    f"🧪 pH: {ph}"
                )

                st.write(
                    f"🌡️ Temperature: "
                    f"{temperature} °C"
                )

                st.write(
                    f"💧 Humidity: "
                    f"{humidity} %"
                )

                st.write(
                    f"🌧️ 7-Day Rainfall: "
                    f"{rainfall:.1f} mm"
                )

                context = f"""

Recommended crop:
{crop}

Location:
{selected_location['display']}

N:
{nitrogen}

P:
{phosphorus}

K:
{potassium}

pH:
{ph}

Temperature:
{temperature} °C

Humidity:
{humidity} %

7-day rainfall:
{rainfall} mm
"""

                question = f"""

The crop recommendation model recommended:

{crop}

Give simple farming suggestions.

Explain:

1. Why it may be suitable
2. Soil requirements
3. Irrigation
4. Weather considerations
5. Crop management
6. Pest and disease monitoring
7. General fertilizer/soil advice
8. What the farmer should check before planting

Do not guarantee yield.
Do not give unsupported fertilizer doses.
"""

                st.subheader(
                    "🤖 AI Farming Suggestions"
                )

                with st.spinner(
                    "🤖 Preparing suggestions..."
                ):

                    answer = ask_agri_ai(
                        question,
                        selected_language,
                        context
                    )

                st.write(
                    answer
                )

                st.warning(
                    "⚠️ The recommendation is based on "
                    "the trained ML model. Actual crop "
                    "suitability also depends on local "
                    "soil, season, water and farming conditions."
                )


# ============================================================
# FERTILIZER RECOMMENDATION
# ============================================================

elif menu == "Fertilizer Recommendation":

    st.header(
        "🧪 Fertilizer Recommendation"
    )

    st.write(
        "Enter soil, crop and nutrient information "
        "to get a fertilizer recommendation."
    )

    if fertilizer_bundle is None:

        st.error(
            "❌ Fertilizer model not found."
        )

    else:

        soil_types, crop_types = (
            get_fertilizer_options()
        )

        st.subheader(
            "🌱 Enter Fertilizer Information"
        )

        c1, c2 = st.columns(2)

        with c1:

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=0.0,
                value=25.0,
                key="fert_temperature"
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                key="fert_humidity"
            )

            moisture = st.number_input(
                "Soil Moisture",
                min_value=0.0,
                value=30.0,
                key="fert_moisture"
            )

            nitrogen = st.number_input(
                "Nitrogen",
                min_value=0.0,
                value=50.0,
                key="fert_nitrogen"
            )

        with c2:

            potassium = st.number_input(
                "Potassium",
                min_value=0.0,
                value=50.0,
                key="fert_potassium"
            )

            phosphorous = st.number_input(
                "Phosphorous",
                min_value=0.0,
                value=50.0,
                key="fert_phosphorous"
            )

            soil_type = st.selectbox(
                "Soil Type",
                soil_types,
                key="fert_soil_type"
            )

            crop_type = st.selectbox(
                "Crop Type",
                crop_types,
                key="fert_crop_type"
            )

        if st.button(
            "🧪 Recommend Fertilizer",
            key="fertilizer_button"
        ):

            with st.spinner(
                "🧪 Finding suitable fertilizer..."
            ):

                fertilizer = predict_fertilizer(

                    temperature,
                    humidity,
                    moisture,
                    soil_type,
                    crop_type,
                    nitrogen,
                    potassium,
                    phosphorous
                )

            if fertilizer:

                st.success(
                    f"🧪 Recommended Fertilizer: "
                    f"**{fertilizer}**"
                )

                context = f"""

Fertilizer model result:

Recommended fertilizer:
{fertilizer}

Temperature:
{temperature}

Humidity:
{humidity}

Soil moisture:
{moisture}

Soil type:
{soil_type}

Crop type:
{crop_type}

Nitrogen:
{nitrogen}

Potassium:
{potassium}

Phosphorous:
{phosphorous}
"""

                question = f"""

The fertilizer recommendation model recommended:

{fertilizer}

Explain this recommendation simply.

Explain:

1. Why this fertilizer may be suitable
2. What nutrients it generally provides
3. Why soil testing matters
4. How the farmer should use the recommendation safely
5. What the farmer should check before applying fertilizer

Do not give exact fertilizer doses.
Do not guarantee crop yield.
"""

                st.subheader(
                    "🤖 AI Fertilizer Explanation"
                )

                with st.spinner(
                    "🤖 Preparing explanation..."
                ):

                    answer = ask_agri_ai(
                        question,
                        selected_language,
                        context
                    )

                st.write(
                    answer
                )

                st.warning(
                    "⚠️ Fertilizer recommendations should "
                    "be confirmed using a soil-test report "
                    "and local agricultural guidance."
                )


# ============================================================
# SOIL NUTRIENT ANALYSIS
# ============================================================

elif menu == "Soil Nutrient Analysis":

    st.header(
        "🧪 Soil Nutrient Analysis"
    )

    st.write(
        "Analyze a nutrient value from a soil-test report."
    )

    if (
        soil_model is None
        or soil_encoders is None
    ):

        st.error(
            "❌ Soil model or encoders not found."
        )

    else:

        try:

            state_encoder = (
                soil_encoders["state_name"]
            )

            district_encoder = (
                soil_encoders["district_name"]
            )

            nutrient_encoder = (
                soil_encoders["nutrient_name"]
            )

            state_list = list(
                state_encoder.classes_
            )

            district_list = list(
                district_encoder.classes_
            )

            nutrient_list = list(
                nutrient_encoder.classes_
            )

            st.subheader(
                "📍 Step 1: Location"
            )

            selected_state = st.selectbox(
                "State",
                state_list,
                key="soil_state_new"
            )

            selected_district = st.selectbox(
                "District",
                district_list,
                key="soil_district_new"
            )

            st.subheader(
                "🧪 Step 2: Nutrient"
            )

            selected_nutrient = st.selectbox(
                "Nutrient",
                nutrient_list,
                key="soil_nutrient_new"
            )

            st.subheader(
                "📊 Step 3: Soil-Test Value"
            )

            nutrient_value = st.number_input(
                "Nutrient Value",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="soil_value_new"
            )

            st.info(
                "Enter the value exactly as shown in "
                "the soil-test report."
            )

            if st.button(
                "🧪 Analyze Soil",
                key="soil_analysis_button"
            ):

                if nutrient_value <= 0:

                    st.warning(
                        "⚠️ Please enter a valid nutrient value."
                    )

                else:

                    with st.spinner(
                        "🧪 Analyzing soil..."
                    ):

                        result = predict_soil_nutrient(

                            selected_state,
                            selected_district,
                            selected_nutrient,
                            nutrient_value
                        )

                    if result:

                        st.success(
                            f"🧪 Predicted Nutrient Level: "
                            f"**{result}**"
                        )

                        context = f"""

State:
{selected_state}

District:
{selected_district}

Nutrient:
{selected_nutrient}

Nutrient value:
{nutrient_value}

Predicted nutrient level:
{result}

This model predicts a nutrient-level category.
It is not a complete soil-health diagnosis.
"""

                        question = f"""

The soil nutrient model predicted:

Nutrient:
{selected_nutrient}

Value:
{nutrient_value}

Level:
{result}

Explain this result simply to a farmer.

Explain:

1. What the nutrient is
2. Why it is important
3. What the predicted level means
4. What farmers can monitor
5. Why soil testing is important
6. What the farmer should do next

Do not give exact fertilizer doses.
Do not claim this is a complete soil-health diagnosis.
"""

                        st.subheader(
                            "🤖 AI Soil Suggestions"
                        )

                        with st.spinner(
                            "🤖 Preparing soil advice..."
                        ):

                            answer = ask_agri_ai(
                                question,
                                selected_language,
                                context
                            )

                        st.write(
                            answer
                        )

                        st.warning(
                            "⚠️ This is a machine-learning "
                            "prediction of a nutrient-level "
                            "category. For fertilizer and soil "
                            "management decisions, use a proper "
                            "laboratory soil-test report."
                        )

                    else:

                        st.error(
                            "❌ Soil prediction failed."
                        )

        except Exception as e:

            st.error(
                f"❌ Soil module error: {e}"
            )


# ============================================================
# WEATHER INFORMATION
# ============================================================

elif menu == "🌦️ Weather Information":

    st.header(
        "🌦️ Weather Information"
    )

    st.write(
        "Search for an Indian location to view current "
        "weather and a 7-day forecast."
    )

    location = st.text_input(
        "📍 Enter location",
        placeholder="Example: Guntur",
        key="weather_location"
    )

    if location.strip():

        locations = search_location(
            location
        )

        if locations:

            selected_name = st.selectbox(
                "Select location",
                [
                    x["display"]
                    for x in locations
                ],
                key="weather_location_select"
            )

            selected = next(
                x for x in locations
                if x["display"] == selected_name
            )

            if st.button(
                "🌦️ Get Weather",
                key="weather_button"
            ):

                try:

                    weather = get_weather(

                        selected["latitude"],

                        selected["longitude"]
                    )

                    current = weather[
                        "current"
                    ]

                    st.subheader(
                        "🌤️ Current Weather"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Temperature",
                            f"{current['temperature_2m']} °C"
                        )

                    with c2:

                        st.metric(
                            "Humidity",
                            f"{current['relative_humidity_2m']} %"
                        )

                    with c3:

                        st.metric(
                            "Wind",
                            f"{current['wind_speed_10m']} km/h"
                        )

                    code = current[
                        "weather_code"
                    ]

                    st.info(
                        f"☁️ Weather: "
                        f"{weather_description(code)}"
                    )

                    st.subheader(
                        "📅 7-Day Forecast"
                    )

                    daily = weather[
                        "daily"
                    ]

                    forecast_df = pd.DataFrame({

                        "Date":
                            daily["time"],

                        "Max °C":
                            daily[
                                "temperature_2m_max"
                            ],

                        "Min °C":
                            daily[
                                "temperature_2m_min"
                            ],

                        "Rainfall mm":
                            daily[
                                "precipitation_sum"
                            ],

                        "Weather":
                            [
                                weather_description(
                                    x
                                )
                                for x in
                                daily[
                                    "weather_code"
                                ]
                            ]
                    })

                    st.dataframe(
                        forecast_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader(
                        "👨‍🌾 Simple Farming Advice"
                    )

                    rain_total = sum(
                        x or 0
                        for x in
                        daily[
                            "precipitation_sum"
                        ]
                    )

                    if rain_total > 30:

                        st.info(
                            "🌧️ Rainfall is expected. "
                            "Avoid unnecessary irrigation "
                            "and monitor drainage."
                        )

                    else:

                        st.info(
                            "💧 Rainfall is relatively limited. "
                            "Monitor soil moisture and irrigation needs."
                        )

                    st.markdown(
                        "### 🌐 Official IMD Weather"
                    )

                    st.link_button(
                        "Open IMD Mausam",
                        "https://mausam.imd.gov.in/"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Weather error: {e}"
                    )


# ============================================================
# GOVERNMENT SCHEMES
# ============================================================

elif menu == "Government Schemes":

    st.header(
        "🏛️ Government Schemes for Farmers"
    )

    st.write(
        "AgriOne AI provides information and official "
        "websites for selected farmer-support schemes."
    )

    schemes = {

        "PM-KISAN": {

            "description":
                "Financial support for eligible landholding "
                "farmer families.",

            "benefit":
                "Financial support according to scheme rules.",

            "website":
                "https://pmkisan.gov.in/"
        },

        "Pradhan Mantri Fasal Bima Yojana (PMFBY)": {

            "description":
                "Crop insurance support against eligible "
                "crop losses.",

            "benefit":
                "Crop insurance protection.",

            "website":
                "https://pmfby.gov.in/"
        },

        "Kisan Credit Card (KCC)": {

            "description":
                "Institutional agricultural credit facility "
                "for eligible farmers.",

            "benefit":
                "Agricultural credit support.",

            "website":
                "https://www.myscheme.gov.in/"
        },

        "Agriculture Infrastructure Fund": {

            "description":
                "Financial support for eligible agricultural "
                "infrastructure and post-harvest facilities.",

            "benefit":
                "Infrastructure financing support.",

            "website":
                "https://agriinfra.dac.gov.in/"
        }
    }

    selected_scheme = st.selectbox(
        "Select Scheme",
        list(schemes.keys())
    )

    scheme = schemes[
        selected_scheme
    ]

    st.subheader(
        f"🏛️ {selected_scheme}"
    )

    st.write(
        scheme["description"]
    )

    st.info(
        f"💰 Support: {scheme['benefit']}"
    )

    st.link_button(
        "🌐 Open Official Website",
        scheme["website"]
    )

    st.warning(
        "⚠️ Eligibility and benefits can depend on "
        "current government rules. Verify details on "
        "the official website before applying."
    )


# ============================================================
# MARKET PRICES
# ============================================================

elif menu == "Market Prices":

    st.header(
        "📈 Agricultural Market Prices"
    )

    st.write(
        "Use official agricultural market portals to "
        "check current commodity prices."
    )

    st.info(
        "AgriOne AI does not invent market prices. "
        "Current prices should be verified from official "
        "market sources."
    )

    st.subheader(
        "🌾 Official Market Price Sources"
    )

    st.link_button(
        "📊 Open Agmarknet",
        "https://agmarknet.gov.in/"
    )

    st.link_button(
        "🏪 Open e-NAM",
        "https://www.enam.gov.in/"
    )

    st.link_button(
        "📚 Open data.gov.in",
        "https://www.data.gov.in/"
    )

    st.markdown("---")

    st.subheader(
        "🤖 Ask AgriOne AI About Market Prices"
    )

    market_question = st.text_input(
        "Example: Where can I check today's rice price?"
    )

    if st.button(
        "🤖 Ask AI",
        key="market_ai_button"
    ):

        if market_question.strip():

            answer = ask_agri_ai(

                market_question,

                selected_language,

                """
Market prices can change frequently.
Do not invent a price.
Tell the farmer to verify the current price
using official market sources such as Agmarknet
or e-NAM.
"""
            )

            st.write(
                answer
            )


# ============================================================
# AI AGRICULTURE ASSISTANT
# ============================================================

elif menu == "AI Agriculture Assistant":

    st.header(
        "🤖 AI Agriculture Assistant"
    )

    st.write(
        "Ask AgriOne AI questions about farming."
    )

    st.info(
        f"🌐 Current language: "
        f"**{selected_language}**"
    )

    question = st.text_area(

        "Ask your farming question",

        placeholder=(
            "Example: My rice leaves are turning yellow. "
            "What should I check?"
        ),

        height=120
    )

    if st.button(
        "🤖 Ask AgriOne AI",
        key="ask_ai_button"
    ):

        if question.strip():

            with st.spinner(
                "🤖 AgriOne AI is thinking..."
            ):

                answer = ask_agri_ai(

                    question,

                    selected_language,

                    ""
                )

            st.subheader(
                "🌱 AgriOne AI Answer"
            )

            st.write(
                answer
            )

            # Text to speech

            audio_file = speak_answer(
                answer,
                selected_language
            )

            if audio_file:

                st.audio(
                    audio_file,
                    format="audio/mp3"
                )

        else:

            st.warning(
                "⚠️ Please enter a question."
            )


# ============================================================
# VOICE ASSISTANT
# ============================================================

elif menu == "Voice Assistant":

    st.header(
        "🎤 Voice Assistant"
    )

    st.write(
        "Speak your farming question and AgriOne AI "
        "will answer in the selected language."
    )

    st.info(
        f"🌐 Selected language: "
        f"**{selected_language}**"
    )

    audio = mic_recorder(

        start_prompt="🎤 Start Recording",

        stop_prompt="⏹️ Stop Recording",

        just_once=True,

        use_container_width=True,

        key="voice_recorder"
    )

    if audio:

        st.success(
            "✅ Voice recorded."
        )

        try:

            # ------------------------------------------------
            # SAVE RECORDED AUDIO
            # ------------------------------------------------

            audio_bytes = audio[
                "bytes"
            ]

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".webm"
            )

            input_file.write(
                audio_bytes
            )

            input_file.close()

            # ------------------------------------------------
            # CONVERT TO WAV
            # ------------------------------------------------

            sound = AudioSegment.from_file(
                input_file.name
            )

            sound = sound.set_channels(
                1
            )

            sound = sound.set_frame_rate(
                16000
            )

            sound = sound.set_sample_width(
                2
            )

            wav_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )

            wav_file.close()

            sound.export(
                wav_file.name,
                format="wav"
            )

            # ------------------------------------------------
            # SPEECH RECOGNITION
            # ------------------------------------------------

            recognizer = sr.Recognizer()

            with sr.AudioFile(
                wav_file.name
            ) as source:

                recorded_audio = (
                    recognizer.record(
                        source
                    )
                )

            with st.spinner(
                "🎤 Converting speech to text..."
            ):

                text = recognizer.recognize_google(

                    recorded_audio,

                    language=LANGUAGES[
                        selected_language
                    ]["speech"]
                )

            st.subheader(
                "📝 Recognized Question"
            )

            st.write(
                text
            )

            # ------------------------------------------------
            # OLLAMA
            # ------------------------------------------------

            with st.spinner(
                "🤖 AgriOne AI is preparing the answer..."
            ):

                answer = ask_agri_ai(

                    text,

                    selected_language,

                    ""
                )

            st.subheader(
                "🤖 AgriOne AI Answer"
            )

            st.write(
                answer
            )

            # ------------------------------------------------
            # TEXT TO SPEECH
            # ------------------------------------------------

            with st.spinner(
                "🔊 Creating voice answer..."
            ):

                audio_file = speak_answer(

                    answer,

                    selected_language
                )

            if audio_file:

                st.audio(
                    audio_file,
                    format="audio/mp3"
                )

            else:

                st.warning(
                    "⚠️ Voice output could not be created."
                )

        except sr.UnknownValueError:

            st.error(
                "❌ I could not understand the speech. "
                "Please speak clearly and try again."
            )

        except sr.RequestError:

            st.error(
                "❌ Speech recognition needs an internet "
                "connection. Please check your internet."
            )

        except Exception as e:

            st.error(
                f"❌ Voice assistant error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌱 AgriOne AI – Multilingual Smart Farming Assistant"
)

st.caption(
    "AI and ML results are provided for assistance and "
    "should be verified with appropriate agricultural experts."
)