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



# ============================================================
# AUTOMATIC UI TRANSLATION (English / Telugu / Hindi / Tamil)
# ============================================================
# Translates visible UI text on every module. Requires internet access and
# deep-translator. If translation service is unavailable, original text is
# shown rather than crashing the app. Model labels/data are not changed.
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

_UI_LANG_CODES = {"English": "en", "Telugu": "te", "Hindi": "hi", "Tamil": "ta"}
_UI_TRANSLATION_CACHE = {}

def _ui_language():
    return st.session_state.get("global_language", "English")

def _tr_ui(value):
    if not isinstance(value, str) or not value.strip():
        return value
    language = _ui_language()
    target = _UI_LANG_CODES.get(language, "en")
    if target == "en" or GoogleTranslator is None:
        return value
    key = (target, value)
    if key in _UI_TRANSLATION_CACHE:
        return _UI_TRANSLATION_CACHE[key]
    try:
        translated = GoogleTranslator(source="auto", target=target).translate(value)
        _UI_TRANSLATION_CACHE[key] = translated or value
        return translated or value
    except Exception:
        return value

# Translate the visible text arguments of Streamlit display/widgets. Keep
# option values unchanged so the app's internal feature routing still works.
def _install_ui_translation():
    names = [
        "title", "header", "subheader", "caption", "write", "markdown",
        "text", "info", "success", "warning", "error", "exception",
        "divider", "button", "download_button", "checkbox", "toggle",
        "radio", "selectbox", "multiselect", "select_slider", "slider",
        "text_input", "text_area", "number_input", "date_input",
        "time_input", "file_uploader", "camera_input", "form_submit_button",
        "chat_input", "metric", "spinner", "expander", "container",
    ]
    for name in names:
        original = getattr(st, name, None)
        if original is None or getattr(original, "_agrione_translated", False):
            continue
        def make_wrapper(fn):
            def wrapper(*args, **kwargs):
                args = list(args)
                if args and isinstance(args[0], str):
                    args[0] = _tr_ui(args[0])
                for field in ("label", "placeholder", "help", "start_prompt", "stop_prompt"):
                    if isinstance(kwargs.get(field), str):
                        kwargs[field] = _tr_ui(kwargs[field])
                # Display translated option labels while preserving original
                # values returned to the prediction/model code.
                if len(args) > 1 and isinstance(args[1], (list, tuple)) and all(isinstance(x, str) for x in args[1]):
                    kwargs.setdefault("format_func", _tr_ui)
                return fn(*args, **kwargs)
            wrapper._agrione_translated = True
            return wrapper
        setattr(st, name, make_wrapper(original))

    # Translate common sidebar labels too, without translating option values.
    for obj in (st.sidebar,):
        for name in ("title", "header", "subheader", "caption", "write", "markdown", "radio", "selectbox"):
            original = getattr(obj, name, None)
            if original is None or getattr(original, "_agrione_translated", False):
                continue
            def make_sidebar_wrapper(fn):
                def wrapper(*args, **kwargs):
                    args = list(args)
                    if args and isinstance(args[0], str):
                        args[0] = _tr_ui(args[0])
                    for field in ("label", "help"):
                        if isinstance(kwargs.get(field), str):
                            kwargs[field] = _tr_ui(kwargs[field])
                    if len(args) > 1 and isinstance(args[1], (list, tuple)) and all(isinstance(x, str) for x in args[1]):
                        kwargs.setdefault("format_func", _tr_ui)
                    return fn(*args, **kwargs)
                wrapper._agrione_translated = True
                return wrapper
            setattr(obj, name, make_sidebar_wrapper(original))

_install_ui_translation()


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

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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

# Streamlit Cloud uses Linux. If ffmpeg is installed and available on PATH,
# pydub can use it automatically. Do not use a hard-coded Windows path here.


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
    if not os.path.isfile(CROP_MODEL_PATH):
        return None
    try:
        return joblib.load(CROP_MODEL_PATH)
    except Exception as exc:
        st.warning(f"Crop model could not be loaded: {exc}")
        return None


crop_model = load_crop_model()


# ============================================================
# LOAD FERTILIZER MODEL
# ============================================================

@st.cache_resource
def load_fertilizer_model():
    if not os.path.isfile(FERTILIZER_MODEL_PATH):
        return None
    try:
        bundle = joblib.load(FERTILIZER_MODEL_PATH)
        # Expected deployment format: {"model": fitted_model, "columns": [...]}.
        if isinstance(bundle, dict) and "model" in bundle and "columns" in bundle:
            return bundle
        # A plain estimator can be loaded, but it needs its original feature columns.
        st.warning("Fertilizer model loaded, but it is not a model+columns bundle. Re-upload the matching fertilizer_recommendation_model.pkl.")
        return None
    except Exception as exc:
        st.warning(f"Fertilizer model could not be loaded: {exc}")
        return None


fertilizer_bundle = load_fertilizer_model()


# ============================================================
# LOAD SOIL MODEL AND ENCODERS
# ============================================================

@st.cache_resource
def load_soil_models():
    if not os.path.isfile(SOIL_MODEL_PATH) or not os.path.isfile(SOIL_ENCODERS_PATH):
        return None, None
    try:
        model = joblib.load(SOIL_MODEL_PATH)
        encoders = joblib.load(SOIL_ENCODERS_PATH)
        required = {"state_name", "district_name", "nutrient_name", "nutrient_level"}
        if not isinstance(encoders, dict) or not required.issubset(encoders.keys()):
            st.warning("Soil encoder file does not contain the required encoder keys.")
            return None, None
        return model, encoders
    except Exception as exc:
        st.warning(f"Soil model or encoders could not be loaded: {exc}")
        return None, None


soil_model, soil_encoders = load_soil_models()


# ============================================================
# LOAD PLANT DISEASE MODEL
# ============================================================

@st.cache_resource
def load_plant_model():
    if not os.path.isfile(PLANT_MODEL_PATH):
        return None
    try:
        return load_model(PLANT_MODEL_PATH, compile=False)
    except Exception as exc:
        st.warning(f"Plant disease model could not be loaded: {exc}")
        return None


plant_model = load_plant_model()


# ============================================================
# LOAD PLANT DISEASE CLASSES
# ============================================================

@st.cache_resource
def load_plant_classes():
    if not os.path.isfile(PLANT_CLASSES_PATH):
        return None
    try:
        with open(PLANT_CLASSES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        st.warning(f"Plant disease class file could not be loaded: {exc}")
        return None


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
    """Get multilingual agriculture help through OpenRouter, not local Ollama."""
    language_names = {"English", "Telugu", "Hindi", "Tamil"}
    if language not in language_names:
        language = "English"

    system_prompt = f"""
You are AgriOne AI, a multilingual smart farming assistant for farmers.
Answer in {language}. Support English, Telugu, Hindi and Tamil.
Help with crop selection, soil nutrients, fertilizer, irrigation, plant diseases,
pests, weather, government schemes, market prices and general farming.
Use simple farmer-friendly language and practical steps. Do not invent facts,
current weather, live market prices, government eligibility or scheme rules.
Do not claim a plant disease prediction is certain. Do not give exact fertilizer
doses unless supported by a soil report or verified recommendation. Encourage
soil testing and local agricultural guidance when appropriate. Keep answers
reasonably short and easy to understand.
Additional information:
{extra_context}
"""

    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return (
                "OpenRouter API key is missing. Add OPENROUTER_API_KEY in "
                "Streamlit Cloud → your app → Settings → Secrets."
            )

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://share.streamlit.io/",
                "X-Title": "AgriOne AI",
            },
            json={
                "model": "openrouter/free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "temperature": 0.4,
            },
            timeout=60,
        )

        if response.status_code != 200:
            return (
                f"AI service error ({response.status_code}). "
                f"Please check the OpenRouter API key and free-model availability. "
                f"Details: {response.text[:400]}"
            )

        data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not answer:
            return "The AI service returned an empty answer. Please try again."
        return answer.strip()

    except Exception as exc:
        return f"AI assistant error: {exc}"


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
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 AgriOne AI")
st.sidebar.caption("🌐 Language selection below applies to all modules and sidebar headings.")

menu = st.sidebar.radio(

    "🌱 Select Feature",

    [

        "Home",

        "Plant Disease Detection",

        "Crop Recommendation",

        "Fertilizer Recommendation",

        "Soil Nutrient Analysis",

        "🌦️ Weather Information",

        "Government Schemes",

        "Market Prices",

        "AI Agriculture Assistant",

        "Voice Assistant"
    ]
)


# ============================================================
# LANGUAGE SELECTOR AVAILABLE ON ALL 10 FEATURE SCREENS
# ============================================================
# The selector is shown in the main page so it is easy to find while using
# any of the 8 non-chat modules as well as the AI and Voice assistants.
# The chosen language is shared by all modules during the session.
selected_language = st.selectbox(
    "🌐 Choose language / భాష ఎంచుకోండి / भाषा चुनें / மொழியைத் தேர்ந்தெடுக்கவும்",
    list(LANGUAGES.keys()),
    key="global_language",
)
st.caption(
    "Selected language applies to all 10 modules, visible headings, side navigation, "
    "buttons, input labels, help text, and explanatory text where translation is available. "
    "Automatic UI translation requires an internet connection. Scientific crop/disease names, "
    "official scheme names, and raw model labels may remain in their original form."
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
                        "English",
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
                        "English",
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
                        "English",
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
                                "English",
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

                "English",

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
