import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
#from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
from tensorflow.keras.preprocessing import image
import os

# -------------------------------
# ✅ Load Real Model + Labels
# -------------------------------

# -------------------------------
# Load Model + Labels
# -------------------------------
from tensorflow.keras.models import load_model

MODEL_PATH = "/Users/omshelar/Desktop/food_analyzer/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
LABELS_PATH = "labels.txt"
TARGET_SIZE = (224, 224)

# Load model safely
# -------------------------------
# Load model safely
# -------------------------------
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
import streamlit as st

MODEL_PATH = "/Users/omshelar/Desktop/food_analyzer/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
LABELS_PATH = "labels.txt"

try:
    # Try to load your fine-tuned Food-101 model
    model = load_model(MODEL_PATH, compile=False)
    st.sidebar.success("✅ Model loaded successfully from file.")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not load model from file ({e}). Using pretrained EfficientNetB0 instead.")
    # Fallback to pretrained ImageNet model for testing
    model = EfficientNetB0(weights="imagenet")
    st.sidebar.info("📦 Using fallback pretrained EfficientNetB0 model (ImageNet).")
    st.sidebar.success("✅ Model loaded successfully!")

# Load class labels
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    st.sidebar.info(f"📘 Loaded {len(class_names)} labels.")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not load labels file: {e}")
    class_names = []


# -------------------------------
# Load Nutrition Data
# -------------------------------
nutrition_df = pd.read_csv("nutrition_data.csv")  # ensure CSV is in same folder


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Food Nutrition Analyzer", layout="centered")
st.title("🍽️ Food Nutrition Analyzer")
st.markdown("Upload a food image or manually search for nutrition information below.")

# --- Manual Food Search ---
st.subheader("🔍 Manual Food Search")
search_query = st.text_input("Search for a food item (e.g., Biryani, Pasta, Idli):")

if search_query:
    search_results = nutrition_df[nutrition_df['Food'].str.contains(search_query, case=False, na=False)]
    if not search_results.empty:
        st.dataframe(search_results.set_index("Food"))
    else:
        st.warning("No matching food found.")

# -------------------------------
# 📷 Food Image Recognition
# -------------------------------
# Use the model already loaded above (either fine-tuned model or fallback EfficientNetB0)
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load class names: prefer `classes.txt` (Food-101 style labels), fall back to earlier `labels.txt` if present
if os.path.exists("classes.txt"):
    with open("classes.txt", "r", encoding="utf-8") as f:
        class_names_used = [line.strip() for line in f if line.strip()]
else:
    # `class_names` was loaded earlier from `labels.txt` (or is an empty list)
    class_names_used = class_names if 'class_names' in globals() else []

st.subheader("📷 Food Image Recognition")
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])


def preprocess_pil(img_pil, target_size=TARGET_SIZE):
    # Ensure RGB
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # Resize with PIL then convert to numpy array
    img_resized = img_pil.resize(target_size)
    x = np.array(img_resized).astype("float32")
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Prepare a placeholder for later code that expects `food_info`
food_info = pd.DataFrame()

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Load a dedicated model for image classification (so we don't rely on the previously-loaded global)
    try:
        img_model = load_model(MODEL_PATH, compile=False)
        st.sidebar.info(f"✅ Image model loaded from {MODEL_PATH} for prediction.")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load image model from {MODEL_PATH}: {e}")
        # Fallback to any model already present in memory
        img_model = globals().get('model')
        if img_model is None:
            st.error("No model available for prediction.")

    try:
        x = preprocess_pil(img)
        preds = img_model.predict(x, verbose=0) if img_model is not None else None
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        preds = None

    if preds is not None:
        # Handle different output shapes
        if preds.ndim == 1:
            probs = tf.nn.softmax(preds).numpy()
        else:
            probs = tf.nn.softmax(preds[0]).numpy()

        # Top-5
        top_idx = probs.argsort()[-5:][::-1]

        # Model vs labels diagnostic
        n_model = probs.size
        n_labels = len(class_names_used)

        # Prepare display list: map indices to label names when possible, otherwise show numeric index
        top_preds = []
        for i in top_idx:
            if i < n_labels:
                name = class_names_used[i]
            else:
                name = None
            top_preds.append((i, name, probs[i]))

        # Display top predictions with clear diagnostics
        st.markdown("**Top predictions:**")
        for i, name, p in top_preds:
            if name:
                display_name = name.replace("_", " ").title()
            else:
                display_name = f"Class #{i} (no matching label)"
            st.write(f"- {display_name}: {p*100:.2f}%")

        # Decide best label to show to user
        best_idx = top_idx[0]
        confidence = probs[best_idx] * 100

        if n_model == n_labels and n_labels > 0:
            # Perfect match: map directly
            best_name = class_names_used[best_idx]
            predicted_food = best_name.replace("_", " ").title()
            st.success(f"🍴 Predicted Food: **{predicted_food}** ({confidence:.2f}% confidence)")
        else:
            # Mismatch between model output size and labels list
            st.warning(f"Model produced {n_model} scores but labels list has {n_labels} entries.\n"
                       "This usually means the loaded model doesn't match the labels file (e.g. you loaded a 'notop' feature extractor or an ImageNet model).")

            # If we can map directly to a label index, show it; otherwise show numeric index and a best-guess (modulo)
            if best_idx < n_labels:
                best_name = class_names_used[best_idx]
                predicted_food = best_name.replace("_", " ").title()
                st.success(f"🍴 Predicted Food: **{predicted_food}** ({confidence:.2f}% confidence)")
            else:
                # No direct mapping — show numeric index and a best-guess (unsafe) to help debugging
                guessed = None
                if n_labels > 0:
                    guessed = class_names_used[best_idx % n_labels]
                    guessed_display = guessed.replace("_", " ").title()
                    st.info(f"Best numeric class index: {best_idx} ({confidence:.2f}% confidence).\n"
                            f"Closest label guess (index modulo labels length): {guessed_display} — this is only a heuristic.")
                    predicted_food = guessed_display
                else:
                    st.info(f"Best numeric class index: {best_idx} ({confidence:.2f}% confidence). No labels available to map to names.")
                    predicted_food = str(best_idx)

        # Try to find nutrition info (case-insensitive contains) using the predicted_food string
        try:
            food_info = nutrition_df[nutrition_df['Food'].str.contains(predicted_food, case=False, na=False)]
        except Exception:
            food_info = pd.DataFrame()

        if not food_info.empty:
            st.markdown("**Matched nutrition info:**")
            st.dataframe(food_info.set_index('Food'))
        else:
            st.info("No exact nutrition match found for the predicted label.")



# -------------------------------
# 📊 Daily Food Log with Quantity
# -------------------------------
if "food_log" not in st.session_state:
    st.session_state["food_log"] = []

# If predicted food from image
if uploaded_file and not food_info.empty:
    qty = st.number_input(f"Enter quantity for {predicted_food} (grams or units):", min_value=1, step=1, key="img_qty")
    if st.button("Add to Daily Log (from Image)"):
        food_data = food_info.to_dict(orient="records")[0]
        factor = qty / 100  # CSV is per 100g
        scaled_data = {
            "Food": food_data["Food"],
            "Calories": food_data["Calories"] * factor,
            "Protein (g)": food_data["Protein (g)"] * factor,
            "Carbs (g)": food_data["Carbs (g)"] * factor,
            "Fats (g)": food_data["Fats (g)"] * factor,
            "Quantity": f"{qty} g"
        }
        st.session_state["food_log"].append(scaled_data)
        st.success(f"✅ Added {qty} g of {predicted_food} to your daily log!")

# If searched manually
if search_query and 'search_results' in locals() and not search_results.empty:
    selected_food = st.selectbox("Select a food from search results to log:", search_results["Food"].tolist())
    qty = st.number_input(f"Enter quantity for {selected_food} (grams or units):", min_value=1, step=1, key="search_qty")

    if st.button("Add to Daily Log (from Search)"):
        food_data = nutrition_df[nutrition_df["Food"] == selected_food].to_dict(orient="records")[0]
        factor = qty / 100  # CSV is per 100g

        # Convert numeric columns to float
        for col in ["Calories", "Protein (g)", "Carbs (g)", "Fats (g)"]:
            food_data[col] = float(food_data[col])

        # Create adjusted data
        adjusted_data = {
            "Food": selected_food,
            "Quantity (g/units)": qty,
            "Calories": food_data["Calories"] * factor,
            "Protein (g)": food_data["Protein (g)"] * factor,
            "Carbs (g)": food_data["Carbs (g)"] * factor,
            "Fats (g)": food_data["Fats (g)"] * factor
        }

        # ✅ Append inside the same block
        st.session_state["food_log"].append(adjusted_data)

        st.success(f"{selected_food} added to your daily log!")

# Show Daily Log
st.markdown("---")
st.subheader("📊 Your Daily Nutrition Log (24 hrs)")

if st.session_state["food_log"]:
    log_df = pd.DataFrame(st.session_state["food_log"])
    st.dataframe(log_df.set_index("Food"))

    # Totals
    total_cal = log_df["Calories"].sum()
    total_protein = log_df["Protein (g)"].sum()
    total_carbs = log_df["Carbs (g)"].sum()
    total_fat = log_df["Fats (g)"].sum()

    st.write(f"**Total Calories:** {total_cal:.1f} kcal")
    st.write(f"**Protein:** {total_protein:.1f} g | **Carbs:** {total_carbs:.1f} g | **Fat:** {total_fat:.1f} g")

    # Clear log button
    if st.button("🗑️ Clear Today's Log"):
        st.session_state["food_log"] = []
        st.success("Daily log cleared!")

else:
    st.info("No foods logged yet today. Add foods above!")

# -------------------------------
# -------------------------------
# ⚖️ BMI Calculator, Diet & Exercise Plan (Collapsible Exercises)
# -------------------------------
st.header("⚖️ BMI Calculator & Health Suggestions")

# User inputs
weight = st.number_input("Enter your weight (kg):", min_value=1, step=1)
height = st.number_input("Enter your height (cm):", min_value=50, step=1)
age = st.number_input("Enter your age:", min_value=1, step=1)
diet_type = st.selectbox("Diet Preference:", ["veg", "nonveg"])
lifestyle = st.selectbox("Lifestyle:", ["sedentary", "moderate", "active"])
condition = st.selectbox("Medical Condition:", ["none", "diabetes", "hypertension", "cholesterol"])

if st.button("Calculate BMI & Get Plan"):

    # ---------- BMI Calculation ----------
    if weight and height:
        bmi_value = weight / ((height / 100) ** 2)
        if bmi_value < 18.5:
            category = "Underweight"
        elif bmi_value < 24.9:
            category = "Normal"
        elif bmi_value < 29.9:
            category = "Overweight"
        else:
            category = "Obese"

        st.success(f"Your BMI is **{bmi_value:.2f}** → Category: **{category}**")

    # ---------- Diet Plan ----------
    st.subheader("🥗 Suggested Diet Plan")
    diet = []
    if category == "Underweight":
        diet = [
            "Breakfast: Banana shake / oats porridge with nuts",
            "Lunch: Rice, dal, 2 rotis, sabzi",
            "Snack: Dry fruits or paneer sandwich",
            "Dinner: Khichdi or chicken curry with rice"
        ]
    elif category in ["Overweight", "Obese"]:
        diet = [
            "Breakfast: Vegetable oats upma + green tea",
            "Lunch: 2 rotis, dal, sabzi, salad",
            "Snack: Fruit bowl (papaya, apple, guava)",
            "Dinner: Grilled paneer/chicken with vegetables"
        ]
    else:  # Normal
        diet = [
            "Breakfast: Poha / idli + chutney",
            "Lunch: Balanced thali with roti, dal, sabzi",
            "Snack: Sprouts or fruit salad",
            "Dinner: Light khichdi or veg pulao"
        ]

    # Condition adjustments
    if condition == "diabetes":
        diet.append("⚠️ Note: Avoid sugar, include high-fiber foods like oats & vegetables.")
    elif condition == "hypertension":
        diet.append("⚠️ Note: Reduce salt, eat more leafy greens and fruits.")
    elif condition == "cholesterol":
        diet.append("⚠️ Note: Avoid fried/oily foods, include flax seeds & omega-3 rich foods.")

    st.write("\n".join([f"- {d}" for d in diet]))

    # ---------- Exercise Plan ----------
    st.subheader("🏃 Suggested Exercise Plan with Videos")

    # BMI-based exercises
    exercise_plan = {
        "Underweight": [
            {"name": "Light strength training (push-ups, squats)", "videos": ["https://www.youtube.com/watch?v=IODxDxX7oi4"]},
            {"name": "Yoga or stretching exercises", "videos": ["https://youtu.be/AUsbthQ9W-I?si=j0ZIeC9ZdGbj5j1WE"]}
        ],
        "Normal": [
            {"name": "Cardio 30 min x 3-4 days/week", "videos": ["https://youtu.be/FrhvZ5pXtr4?si=4gSDtWHNw1Am6oQY"]},
            {"name": "Strength training 2-3 days/week", "videos": ["https://youtu.be/NsIn-z5bOWk?si=hYYOro7TuUfcUxn2"]},
            {"name": "Yoga/meditation for flexibility", "videos": ["https://youtu.be/3rTdYCWrm8c?si=LIwYFIjXBjs9rWlC"]}
        ],
        "Overweight": [
            {"name": "30 minutes brisk walking daily", "videos": ["https://www.youtube.com/watch?v=1minvideoexample"]},
            {"name": "20 minutes fatloss home workout", "videos": ["https://youtu.be/FeR-4_Opt-g?si=lFuR5tsnGN_0Eqtr"]},
            {"name": "Beginner yoga or stretching", "videos": ["https://www.youtube.com/watch?v=v7AYKMP6rOE"]}
        ],
        "Obese": [
            {"name": "Start with 15-20 minutes walking", "videos": ["https://youtu.be/wQrV75N2BrI?si=v2Qb_-M3FTKGh5b3"]},
             {"name": "Strength training 2-3 days/week", "videos": ["https://youtu.be/NsIn-z5bOWk?si=hYYOro7TuUfcUxn2"]},
            {"name": "Avoid heavy lifting in the beginning", "videos": ["https://www.youtube.com/watch?v=IODxDxX7oi4"]}
        ]
    }

    # Condition-based exercises
    condition_exercises = {
        "diabetes": [
            {"name": "Gentle walking after meals", "videos": ["https://youtu.be/sxBmITOwQ54?si=cJpyYnAU-9E52xCz"]},
            {"name": "Low-impact aerobic exercises", "videos": ["https://youtu.be/v8CDptlpeys?si=Guh0lbbtsZCFMbiS"]}
        ],
        "hypertension": [
            {"name": "Moderate yoga for relaxation", "videos": ["https://www.youtube.com/watch?v=v7AYKMP6rOE"]},
            {"name": "Brisk walking 20-30 mins", "videos": ["https://youtu.be/wQrV75N2BrI?si=v2Qb_-M3FTKGh5b3"]}
        ],
        "cholesterol": [
            {"name": "20 minutes fatloss home workout", "videos": ["https://youtu.be/FeR-4_Opt-g?si=lFuR5tsnGN_0Eqtr"]},
            {"name": "Strength training 2 days/week", "videos": ["https://youtu.be/NsIn-z5bOWk?si=hYYOro7TuUfcUxn2"]}
        ]
    }

    # Combine exercises
    selected_exercises = exercise_plan.get(category, [])
    if condition != "none":
        selected_exercises += condition_exercises.get(condition, [])

    # Display collapsible exercises with videos
    for exercise in selected_exercises:
        with st.expander(exercise["name"]):
            for video_url in exercise["videos"]:
                st.video(video_url)


   # -------------------------------
# 🤖 AI Chatbot Section (via OpenRouter)
# -------------------------------
import requests
import streamlit as st

# Use the free GPT-OSS 20B model from OpenRouter
# 🔑 Register and get your API key from https://openrouter.ai
OPENROUTER_API_KEY = "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"   # Replace with your key

st.header("🤖 Nutrition Chatbot")
st.markdown("Ask me anything about food, nutrition, or diet plans!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input
user_input = st.text_input("You:", key="chat_input")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a friendly nutrition and diet assistant. \
        Answer questions about foods, calories, nutrients, diet plans, medical conditions, \
        and healthy lifestyle tips. Keep your answers short, clear, and practical."}
    ] + st.session_state.chat_history

    # Call OpenRouter API
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-20b",   # 🧠 GPT-OSS 20B free model
                "messages": messages,
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            bot_reply = data["choices"][0]["message"]["content"]
        else:
            bot_reply = f"Error {response.status_code}: {response.text}"

    except Exception as e:
        bot_reply = f"⚠️ Error: {e}"

    # Add bot reply
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}**")
    else:
        st.markdown(f"**Bot:** {chat['content']}**")
