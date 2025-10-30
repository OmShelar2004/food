import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import requests
import base64
from io import BytesIO


st.set_page_config(page_title="Food Nutrition Analyzer", layout="centered")

# -------------------------------
# 🔐 LOGIN SYSTEM
# -------------------------------
def login_page():
    """Display login/signup page"""
    st.title("🍽️ Food Nutrition Analyzer")
    st.markdown("### Welcome! Please login to continue")
    
    # Create tabs for Login and Signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", key="login_btn"):
            if login_username and login_password:
                # Check credentials
                if "users" not in st.session_state:
                    st.session_state["users"] = {}
                
                if login_username in st.session_state["users"]:
                    if st.session_state["users"][login_username] == login_password:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = login_username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password")
                else:
                    st.error("❌ Username not found. Please sign up first.")
            else:
                st.warning("⚠️ Please enter both username and password")
    
    with tab2:
        st.subheader("Sign Up")
        signup_username = st.text_input("Choose Username", key="signup_user")
        signup_password = st.text_input("Choose Password", type="password", key="signup_pass")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_pass_confirm")
        
        if st.button("Sign Up", key="signup_btn"):
            if signup_username and signup_password and signup_password_confirm:
                if "users" not in st.session_state:
                    st.session_state["users"] = {}
                
                if signup_username in st.session_state["users"]:
                    st.error("❌ Username already exists. Please choose another.")
                elif signup_password != signup_password_confirm:
                    st.error("❌ Passwords don't match")
                elif len(signup_password) < 4:
                    st.error("❌ Password must be at least 4 characters long")
                else:
                    st.session_state["users"][signup_username] = signup_password
                    st.success("✅ Account created successfully! Please login.")
            else:
                st.warning("⚠️ Please fill all fields")

# -------------------------------
# Check if user is logged in
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# -------------------------------
# MAIN APPLICATION (Only shows after login)
# -------------------------------

# Roboflow API Configuration
ROBOFLOW_API_KEY = "miqjbOE79Xa7IqeGO0fg"
ROBOFLOW_MODEL_ENDPOINT = "https://outline.roboflow.com/food-image-segmentation-yolov5-wm2it/1"

def classify_food_with_roboflow(image_file):
    """
    Send image to Roboflow API and get predictions
    
    Args:
        image_file: PIL Image or file-like object
        
    Returns:
        dict: Predictions from Roboflow API
    """
    try:
        # Convert PIL Image to base64
        if isinstance(image_file, Image.Image):
            buffered = BytesIO()
            image_file.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
        else:
            img_bytes = image_file.read()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Make API request
        response = requests.post(
            f"{ROBOFLOW_MODEL_ENDPOINT}?api_key={ROBOFLOW_API_KEY}",
            data=img_base64,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Roboflow API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error calling Roboflow API: {e}")
        return None

# -------------------------------
# Load Nutrition Data
# -------------------------------
nutrition_df = pd.read_csv(os.path.join("../data", "nutrition_data.csv"))

# -------------------------------
# Streamlit UI
# -------------------------------

# Add logout button in sidebar
st.sidebar.markdown(f"### 👤 Logged in as: **{st.session_state['username']}**")
if st.sidebar.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.rerun()

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
# 📷 Food Image Recognition with Roboflow
# -------------------------------
st.subheader("📷 Food Image Recognition")
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

food_info = pd.DataFrame()
predicted_food = None

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("🔍 Analyzing image with AI..."):
        # Call Roboflow API
        predictions = classify_food_with_roboflow(img)
    
    if predictions and 'predictions' in predictions:
        st.markdown("**🎯 Detection Results:**")
        
        # Sort predictions by confidence
        sorted_predictions = sorted(
            predictions['predictions'], 
            key=lambda x: x.get('confidence', 0), 
            reverse=True
        )
        
        # Display top predictions
        for idx, pred in enumerate(sorted_predictions[:5]):
            class_name = pred.get('class', 'Unknown')
            confidence = pred.get('confidence', 0) * 100
            st.write(f"{idx+1}. **{class_name}**: {confidence:.2f}% confidence")
        
        # Use the highest confidence prediction
        if sorted_predictions:
            best_prediction = sorted_predictions[0]
            predicted_food = best_prediction.get('class', 'Unknown')
            confidence = best_prediction.get('confidence', 0) * 100
            
            st.success(f"🍴 Predicted Food: **{predicted_food}** ({confidence:.2f}% confidence)")
            
            # Try to match with nutrition database
            try:
                food_info = nutrition_df[nutrition_df['Food'].str.contains(predicted_food, case=False, na=False)]
            except Exception:
                food_info = pd.DataFrame()
            
            if not food_info.empty:
                st.markdown("**📊 Nutrition Information:**")
                st.dataframe(food_info.set_index('Food'))
            else:
                st.info(f"ℹ️ No nutrition data found for '{predicted_food}' in database.")
    else:
        st.warning("⚠️ Could not detect any food items in the image. Please try another image.")

# -------------------------------
# 📊 Daily Food Log with Quantity
# -------------------------------
if "food_log" not in st.session_state:
    st.session_state["food_log"] = []

# Add from image prediction
if uploaded_file and predicted_food and not food_info.empty:
    qty = st.number_input(f"Enter quantity for {predicted_food} (grams):", min_value=1, step=1, key="img_qty")
    if st.button("Add to Daily Log (from Image)"):
        food_data = food_info.to_dict(orient="records")[0]
        factor = qty / 100
        scaled_data = {
            "Food": food_data["Food"],
            "Quantity (g)": qty,
            "Calories": food_data["Calories"] * factor,
            "Protein (g)": food_data["Protein (g)"] * factor,
            "Carbs (g)": food_data["Carbs (g)"] * factor,
            "Fats (g)": food_data["Fats (g)"] * factor
        }
        st.session_state["food_log"].append(scaled_data)
        st.success(f"✅ Added {qty}g of {predicted_food} to your daily log!")

# Add from search results
if search_query and 'search_results' in locals() and not search_results.empty:
    selected_food = st.selectbox("Select a food from search results to log:", search_results["Food"].tolist())
    qty = st.number_input(f"Enter quantity for {selected_food} (grams):", min_value=1, step=1, key="search_qty")

    if st.button("Add to Daily Log (from Search)"):
        food_data = nutrition_df[nutrition_df["Food"] == selected_food].to_dict(orient="records")[0]
        factor = qty / 100

        for col in ["Calories", "Protein (g)", "Carbs (g)", "Fats (g)"]:
            food_data[col] = float(food_data[col])

        adjusted_data = {
            "Food": selected_food,
            "Quantity (g)": qty,
            "Calories": food_data["Calories"] * factor,
            "Protein (g)": food_data["Protein (g)"] * factor,
            "Carbs (g)": food_data["Carbs (g)"] * factor,
            "Fats (g)": food_data["Fats (g)"] * factor
        }

        st.session_state["food_log"].append(adjusted_data)
        st.success(f"✅ {selected_food} added to your daily log!")

st.markdown("---")
st.subheader("📊 Your Daily Nutrition Log")

if st.session_state["food_log"]:
    log_df = pd.DataFrame(st.session_state["food_log"])
    st.dataframe(log_df.set_index("Food"))

    total_cal = log_df["Calories"].sum()
    total_protein = log_df["Protein (g)"].sum()
    total_carbs = log_df["Carbs (g)"].sum()
    total_fat = log_df["Fats (g)"].sum()

    st.write(f"**Total Calories:** {total_cal:.1f} kcal")
    st.write(f"**Protein:** {total_protein:.1f}g | **Carbs:** {total_carbs:.1f}g | **Fat:** {total_fat:.1f}g")

    if st.button("🗑️ Clear Today's Log"):
        st.session_state["food_log"] = []
        st.success("Daily log cleared!")
        st.rerun()
else:
    st.info("No foods logged yet today. Add foods above!")

# -------------------------------
# ⚖️ BMI Calculator, Diet & Exercise Plan
# -------------------------------
st.header("⚖️ BMI Calculator & Health Suggestions")

weight = st.number_input("Enter your weight (kg):", min_value=1, step=1)
height = st.number_input("Enter your height (cm):", min_value=50, step=1)
age = st.number_input("Enter your age:", min_value=1, step=1)
diet_type = st.selectbox("Diet Preference:", ["veg", "nonveg"])
lifestyle = st.selectbox("Lifestyle:", ["sedentary", "moderate", "active"])
condition = st.selectbox("Medical Condition:", ["none", "diabetes", "hypertension", "cholesterol"])

if st.button("Calculate BMI & Get Plan"):
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
        else:
            diet = [
                "Breakfast: Poha / idli + chutney",
                "Lunch: Balanced thali with roti, dal, sabzi",
                "Snack: Sprouts or fruit salad",
                "Dinner: Light khichdi or veg pulao"
            ]

        if condition == "diabetes":
            diet.append("⚠️ Note: Avoid sugar, include high-fiber foods like oats & vegetables.")
        elif condition == "hypertension":
            diet.append("⚠️ Note: Reduce salt, eat more leafy greens and fruits.")
        elif condition == "cholesterol":
            diet.append("⚠️ Note: Avoid fried/oily foods, include flax seeds & omega-3 rich foods.")

        st.write("\n".join([f"- {d}" for d in diet]))

        st.subheader("🏃 Suggested Exercise Plan with Videos")

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

        selected_exercises = exercise_plan.get(category, [])
        if condition != "none":
            selected_exercises += condition_exercises.get(condition, [])

        for exercise in selected_exercises:
            with st.expander(exercise["name"]):
                for video_url in exercise["videos"]:
                    st.video(video_url)

# -------------------------------
# 🤖 AI Chatbot Section
# -------------------------------
OPENROUTER_API_KEY = "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"

st.header("🤖 Nutrition Chatbot")
st.markdown("Ask me anything about food, nutrition, or diet plans!")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("You:", key="chat_input")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    messages = [
        {"role": "system", "content": "You are a friendly nutrition and diet assistant. \
        Answer questions about foods, calories, nutrients, diet plans, medical conditions, \
        and healthy lifestyle tips. Keep your answers short, clear, and practical."}
    ] + st.session_state.chat_history

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-20b",
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

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Bot:** {chat['content']}")