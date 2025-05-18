import streamlit as st
import json
import random
import re
from langchain_ollama import ChatOllama
import pandas as pd
from joblib import load


# Add basic CSS for left/right chat bubbles
st.markdown("""
<style>
.chat-bubble {
    padding: 10px 15px;
    margin: 10px;
    border-radius: 15px;
    max-width: 70%;
    display: inline-block;
    line-height: 1.4;
}

.bot-bubble {
    background-color: #f1f3f4;
    color: #333;
    text-align: left;
    float: left;
    clear: both;
}

.user-bubble {
    background-color: #0083B8;
    color: white;
    text-align: right;
    float: right;
    clear: both;
}

.welcome {
    background-color: #eef1f5;
    border-left: 6px solid #FF4B4B;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 10px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


required_features = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

feature_questions = {
    "age": ["How old are you?", "May I ask your age?", "Please share your age?"],
    "gender": ["What's your gender? (male/female)", "Could you tell me your gender?", "Mind sharing your gender?"],
    "height": ["What's your height in cm?", "How tall are you?", "Could you tell me your height?"],
    "weight": ["What's your weight in kg?", "Can you tell me your weight?", "How much do you weigh?"],
    "ap_hi": ["What is your systolic blood pressure? (X/Y, this is the X)", "Could you provide your upper blood pressure value?", "What's your systolic (high) BP?"],
    "ap_lo": ["What is your diastolic blood pressure? (X/Y, this is the Y)", "Could you share your lower blood pressure value?", "What's your diastolic (low) BP?"],
    "cholesterol": ["How is your cholesterol? (normal, above normal, well above normal)", "Can you tell me your cholesterol level?", "Is your cholesterol normal or elevated?"],
    "gluc": ["How is your glucose level? (normal, above normal, well above normal)", "Can you tell me your blood sugar level?", "Is your glucose within normal range?"],
    "smoke": ["Do you smoke?", "Any smoking habits?", "Are you a smoker?"],
    "alco": ["Do you consume alcohol?", "Do you drink alcohol?", "Any alcohol consumption?"],
    "active": ["Are you physically active?", "Do you engage in physical activity?", "Would you say you're active daily?"]
}

# Streamlit session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_data" not in st.session_state:
    st.session_state.user_data = {k: None for k in required_features}
if "last_feature" not in st.session_state:
    st.session_state.last_feature = None
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "prediction_explanation" not in st.session_state:
    st.session_state.prediction_explanation = None
if "chat_closed" not in st.session_state:
    st.session_state.chat_closed = False

llm = ChatOllama(model="gemma2:latest", temperature=0.1)

def extract_features_from_text(text):
    """Extract all possible features from free text using LLM"""
    prompt = f"""
You are a smart assistant. From the following user input, extract all health-related values for:

- age (in years, integer)
- gender (1 = female, 2 = male)
- height (in cm)
- weight (in kg)
- ap_hi (systolic BP)
- ap_lo (diastolic BP)
- cholesterol (1=normal,2=above normal,3=well above normal)
- gluc (same as cholesterol)
- smoke, alco, active (0=no,1=yes)

Handle variations like: "No I am not", "Yes I do", "I'm not", "Never", "I am very active", etc.

User says: "{text}"

Respond in valid JSON format like this:
{{
  "age": int or null,
  "gender": int or null,
  "height": int or null,
  "weight": int or null,
  "ap_hi": int or null,
  "ap_lo": int or null,
  "cholesterol": int or null,
  "gluc": int or null,
  "smoke": int or null,
  "alco": int or null,
  "active": int or null
}}
"""

    result = llm.invoke(prompt)
    txt = result.content if hasattr(result, "content") else str(result)
    try:
        match = re.search(r'\{[\s\S]*\}', txt)
        if match:
            return json.loads(match.group(0))
    except:
        return {}
    return {}

def extract_number_from_text(text, feature_type="number"):
    """Extract numeric values from text responses using LLM"""
    prompt = f"""Extract the {feature_type} from this text. 
    Only return the number without any explanation.
    If no number is found, return 'None'.
    
    Text: "{text}"
    
    {feature_type.capitalize()}:"""
    
    result = llm.invoke(prompt)
    extracted = result.content if hasattr(result, "content") else str(result)
    
    # Clean the response
    extracted = extracted.strip()
    
    # Try to convert to integer
    try:
        # Check if it's a number
        if re.match(r'^-?\d+$', extracted):
            return int(extracted)
        # Fall back to regex if LLM fails
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[0])
    except:
        pass
    
    return None

def map_value(feature, input_text):
    """Enhanced mapping function that handles more complex responses"""
    val = input_text.lower().strip()
    
    # Prioritize extracting a number first for numeric features
    if feature in ["age", "height", "weight", "ap_hi", "ap_lo"]:
        extracted_num = extract_number_from_text(input_text, feature)
        if extracted_num is not None:
            return extracted_num


    # Direct mappings for yes/no and categorical values
    YES_KEYWORDS = ["yes", "1", "true", "yeah", "yep", "sure", "i am", "i do", "i am active", "very active"]
    NO_KEYWORDS = ["no", "0", "false", "nope", "nah", "i am not", "not really", "never", "no i am not", "i don’t", "i'm not"]

    if any(k in val for k in YES_KEYWORDS):
        return 1
    if any(k in val for k in NO_KEYWORDS):
        return 0

    if val == "normal": 
        return 1
    if val == "above normal": 
        return 2
    if val == "well above normal": 
        return 3
    if val in ["female", "woman", "f"]: 
        return 1
    if val in ["male", "man", "m"]: 
        return 2

    
    # If it's a simple digit, return it
    if val.isdigit(): 
        return int(val)
    
    # If direct methods fail, try the LLM for more complex responses
    if st.session_state.last_feature:
        feature_prompt = f"""From this text, extract the value for {feature}.
        For yes/no answers, return 1 for yes and 0 for no.
        For gender, return 1 for female and 2 for male.
        For cholesterol and glucose, return:
          1 for normal
          2 for above normal
          3 for well above normal
        For numerical values, just return the number.
        Only return the extracted value without explanation.
        
        Text: "{input_text}"
        
        Extracted value:"""
        
        result = llm.invoke(feature_prompt)
        extracted = result.content if hasattr(result, "content") else str(result)
        extracted = extracted.strip()
        
        try:
            # Try to convert to integer
            if re.match(r'^-?\d+$', extracted):
                return int(extracted)
        except:
            pass
    
    return None

def ask_next_question():
    """Find the next unanswered feature and return a question for it"""
    for f, v in st.session_state.user_data.items():
        if v is None:
            q = random.choice(feature_questions[f])
            st.session_state.last_feature = f
            return q
    return None

def get_bmi_display(bmi):
    """Format BMI value and category for display"""
    if bmi is None:
        return "Unknown", "Unknown"
    bmi_val = round(bmi, 1)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return bmi_val, category

def llm_predict_cardiovascular(features, model_name="mistral:latest", temperature=0.1, top_p=0.95, repeat_penalty=1.2):
    """Predict cardiovascular disease risk using LLM and feature-based risk scoring"""
    row = features
    risk_score = 0
    
    # Calculate BMI if possible
    bmi = None
    if row.get('height') and row.get('weight') and row['height'] > 0:
        height_m = row['height'] / 100
        bmi = row['weight'] / (height_m * height_m)
    
    # Calculate risk score based on features
    if row.get('age') and row['age'] > 55:
        risk_score += 2
    if (row.get('ap_hi') and row['ap_hi'] >= 140) or (row.get('ap_lo') and row['ap_lo'] >= 90):
        risk_score += 2
    if row.get('cholesterol'):
        if row['cholesterol'] == 3:
            risk_score += 2
        elif row['cholesterol'] == 2:
            risk_score += 1
    if bmi is not None and bmi >= 30:
        risk_score += 1
    if row.get('gluc') and row['gluc'] == 3:
        risk_score += 1
    if row.get('smoke') and row['smoke'] == 1:
        risk_score += 2
    if row.get('active') and row['active'] == 0:
        risk_score += 1
    
    # Get formatted BMI value
    bmi_value, bmi_category = get_bmi_display(bmi)
    
    prompt = f"""You are a medical expert analyzing cardiovascular disease risk. 
Your task is to predict whether the patient has cardiovascular disease (0 = no, 1 = yes) based on the data provided.

Analyze the patient's full clinical profile carefully, considering all these factors:

- Age: Higher age increases cardiovascular risk significantly.
- Gender: 1 = female, 2 = male. Men typically have higher cardiovascular risk.
- Height and weight: Used to calculate BMI. BMI > 25 indicates overweight, > 30 indicates obesity.
- Blood pressure: ap_hi = systolic, ap_lo = diastolic. Elevated BP (≥140/90) is a major risk factor.
- Cholesterol: 1 = normal, 2 = above normal, 3 = well above normal. Higher levels increase risk.
- Glucose: 1 = normal, 2 = above normal, 3 = well above normal. Elevated glucose increases risk.
- Smoking: 1 = smoker, 0 = non-smoker. Smoking significantly increases cardiovascular risk.
- Alcohol intake: 1 = drinks alcohol, 0 = doesn't drink. Excessive alcohol increases risk.
- Physical activity: 1 = active, 0 = inactive. Lack of physical activity increases risk.

Additional calculated risk factors for this patient:
- BMI: {bmi_value} kg/m² (Underweight < 18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥ 30)
- Overall risk score: {risk_score}/10 (Based on key risk factors)

Follow this step-by-step reasoning process:

1. Review the patient's age - patients over 55 have significantly higher risk.
2. Assess blood pressure - hypertension (≥140/90) is a major risk factor.
3. Consider cholesterol levels - higher levels increase risk.
4. Examine BMI - obesity (BMI ≥ 30) increases risk.
5. Check other risk factors (smoking, physical inactivity, glucose).
6. Evaluate the combination of factors to determine overall risk.
7. Make your final prediction.

Patient's data:
- Age: {row.get('age', 'Unknown')} years
- Gender: {row.get('gender', 'Unknown')} (1 = female, 2 = male)
- Height: {row.get('height', 'Unknown')} cm
- Weight: {row.get('weight', 'Unknown')} kg
- Systolic BP: {row.get('ap_hi', 'Unknown')} mmHg
- Diastolic BP: {row.get('ap_lo', 'Unknown')} mmHg
- Cholesterol: {row.get('cholesterol', 'Unknown')} (1 = normal, 2 = above normal, 3 = well above normal)
- Glucose: {row.get('gluc', 'Unknown')} (1 = normal, 2 = above normal, 3 = well above normal)
- Smoking: {row.get('smoke', 'Unknown')} (0 = no, 1 = yes)
- Alcohol: {row.get('alco', 'Unknown')} (0 = no, 1 = yes)
- Physical activity: {row.get('active', 'Unknown')} (1 = active, 0 = inactive)
- BMI: {bmi_value} kg/m²
- Risk score: {risk_score}/10

Provide thorough step-by-step medical reasoning explaining your assessment. On the final line of your response, write "FINAL_PREDICTION: [0 or 1]" where 0 = no cardiovascular disease, 1 = has cardiovascular disease.
"""

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty
    )
    response = llm.invoke(prompt)
    
    # Convert AIMessage to string if needed
    response_text = response.content if hasattr(response, 'content') else str(response)

    match = re.search(r"FINAL_PREDICTION:\s*([01])", response_text)
    prediction = int(match.group(1)) if match else 0

    return prediction, response_text

def create_user_friendly_summary(prediction, explanation, features):
    """Create a user-friendly summary of the cardiovascular risk prediction"""
    # Calculate BMI if possible
    bmi = None
    if features.get('height') and features.get('weight') and features['height'] > 0:
        height_m = features['height'] / 100
        bmi = features['weight'] / (height_m ** 2)
    
    # Get formatted BMI value and category
    bmi_value, bmi_category = get_bmi_display(bmi)
    
    # Create a short summary with key points
    summary = f"**Cardiovascular Disease Risk Assessment**\n\n"
    
    if prediction == 1:
        summary += "⚠️ **Result: Higher risk of cardiovascular disease detected**\n\n"
    else:
        summary += "✅ **Result: Lower risk of cardiovascular disease detected**\n\n"
    
    summary += "**Key health metrics:**\n"
    if features.get('age'):
        summary += f"- Age: {features['age']} years\n"
    if features.get('ap_hi') and features.get('ap_lo'):
        summary += f"- Blood Pressure: {features['ap_hi']}/{features['ap_lo']} mmHg\n"
    if bmi:
        summary += f"- BMI: {bmi_value} kg/m² ({bmi_category})\n"
    if features.get('cholesterol'):
        chol_text = {1: "Normal", 2: "Above normal", 3: "Well above normal"}.get(features['cholesterol'], "Unknown")
        summary += f"- Cholesterol: {chol_text}\n"
    if features.get('gluc'):
        gluc_text = {1: "Normal", 2: "Above normal", 3: "Well above normal"}.get(features['gluc'], "Unknown")
        summary += f"- Glucose: {gluc_text}\n"
    if features.get('smoke') is not None:
        summary += f"- Smoking: {'Yes' if features['smoke'] == 1 else 'No'}\n"
    if features.get('active') is not None:
        summary += f"- Physically active: {'Yes' if features['active'] == 1 else 'No'}\n"
    
    # Extract a concise explanation from the full explanation
    key_sentences = []
    important_patterns = [
        r"(?:age|blood pressure|cholesterol|bmi|smoking|physical activity|glucose)[^.]*\.",
        r"(?:based on|considering|given|overall)[^.]*risk[^.]*\."
    ]
    
    for pattern in important_patterns:
        matches = re.findall(pattern, explanation, re.IGNORECASE)
        for match in matches:
            if len(match) > 10 and match not in key_sentences:  # Avoid very short matches
                key_sentences.append(match)
    
    if key_sentences:
        summary += "\n**Key factors in this assessment:**\n"
        for idx, sentence in enumerate(key_sentences[:3]):  # Limit to top 3 key points
            summary += f"- {sentence.strip()}\n"
    
    # Add a final closing message
    summary += "\n**Thank you for completing the cardiovascular risk assessment.**\n"
    summary += "This chat session is now closed."
    
    return summary

def display_prediction_results(pred, explanation, features):
    """Helper function to display prediction results"""
    if pred == 1:
        st.warning("⚠️ Higher risk of cardiovascular disease detected")
    else:
        st.success("✅ Lower risk of cardiovascular disease detected")

    st.metric("Prediction", "Has CVD" if pred == 1 else "No CVD")

    with st.expander("View detailed explanation"):
        st.write(explanation)

    st.subheader("Extracted Features")
    st.json(features)

    # Calculate and display BMI if height and weight are available
    bmi = None
    if features.get('height') and features.get('weight') and features['height'] > 0:
        height_m = features['height'] / 100
        bmi = features['weight'] / (height_m ** 2)
        
    # Get formatted BMI value and category
    bmi_value, bmi_category = get_bmi_display(bmi)
            
    st.subheader("Key Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("BMI", bmi_value)
    with col2:
        st.metric("Category", bmi_category)
    
    # Add a visual indicator that the chat is closed
    st.info("📝 The chat session is now closed. Thank you for completing the assessment.")

def debug_extraction(text, feature=None):
    """Helper function to show extraction results in detail (for debugging)"""
    st.subheader("Debug Extraction")
    st.write(f"Text: '{text}'")
    
    # Try direct mapping if feature is specified
    if feature:
        mapped = map_value(feature, text)
        st.write(f"Direct mapping for {feature}: {mapped}")
        
        # Show numeric extraction
        if feature in ["age", "height", "weight", "ap_hi", "ap_lo"]:
            extracted_num = extract_number_from_text(text, feature)
            st.write(f"Numeric extraction: {extracted_num}")
    
    # Try LLM extraction
    extracted = extract_features_from_text(text)
    st.write("LLM extraction:")
    st.json(extracted)

# UI starts
st.title("🩺 Chat-based Health Data Collector")

# Always show the welcome message
st.markdown("""
<div style="background-color: #eef1f5; border-left: 6px solid #FF4B4B;
            padding: 15px; margin-bottom: 20px; border-radius: 10px;
            font-size: 16px;">
<strong>Hello!</strong> Welcome to the cardiovascular risk assessment.<br>
I'll ask you some health questions to evaluate your risk of cardiovascular disease.<br>
You can start the chat when you're ready.
</div>
""", unsafe_allow_html=True)



def svm_predict_cardiovascular(features):
    model = load("svm_model.joblib")  
    df = pd.DataFrame([features])
    return int(model.predict(df)[0])

def rf_predict_cardiovascular(features):
    model = load("rf_heart_model.joblib")
    df = pd.DataFrame([features])
    return int(model.predict(df)[0])



def resolve_heart_debate(llm_pred, llm_expl, svm_pred, rf_pred, features):
    llm = ChatOllama(model="mistral:latest", temperature=0.2)

    prompt = f"""
You are a trusted medical assistant reviewing predictions for cardiovascular disease from 3 systems:

🧠 LLM Prediction: {llm_pred}
Explanation: {llm_expl}

🤖 SVM Prediction: {svm_pred}
🌲 Random Forest Prediction: {rf_pred}

Patient features:
{json.dumps(features, indent=2)}

Please compare the predictions and choose the most medically reliable.
Respond only with one word: "llm", "svm", or "rf".
"""

    result = llm.invoke(prompt)
    choice = result.content.strip().lower()
    if choice == "svm":
        return svm_pred
    elif choice == "rf":
        return rf_pred
    return llm_pred


def strip_html_tags(text):
    return re.sub(r'</?div[^>]*>', '', text)


def check_for_results_request(user_input, prediction_done):
    if prediction_done:
        lower_input = user_input.lower()
        results_keywords = ["result", "prediction", "assessment", "tell me", "what's my", "what is my", "diagnosis"]
        return any(keyword in lower_input for keyword in results_keywords)
    return False

# Show chat history with manual alignment
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"""
        <div style='text-align: right; clear: both; padding: 5px 0;'>
            <div style='display: inline-block; background-color: #0083B8; color: white;
                        padding: 10px 15px; border-radius: 15px; max-width: 70%;'>
                {msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='text-align: left; clear: both; padding: 5px 0;'>
            <div style='display: inline-block; background-color: #f1f3f4; color: black;
                        padding: 10px 15px; border-radius: 15px; max-width: 70%;'>
                {msg}
            </div>
        </div>
        """, unsafe_allow_html=True)


# Show chat input only if the chat is not closed
if not st.session_state.chat_closed:
    user_input = st.chat_input("Your message...")
    if user_input:
        # Render the user's message immediately and right-aligned
        st.markdown(f"""
        <div style='text-align: right; clear: both; padding: 5px 0;'>
            <div style='display: inline-block; background-color: #0083B8; color: white;
                        padding: 10px 15px; border-radius: 15px; max-width: 70%;'>
                {user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Now append to chat history so it shows up correctly on next rerun
        st.session_state.chat_history.append(("user", user_input))

        # Check if user is asking for results and we already have them
        if check_for_results_request(user_input, st.session_state.prediction_done):
            if st.session_state.prediction_results is not None:
                user_friendly_summary = create_user_friendly_summary(
                    st.session_state.prediction_results, 
                    st.session_state.prediction_explanation, 
                    st.session_state.user_data
                )
                st.session_state.chat_history.append(("bot", user_friendly_summary))
                st.chat_message("bot").markdown(user_friendly_summary)
                # Close the chat after showing results
                st.session_state.chat_closed = True
                # Force a page refresh to update the UI
                st.rerun()
            else:
                st.session_state.chat_history.append(("bot", "I'm sorry, I don't have your prediction results yet. Let's complete the health assessment first."))
                st.chat_message("bot").markdown("I'm sorry, I don't have your prediction results yet. Let's complete the health assessment first.")
        else:
            updated = False
            last = st.session_state.last_feature
            cleaned = user_input.strip().lower()
            
            # Try short direct mapping first
            if last and st.session_state.user_data[last] is None:
                mapped = map_value(last, cleaned)
                if mapped is not None:
                    st.session_state.user_data[last] = mapped
                    st.session_state.last_feature = None
                    updated = True

            # If direct mapping failed, try extraction from text
            if not updated:
                extracted = extract_features_from_text(user_input)
                for k, v in extracted.items():
                    if v is not None and st.session_state.user_data[k] is None:
                        st.session_state.user_data[k] = v
                        updated = True

            # Bot response
            if updated:
                next_q = ask_next_question()
                if next_q:
                    st.session_state.chat_history.append(("bot", next_q))
                    st.chat_message("bot").markdown(next_q)
                else:
                    # Store the prediction in session state
                    pred, explanation = llm_predict_cardiovascular(st.session_state.user_data)
                    svm_pred = svm_predict_cardiovascular(st.session_state.user_data)
                    rf_pred = rf_predict_cardiovascular(st.session_state.user_data)

                    final_pred = resolve_heart_debate(pred, explanation, svm_pred, rf_pred, st.session_state.user_data)

                    with st.expander("🔍 Model Predictions"):
                        st.markdown(f"- **LLM Prediction:** {pred}")
                        st.markdown(f"- **SVM Prediction:** {svm_pred}")
                        st.markdown(f"- **Random Forest Prediction:** {rf_pred}")
                        st.markdown(f"- **Final Decision:** {final_pred}")

                    st.session_state.prediction_done = True
                    st.session_state.prediction_results = final_pred
                    st.session_state.prediction_explanation = explanation

                    st.session_state.prediction_done = True
                    st.session_state.prediction_results = final_pred
                    st.session_state.prediction_explanation = explanation

                    # Create user-friendly summary
                    user_friendly_summary = create_user_friendly_summary(final_pred, explanation, st.session_state.user_data)

                    # Append model prediction section
                    model_summary = "\n\n**📊 Model Prediction Comparison**\n"
                    model_summary += f"- LLM Prediction: {pred}\n"
                    model_summary += f"- SVM Prediction: {svm_pred}\n"
                    model_summary += f"- Random Forest Prediction: {rf_pred}\n"
                    model_summary += f"- Final Decision: {final_pred}\n"

                    full_summary = user_friendly_summary + model_summary

                    # Display prediction
                    display_prediction_results(final_pred, explanation, st.session_state.user_data)

                    # Send full summary to chat
                    clean_summary = strip_html_tags(full_summary)
                    st.session_state.chat_history.append(("bot", clean_summary))
                    st.chat_message("bot").markdown(clean_summary)

                    # Close the chat after showing results
                    st.session_state.chat_closed = True
                    # Force a page refresh to update the UI
                    st.rerun()
            else:
                # If no features were extracted, just continue the conversation
                reply = llm.invoke(user_input)
                content = reply.content if hasattr(reply, "content") else str(reply)
                st.session_state.chat_history.append(("bot", content))
                st.chat_message("bot").markdown(content)

                # Continue asking
                next_q = ask_next_question()
                if next_q:
                    st.session_state.chat_history.append(("bot", next_q))
                    st.chat_message("bot").markdown(next_q)
else:
    # Show a message that the chat is closed
    st.info("📝 The assessment is complete and the chat session is now closed.")



# Display parsed features on the left
with st.sidebar:
    st.markdown("### 🧾 Extracted Data So Far")
    for feature, value in st.session_state.user_data.items():
        if value is not None:
            st.write(f"**{feature.capitalize()}**: {value}")

# Place this near the bottom of your app, outside of conditional blocks
with st.sidebar:
    if st.button("🔄 Start New Conversation"):
        for key in [
            "chat_history", "user_data", "last_feature", 
            "prediction_done", "prediction_results", 
            "prediction_explanation", "chat_closed"
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Add return button to go back to main complaint entry
with st.sidebar:
    if st.button("🏠 Return to Health Triage Home"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("app.py")




