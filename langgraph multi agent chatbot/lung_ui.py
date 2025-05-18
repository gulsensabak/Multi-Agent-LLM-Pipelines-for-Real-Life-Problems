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
    "gender", "age", "smoking", "yellow_fingers", "anxiety", 
    "peer_pressure", "chronic_disease", "fatigue", "allergy", 
    "wheezing", "alcohol_consuming", "coughing", 
    "shortness_of_breath", "swallowing_difficulty", "chest_pain"
]

feature_questions = {
    "gender": ["What's your gender? (male/female)", "Could you tell me your gender?", "Please share your gender (M/F)?"],
    "age": ["How old are you?", "May I ask your age?", "Please share your age?"],
    "smoking": ["Do you smoke?", "Any smoking habits?", "Are you a smoker?"],
    "yellow_fingers": ["Do you have yellow fingers?", "Have you noticed any yellowing of your fingers?", "Do your fingers have a yellow tint?"],
    "anxiety": ["Do you suffer from anxiety?", "Would you say you experience anxiety regularly?", "Have you been diagnosed with or experience anxiety?"],
    "peer_pressure": ["Do you feel you're under peer pressure?", "Are you experiencing peer pressure?", "Would you say peer pressure affects your choices?"],
    "chronic_disease": ["Do you have any chronic diseases?", "Have you been diagnosed with any chronic conditions?", "Do you suffer from any long-term health issues?"],
    "fatigue": ["Do you experience fatigue regularly?", "Do you often feel tired or fatigued?", "Would you say you suffer from fatigue?"],
    "allergy": ["Do you have any allergies?", "Do you suffer from allergic reactions?", "Have you been diagnosed with any allergies?"],
    "wheezing": ["Do you experience wheezing?", "Have you noticed any wheezing when you breathe?", "Do you wheeze when breathing?"],
    "alcohol_consuming": ["Do you consume alcohol?", "Do you drink alcohol?", "How would you describe your alcohol consumption?"],
    "coughing": ["Do you cough regularly?", "Have you been experiencing persistent coughing?", "Do you have a cough that won't go away?"],
    "shortness_of_breath": ["Do you experience shortness of breath?", "Do you have difficulty breathing sometimes?", "Have you noticed getting out of breath easily?"],
    "swallowing_difficulty": ["Do you have difficulty swallowing?", "Have you experienced any problems with swallowing?", "Is swallowing difficult for you?"],
    "chest_pain": ["Do you experience chest pain?", "Have you had any pain in your chest area?", "Do you suffer from chest discomfort or pain?"]
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

- gender (M or F)
- age (in years, integer)
- smoking (0=no, 1=yes)
- yellow_fingers (0=no, 1=yes)
- anxiety (0=no, 1=yes)
- peer_pressure (0=no, 1=yes)
- chronic_disease (0=no, 1=yes)
- fatigue (0=no, 1=yes)
- allergy (0=no, 1=yes)
- wheezing (0=no, 1=yes)
- alcohol_consuming (0=no, 1=yes)
- coughing (0=no, 1=yes)
- shortness_of_breath (0=no, 1=yes)
- swallowing_difficulty (0=no, 1=yes)
- chest_pain (0=no, 1=yes)

Handle variations like: "No I am not", "Yes I do", "I'm not", "Never", "I get tired easily", etc. If gives another value, try to convert it to 0 and 1 according to meaning of the sentence user provided to you.

User says: "{text}"

Respond in valid JSON format like this:
{{
  "gender": "M" or "F" or null,
  "age": int or null,
  "smoking": int or null,
  "yellow_fingers": int or null,
  "anxiety": int or null,
  "peer_pressure": int or null,
  "chronic_disease": int or null,
  "fatigue": int or null,
  "allergy": int or null,
  "wheezing": int or null,
  "alcohol_consuming": int or null,
  "coughing": int or null,
  "shortness_of_breath": int or null,
  "swallowing_difficulty": int or null,
  "chest_pain": int or null
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
    
    # Prioritize extracting a number first for age
    if feature == "age":
        extracted_num = extract_number_from_text(input_text, feature)
        if extracted_num is not None:
            return extracted_num

    # Direct mappings for yes/no and categorical values
    YES_KEYWORDS = ["yes", "1", "true", "yeah", "yep", "sure", "i am", "i do", "i have", 
                   "very", "often", "regularly", "sometimes", "most of the time", 
                   "always", "i get", "i feel", "i suffer", "i experience", "i have been"]
    NO_KEYWORDS = [
    "no", "0", "false", "nope", "nah", "i am not", "not really", "never", 
    "no i am not", "i don't", "i'm not", "don't have", "i don't have",
    "i don't feel", "i don't get", "i don't experience", "rarely", "seldom", 
    "occasionally", "i have not", "i do not feel", "i do not experience", 
    "i have never", "i have never felt", "i have never experienced",
    "i do not suffer", "i dont suffer", "i don't suffer", "i do not have",
    "i dont have", "not at all", "absolutely not", "definitely not",
    "no i don't", "no i do not", "no i dont", "no i never", "no never",
    "not suffering", "not feeling", "not experiencing", "not getting"
]
    
    # Check for negative patterns first to avoid false positives
    if feature != "gender" and feature != "age":
        # More thorough negative checking
        if any(k in val for k in NO_KEYWORDS):
            return 0
        # Check for positive patterns only if no negative was found
        if any(k in val for k in YES_KEYWORDS):
            return 1
    
    if feature == "gender":
        if val in ["male", "man", "m", "boy"]:
            return "M"
        if val in ["female", "woman", "f", "girl"]:
            return "F"
    
    # If it's a simple digit, return it
    if val.isdigit(): 
        return int(val)
    
    # If direct methods fail, try the LLM for more complex responses
    if st.session_state.last_feature:
        feature_prompt = f"""From this text, extract the value for {feature}.
        For gender, return 'M' for male and 'F' for female.
        For age, return the age as a number.
        For all other features (smoking, yellow_fingers, etc.), return:
          1 for yes/present/true
          0 for no/absent/false
        Only return the extracted value without explanation.
        
        Text: "{input_text}"
        
        Extracted value:"""
        
        result = llm.invoke(feature_prompt)
        extracted = result.content if hasattr(result, "content") else str(result)
        extracted = extracted.strip()
        
        try:
            # Check if it's a gender value
            if feature == "gender" and extracted.upper() in ["M", "F"]:
                return extracted.upper()
            
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

def llm_predict_lung_cancer(features, model_name="mistral:latest", temperature=0.1, top_p=0.95, repeat_penalty=1.2):
    """Predict lung cancer risk using LLM and feature-based risk scoring"""
    row = features
    risk_score = 0
    
    # Calculate risk score based on features
    # The weights are assigned based on typical importance of features
    if row.get('age') and row['age'] > 60:
        risk_score += 2
        
    if row.get('smoking') == 1:
        risk_score += 3  # Smoking is a major risk factor
        
    if row.get('yellow_fingers') == 1:
        risk_score += 1  # Associated with heavy smoking
        
    if row.get('coughing') == 1:
        risk_score += 2  # Key respiratory symptom
        
    if row.get('shortness_of_breath') == 1:
        risk_score += 2  # Key respiratory symptom
        
    if row.get('wheezing') == 1:
        risk_score += 1  # Respiratory symptom
        
    if row.get('chest_pain') == 1:
        risk_score += 2  # Serious symptom
        
    if row.get('fatigue') == 1:
        risk_score += 1  # General symptom
        
    if row.get('swallowing_difficulty') == 1:
        risk_score += 1  # Can indicate advanced disease
    
    prompt = f"""You are a medical expert analyzing lung cancer risk. 
Your task is to predict whether the patient has lung cancer (0 = no, 1 = yes) based on the data provided.

Analyze the patient's full clinical profile carefully, considering all these factors:

- Gender: There can be differences in lung cancer risk between males and females.
- Age: Higher age increases lung cancer risk significantly.
- Smoking: One of the strongest risk factors for lung cancer.
- Yellow Fingers: Can indicate heavy smoking and tar buildup.
- Anxiety: May be related to health concerns or affect reporting.
- Peer Pressure: May influence health behaviors like smoking.
- Chronic Disease: Pre-existing conditions may increase vulnerability.
- Fatigue: Common symptom in cancer patients.
- Allergy: Respiratory allergies may mask or complicate diagnosis.
- Wheezing: Respiratory symptom that may indicate airway problems.
- Alcohol Consuming: May interact with other risk factors.
- Coughing: Key symptom of lung cancer, especially if persistent.
- Shortness of Breath: Important symptom of lung problems.
- Swallowing Difficulty: Can indicate tumor affecting esophagus.
- Chest Pain: Serious symptom that may indicate advanced disease.

Additional calculated risk factors for this patient:
- Overall risk score: {risk_score}/15 (Based on key risk factors)

Follow this step-by-step reasoning process:

1. Review the patient's demographic factors (age, gender).
2. Assess smoking status - the primary risk factor for lung cancer.
3. Consider respiratory symptoms (coughing, wheezing, shortness of breath).
4. Evaluate other physical symptoms (chest pain, swallowing difficulty, fatigue).
5. Consider behavioral factors (alcohol consumption, response to peer pressure).
6. Assess the combination of risk factors to determine overall risk.
7. Make your final prediction.

Patient's data:
- Gender: {row.get('gender', 'Unknown')}
- Age: {row.get('age', 'Unknown')} years
- Smoking: {row.get('smoking', 'Unknown')} (0=no, 1=yes)
- Yellow Fingers: {row.get('yellow_fingers', 'Unknown')} (0=no, 1=yes)
- Anxiety: {row.get('anxiety', 'Unknown')} (0=no, 1=yes)
- Peer Pressure: {row.get('peer_pressure', 'Unknown')} (0=no, 1=yes)
- Chronic Disease: {row.get('chronic_disease', 'Unknown')} (0=no, 1=yes)
- Fatigue: {row.get('fatigue', 'Unknown')} (0=no, 1=yes)
- Allergy: {row.get('allergy', 'Unknown')} (0=no, 1=yes)
- Wheezing: {row.get('wheezing', 'Unknown')} (0=no, 1=yes)
- Alcohol Consuming: {row.get('alcohol_consuming', 'Unknown')} (0=no, 1=yes)
- Coughing: {row.get('coughing', 'Unknown')} (0=no, 1=yes)
- Shortness of Breath: {row.get('shortness_of_breath', 'Unknown')} (0=no, 1=yes)
- Swallowing Difficulty: {row.get('swallowing_difficulty', 'Unknown')} (0=no, 1=yes)
- Chest Pain: {row.get('chest_pain', 'Unknown')} (0=no, 1=yes)
- Risk score: {risk_score}/15

Provide thorough step-by-step medical reasoning explaining your assessment. On the final line of your response, write "FINAL_PREDICTION: [0 or 1]" where 0 = no lung cancer, 1 = potential lung cancer.
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


def svm_predict_lung(features):
    model = load("svm_lung_model.joblib")
    features = features.copy()

    # GENDER corrected (string to number)
    if isinstance(features.get("gender"), str):
        gender_val = features["gender"].lower()
        if gender_val == "m":
            features["gender"] = 1
        elif gender_val == "f":
            features["gender"] = 0

    df = pd.DataFrame([features])
    return int(model.predict(df)[0])



def rf_predict_lung(features):
    model = load("rf_lung_model.joblib")
    features = features.copy()

    # GENDER corrected (string to number)
    if isinstance(features.get("gender"), str):
        gender_val = features["gender"].lower()
        if gender_val == "m":
            features["gender"] = 1
        elif gender_val == "f":
            features["gender"] = 0

    df = pd.DataFrame([features])
    return int(model.predict(df)[0])



def resolve_lung_debate(llm_pred, llm_expl, svm_pred, rf_pred, features):
    llm = ChatOllama(model="mistral:latest", temperature=0.2)

    prompt = f"""
You are a trusted medical assistant reviewing predictions for lung cancer from 3 systems:

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



def create_user_friendly_summary(prediction, explanation, features):
    """Create a user-friendly summary of the lung cancer risk prediction"""
    # Create a short summary with key points
    summary = f"*Lung Cancer Risk Assessment*\n\n"
    
    if prediction == 1:
        summary += "⚠ *Result: Higher risk of lung cancer detected*\n\n"
    else:
        summary += "✅ *Result: Lower risk of lung cancer detected*\n\n"
    
    summary += "*Key health factors:*\n"
    if features.get('gender'):
        summary += f"- Gender: {'Male' if features['gender'] == 'M' else 'Female'}\n"
    if features.get('age'):
        summary += f"- Age: {features['age']} years\n"
    if features.get('smoking') is not None:
        summary += f"- Smoking: {'Yes' if features['smoking'] == 1 else 'No'}\n"
    
    # Extract major symptoms
    symptoms = []
    if features.get('coughing') == 1:
        symptoms.append("Coughing")
    if features.get('shortness_of_breath') == 1:
        symptoms.append("Shortness of breath")
    if features.get('wheezing') == 1:
        symptoms.append("Wheezing")
    if features.get('chest_pain') == 1:
        symptoms.append("Chest pain")
    if features.get('fatigue') == 1:
        symptoms.append("Fatigue")
    if features.get('swallowing_difficulty') == 1:
        symptoms.append("Swallowing difficulty")
    
    if symptoms:
        summary += f"- Key symptoms: {', '.join(symptoms)}\n"
    
    # Extract a concise explanation from the full explanation
    key_sentences = []
    important_patterns = [
        r"(?:smoking|age|respiratory|symptoms|risk)[^.]*\.",
        r"(?:based on|considering|given|overall)[^.]risk[^.]\."
    ]
    
    for pattern in important_patterns:
        matches = re.findall(pattern, explanation, re.IGNORECASE)
        for match in matches:
            if len(match) > 10 and match not in key_sentences:  # Avoid very short matches
                key_sentences.append(match)
    
    if key_sentences:
        summary += "\n*Key factors in this assessment:*\n"
        for idx, sentence in enumerate(key_sentences[:3]):  # Limit to top 3 key points
            summary += f"- {sentence.strip()}\n"
    
    # Add disclaimer
    summary += "\n**Important**: This assessment is for informational purposes only and should not be considered a medical diagnosis. Please consult with a healthcare professional for proper evaluation and diagnosis.\n"
    
    # Add a final closing message
    summary += "\n*Thank you for completing the lung cancer risk assessment.*\n"
    summary += "This chat session is now closed."
    
    return summary

def display_prediction_results(pred, explanation, features):
    """Helper function to display prediction results"""
    if pred == 1:
        st.warning("⚠ Higher risk of lung cancer detected")
    else:
        st.success("✅ Lower risk of lung cancer detected")

    st.metric("Prediction", "Higher Risk" if pred == 1 else "Lower Risk")

    with st.expander("View detailed explanation"):
        st.write(explanation)

    st.subheader("Extracted Features")
    st.json(features)
            
    st.subheader("Key Risk Factors")
    
    # Display smoking status prominently
    col1, col2 = st.columns(2)
    with col1:
        smoking_status = "Yes" if features.get('smoking') == 1 else "No" if features.get('smoking') == 0 else "Unknown"
        st.metric("Smoking", smoking_status)
    with col2:
        age = features.get('age', "Unknown")
        st.metric("Age", age)
    
    # Add a visual indicator that the chat is closed
    st.info("📝 The chat session is now closed. Thank you for completing the assessment.")

def strip_html_tags(text):
    return re.sub(r'</?div[^>]*>', '', text)


def debug_extraction(text, feature=None):
    """Helper function to show extraction results in detail (for debugging)"""
    st.subheader("Debug Extraction")
    st.write(f"Text: '{text}'")
    
    # Try direct mapping if feature is specified
    if feature:
        mapped = map_value(feature, text)
        st.write(f"Direct mapping for {feature}: {mapped}")
        
        # Show numeric extraction
        if feature == "age":
            extracted_num = extract_number_from_text(text, feature)
            st.write(f"Numeric extraction: {extracted_num}")
    
    # Try LLM extraction
    extracted = extract_features_from_text(text)
    st.write("LLM extraction:")
    st.json(extracted)

# UI starts
st.title("🫁 Chat-based Lung Cancer Risk Assessment")

# Always show the welcome message
st.markdown("""
<div style="background-color: #eef1f5; border-left: 6px solid #FF4B4B;
            padding: 15px; margin-bottom: 20px; border-radius: 10px;
            font-size: 16px;">
<strong>Hello!</strong> Welcome to the lung cancer risk assessment.<br>
I'll ask you some health questions to evaluate your risk of lung cancer.<br>
You can start the chat when you're ready.
</div>
""", unsafe_allow_html=True)

# Check if user is asking for results and we have them
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
                    pred, explanation = llm_predict_lung_cancer(st.session_state.user_data)
                    svm_pred = svm_predict_lung(st.session_state.user_data)
                    rf_pred = rf_predict_lung(st.session_state.user_data)

                    final_pred = resolve_lung_debate(pred, explanation, svm_pred, rf_pred, st.session_state.user_data)

                    with st.expander("🔍 Model Predictions"):
                        st.markdown(f"- **LLM Prediction:** {pred}")
                        st.markdown(f"- **SVM Prediction:** {svm_pred}")
                        st.markdown(f"- **Random Forest Prediction:** {rf_pred}")
                        st.markdown(f"- **Final Decision:** {final_pred}")

                    st.session_state.prediction_done = True
                    st.session_state.prediction_results = final_pred
                    st.session_state.prediction_explanation = explanation
                                       
                    # Create a user-friendly summary
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

# Display parsed features on the left (sidebar)
with st.sidebar:
    st.markdown("### 🧾 Extracted Data So Far")
    for feature, value in st.session_state.user_data.items():
        if value is not None:
            if feature == "gender":
                display_value = "Male" if value == "M" else "Female"
            elif feature in ["smoking", "yellow_fingers", "anxiety", "peer_pressure", 
                           "chronic_disease", "fatigue", "allergy", "wheezing",
                           "alcohol_consuming", "coughing", "shortness_of_breath",
                           "swallowing_difficulty", "chest_pain"]:
                display_value = "Yes" if value == 1 else "No"
            else:
                display_value = value
            st.write(f"{feature.replace('_', ' ').capitalize()}: {display_value}")

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




