import streamlit as st
import json
import random
import re
from langchain_ollama import ChatOllama
from tool_loader import svm_heart_predictor, rf_heart_predictor



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

# Feature descriptions for better understanding
feature_descriptions = {
    "age": "age in years (exact number)",
    "gender": "gender (1 for female, 2 for male)",
    "height": "height in centimeters",
    "weight": "weight in kilograms", 
    "ap_hi": "systolic blood pressure (the higher number in BP reading)",
    "ap_lo": "diastolic blood pressure (the lower number in BP reading)",
    "cholesterol": "cholesterol level (1=normal, 2=above normal, 3=well above normal)",
    "gluc": "glucose/blood sugar level (1=normal, 2=above normal, 3=well above normal)",
    "smoke": "smoking status (0=no, 1=yes)",
    "alco": "alcohol consumption (0=no, 1=yes)",
    "active": "physical activity level (0=inactive, 1=active)"
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

class HealthDataExtractionAgent:
    """Enhanced agent for extracting health data with confidence scoring"""
    
    def __init__(self, llm):
        self.llm = llm
        self.required_features = required_features
        self.feature_descriptions = feature_descriptions

    def parse_user_response_manually(self, user_response):
        """
        Enhanced manual extraction for health features in cardiovascular risk assessment.
        """
        response_lower = user_response.lower()
        extracted_data = {}

        # Gender
        if any(word in response_lower for word in ['male', 'man', 'guy']):
            extracted_data['gender'] = {'value': 2, 'confidence': 9}
        if any(word in response_lower for word in ['female', 'woman', 'lady']):
            extracted_data['gender'] = {'value': 1, 'confidence': 9}

        # Age
        age_patterns = [
            r"i'?m\s+(\d+)",  # I'm 45
            r"(\d+)\s+years?\s+old",  # 45 years old
            r"age\s+is\s+(\d+)",  # age is 45
            r"(\d+)\s*yo",  # 45yo
            r"in\s+my\s+(\d+)'?s",  # in my 40s
        ]
        for pattern in age_patterns:
            match = re.search(pattern, response_lower)
            if match:
                age = int(match.group(1))
                if 10 <= age <= 120:
                    extracted_data['age'] = {'value': age, 'confidence': 9}
                    break

        # Height
        match = re.search(r'(\d+)\s*(cm|centimeters?)', response_lower)
        if match:
            height = int(match.group(1))
            if 100 <= height <= 250:
                extracted_data['height'] = {'value': height, 'confidence': 9}

        # Weight
        match = re.search(r'(\d+)\s*(kg|kilograms?)', response_lower)
        if match:
            weight = int(match.group(1))
            if 30 <= weight <= 300:
                extracted_data['weight'] = {'value': weight, 'confidence': 9}

        # Blood pressure
        if "blood pressure" in response_lower or "bp" in response_lower:
            bp_match = re.search(r'(\d{2,3})\s*(over|\/)\s*(\d{2,3})', response_lower)
            if bp_match:
                extracted_data['ap_hi'] = {'value': int(bp_match.group(1)), 'confidence': 9}
                extracted_data['ap_lo'] = {'value': int(bp_match.group(3)), 'confidence': 9}

        # Smoking
        if re.search(r"\b(don'?t|do not|never|non[-\s]?smoker|quit)\s*(smoke|smoking)\b", response_lower):
            extracted_data['smoke'] = {'value': 0, 'confidence': 9}
        elif re.search(r"\b(smoke|smoking|smoker)\b", response_lower):
            extracted_data['smoke'] = {'value': 1, 'confidence': 8}

        # Alcohol
        if re.search(r"\b(don'?t|do not|no)\s+(drink|consume).*alcohol", response_lower) or \
        re.search(r"\b(non[-\s]?drinker|no\s+alcohol)\b", response_lower):
            extracted_data['alco'] = {'value': 0, 'confidence': 9}
        elif "alcohol" in response_lower or "drink" in response_lower:
            extracted_data['alco'] = {'value': 1, 'confidence': 8}

        # Physical activity
        if re.search(r"not very (physically )?active", response_lower) or \
        re.search(r"(physically )?inactive|sedentary", response_lower):
            extracted_data['active'] = {'value': 0, 'confidence': 9}
        elif "physically active" in response_lower or "exercise" in response_lower or "workout" in response_lower:
            extracted_data['active'] = {'value': 1, 'confidence': 8}

        # Cholesterol
        if re.search(r"cholesterol.*(well above normal|very high)", response_lower):
            extracted_data['cholesterol'] = {'value': 3, 'confidence': 8}
        elif re.search(r"cholesterol.*(above normal|slightly high)", response_lower):
            extracted_data['cholesterol'] = {'value': 2, 'confidence': 8}
        elif re.search(r"cholesterol.*(normal|within (the )?normal)", response_lower):
            extracted_data['cholesterol'] = {'value': 1, 'confidence': 8}

        # Glucose
        if re.search(r"(glucose|blood sugar).*well above normal|very high", response_lower):
            extracted_data['gluc'] = {'value': 3, 'confidence': 8}
        elif re.search(r"(glucose|blood sugar).*above normal|slightly high", response_lower):
            extracted_data['gluc'] = {'value': 2, 'confidence': 8}
        elif re.search(r"(glucose|blood sugar).*(normal|within (the )?normal)", response_lower):
            extracted_data['gluc'] = {'value': 1, 'confidence': 8}

        return extracted_data
    
    def _llm_extract_data(self, question_asked, user_response, expected_feature=None):
        """
        Main extraction method that analyzes user response and extracts data with confidence
        
        """
        
        prompt = f"""
You are an intelligent health data extraction agent. Your task is to analyze a user's response to a health question and extract relevant health data.

CONTEXT:
- Question asked: "{question_asked}"
- User's response: "{user_response}"
- Expected feature: {expected_feature if expected_feature else "Any health feature"}

HEALTH FEATURES TO EXTRACT:
{json.dumps(self.feature_descriptions, indent=2)}

EXTRACTION RULES:
1. Extract ANY health-related data mentioned, even if it doesn't match the expected feature
2. Assign confidence scores (1-10) based on clarity and specificity
3. For vague responses like "in my 50s", "around 40", "late 60s" - assign confidence ≤ 5
4. For exact numbers or clear categorical answers - assign confidence ≥ 7
5. Do NOT guess or invent data
6. Handle variations like "No I don't", "Yes I do", "I'm very active", etc.

SPECIAL ATTENTION FOR COMMON EXPRESSIONS:
- "I am active" / "I'm active" / "Yes I am" (for activity) → active = 1, confidence 8-9
- "I am not active" / "I'm not active" / "No I'm not" (for activity) → active = 0, confidence 8-9
- "I smoke" / "Yes I smoke" / "I'm a smoker" → smoke = 1, confidence 8-9
- "I don't smoke" / "No I don't" / "Non-smoker" → smoke = 0, confidence 8-9
- "I drink" / "Yes I drink" / "Social drinker" → alco = 1, confidence 7-8
- "I don't drink" / "No alcohol" / "Teetotaler" → alco = 0, confidence 8-9
- "Normal" / "Fine" / "Good" (for cholesterol/glucose) → value = 1, confidence 7-8
- "High" / "Elevated" (for cholesterol/glucose) → value = 2 or 3, confidence 7-8
- "Male" / "Man" / "Guy" → gender = 2, confidence 9-10
- "Female" / "Woman" / "Lady" → gender = 1, confidence 9-10

CONFIDENCE SCORING:
- 9-10: Exact, clear, unambiguous data (e.g., "I am 45 years old", "120/80")
- 7-8: Clear but slightly ambiguous (e.g., "I'm 45", "normal cholesterol")
- 5-6: Somewhat clear but needs confirmation (e.g., "around 45", "pretty active")
- 3-4: Vague or unclear (e.g., "in my 40s", "sometimes active")
- 1-2: Very vague or unclear (e.g., "middle-aged", "not really sure")

Return ONLY valid JSON in this exact format:
{{
  "extracted_features": {{
    "age": {{"value": int_or_null, "confidence": int_1_to_10}},
    "gender": {{"value": int_or_null, "confidence": int_1_to_10}},
    "height": {{"value": int_or_null, "confidence": int_1_to_10}},
    "weight": {{"value": int_or_null, "confidence": int_1_to_10}},
    "ap_hi": {{"value": int_or_null, "confidence": int_1_to_10}},
    "ap_lo": {{"value": int_or_null, "confidence": int_1_to_10}},
    "cholesterol": {{"value": int_or_null, "confidence": int_1_to_10}},
    "gluc": {{"value": int_or_null, "confidence": int_1_to_10}},
    "smoke": {{"value": int_or_null, "confidence": int_1_to_10}},
    "alco": {{"value": int_or_null, "confidence": int_1_to_10}},
    "active": {{"value": int_or_null, "confidence": int_1_to_10}}
  }},
  "analysis": {{
    "expected_feature_provided": boolean,
    "other_features_provided": [list_of_feature_names],
    "needs_clarification": boolean,
    "clarification_reason": "string_explanation"
  }}
}}
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                return self._get_fallback_result()
                
        except Exception as e:
            print(f"Extraction error: {e}")
            return self._get_fallback_result()
    

    
    def extract_data_with_confidence(self, question_asked, user_response, expected_feature=None):
        """
        Enhanced extraction method that combines manual parsing with LLM analysis
        """
        
        # First try manual parsing
        manual_extraction = self.parse_user_response_manually(user_response)
        
        # Create the result structure
        result = {
            "extracted_features": {feature: {"value": None, "confidence": 0} for feature in self.required_features},
            "analysis": {
                "expected_feature_provided": False,
                "other_features_provided": [],
                "needs_clarification": False,
                "clarification_reason": ""
            }
        }
        
        # Fill in manually extracted data
        for feature, data in manual_extraction.items():
            if feature in result["extracted_features"]:
                result["extracted_features"][feature] = data
                result["analysis"]["other_features_provided"].append(feature)
        
        # Check if expected feature was provided
        if expected_feature and expected_feature in manual_extraction:
            result["analysis"]["expected_feature_provided"] = True
        
        # If no data was extracted manually, try LLM
        if not manual_extraction:
            try:
                llm_result = self._llm_extract_data(question_asked, user_response, expected_feature)
                if llm_result:
                    result = llm_result
            except Exception as e:
                print(f"LLM extraction failed: {e}")
                result["analysis"]["needs_clarification"] = True
                result["analysis"]["clarification_reason"] = "Could not understand the response clearly"
        
        return result
    def _get_fallback_result(self):
        """Fallback result structure when extraction fails"""
        return {
            "extracted_features": {feature: {"value": None, "confidence": 0} for feature in self.required_features},
            "analysis": {
                "expected_feature_provided": False,
                "other_features_provided": [],
                "needs_clarification": True,
                "clarification_reason": "Could not understand the response clearly"
            }
        }
    
    def generate_response(self, extraction_result, expected_feature):
        """Generate appropriate bot response based on extraction results"""
        
        analysis = extraction_result["analysis"]
        extracted_features = extraction_result["extracted_features"]
        
        # Check if expected feature was provided with high confidence
        if expected_feature and extracted_features[expected_feature]["value"] is not None:
            confidence = extracted_features[expected_feature]["confidence"]
            
            if confidence >= 7:
                # High confidence - accept the data and move to next question
                return self._accept_data_and_continue(expected_feature, extracted_features)
            elif confidence >= 4:
                # Medium confidence - ask for clarification
                return self._ask_for_clarification(expected_feature, "partially_understood")
            else:
                # Low confidence - ask to repeat
                return self._ask_for_clarification(expected_feature, "not_understood")
        
        # Check if other features were provided with high confidence
        other_high_confidence = []
        for feature, data in extracted_features.items():
            if data["value"] is not None and data["confidence"] >= 7 and feature != expected_feature:
                other_high_confidence.append(feature)
        
        if other_high_confidence:
            # User provided other data - acknowledge and redirect
            return self._acknowledge_other_data_and_redirect(other_high_confidence, expected_feature, extracted_features)
        
        # Default case - ask for clarification
        return self._ask_for_clarification(expected_feature, analysis.get("clarification_reason", "not_clear"))
    
    def _accept_data_and_continue(self, feature, extracted_features):
        """Accept the data and continue to next question"""
        value = extracted_features[feature]["value"]
        st.session_state.user_data[feature] = value
        
        # Find next question
        next_feature = self._get_next_missing_feature()
        if next_feature:
            st.session_state.last_feature = next_feature
            question = random.choice(feature_questions[next_feature])
            return f"Got it! Now, {question.lower()}"
        else:
            # All data collected - proceed to prediction
            return self._initiate_prediction()
    
    def _acknowledge_other_data_and_redirect(self, other_features, expected_feature, extracted_features):
        """Acknowledge other data provided and redirect to expected question"""
        
        # Save the other data provided
        for feature in other_features:
            value = extracted_features[feature]["value"]
            st.session_state.user_data[feature] = value
        
        # Create acknowledgment message
        feature_names = [feature.replace('_', ' ') for feature in other_features]
        if len(feature_names) == 1:
            ack_msg = f"I noted your {feature_names[0]} information."
        else:
            ack_msg = f"I noted your {', '.join(feature_names)} information."
        
        # Redirect to expected question
        if expected_feature:
            question = random.choice(feature_questions[expected_feature])
            return f"{ack_msg} But I still need to know: {question.lower()}"
        else:
            next_feature = self._get_next_missing_feature()
            if next_feature:
                st.session_state.last_feature = next_feature
                question = random.choice(feature_questions[next_feature])
                return f"{ack_msg} Now, {question.lower()}"
            else:
                return self._initiate_prediction()
    
    def _ask_for_clarification(self, feature, reason):
        """Ask for clarification on the expected feature"""
        
        if not feature:
            return "I didn't quite understand. Could you please provide more specific information?"
        
        clarification_messages = {
            "not_understood": [
                f"I couldn't clearly understand your {feature.replace('_', ' ')}. Could you please be more specific?",
                f"I need a clearer answer about your {feature.replace('_', ' ')}. Could you provide more details?",
                f"Sorry, I didn't catch that clearly. Could you tell me your {feature.replace('_', ' ')} again?"
            ],
            "partially_understood": [
                f"I partially understood your {feature.replace('_', ' ')}, but could you be more specific?",
                f"Could you provide a more exact answer for your {feature.replace('_', ' ')}?",
                f"I need a bit more precision about your {feature.replace('_', ' ')}. Could you clarify?"
            ]
        }
        
        messages = clarification_messages.get(reason, clarification_messages["not_understood"])
        return random.choice(messages)
    
    def _get_next_missing_feature(self):
        """Find the next feature that hasn't been collected"""
        for feature in self.required_features:
            if st.session_state.user_data[feature] is None:
                return feature
        return None
    
    def _initiate_prediction(self):
        """Initiate the prediction process when all data is collected"""
        st.session_state.prediction_done = True
        return "Perfect! I have all the information I need. Let me analyze your cardiovascular risk now..."

# Initialize the agent
agent = HealthDataExtractionAgent(llm)

def ask_next_question():
    """Find the next unanswered feature and return a question for it"""
    for f, v in st.session_state.user_data.items():
        if v is None:
            q = random.choice(feature_questions[f])
            st.session_state.last_feature = f
            return q
    return None


def is_information_request(message: str) -> bool:
    """
    Check if the user message is an information-seeking question.
    """
    message = message.lower().strip()
    question_keywords = ["what", "why", "how", "when", "where", "which", "explain", "meaning", "information", "difference", "does", "do", "can", "should"]
    return message.endswith("?") or any(word in message for word in question_keywords)


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


def llm_predict_cardiovascular(features, model_name="gemma2:latest", temperature=0.1, top_p=0.95, repeat_penalty=1.2):
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

def debate_with_llm_and_ml_models_v2(features: dict):

    svm_result = svm_heart_predictor(features)
    rf_result = rf_heart_predictor(features)
    llm_result, llm_explanation = llm_predict_cardiovascular(features)

    votes = [svm_result, rf_result, llm_result]
    final_decision = 1 if votes.count(1) >= 2 else 0

    prompt = f"""
You are a cardiovascular disease prediction expert.

Three independent models gave the following results:
- SVM: {svm_result}
- Random Forest: {rf_result}
- LLM (based on clinical reasoning): {llm_result}

Patient data:
{json.dumps(features, indent=2)}

The final decision was made by majority voting (at least 2 out of 3 must agree). Your task is to:
1. Explain the decision-making process.
2. Validate if the majority vote aligns with the patient data.
3. Comment if any model seems off.
4. End with: FINAL_DECISION: {final_decision}
"""

    response = llm.invoke(prompt)
    explanation = response.content if hasattr(response, "content") else str(response)

    return {
        "svm_result": svm_result,
        "rf_result": rf_result,
        "llm_model_result": llm_result,
        "llm_final_explanation": explanation,
        "llm_predictor_reasoning": llm_explanation,
        "final_decision": final_decision
    }


def display_prediction_results(pred, explanation, features):
    """Helper function to display prediction results"""
    if pred == 1:
        st.warning("⚠️ Higher risk of cardiovascular disease detected")
    else:
        st.success("✅ Lower risk of cardiovascular disease detected")

    st.metric("Prediction", "Has CVD" if pred == 1 else "No CVD")

    with st.expander("View detailed explanation"):
        st.write(explanation)

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

# UI starts
st.title("🫀 Cardiovascular Disease Risk Assessment Chat")

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

# Handle first interaction - start the conversation
if not st.session_state.chat_history:
    first_question = ask_next_question()
    if first_question:
        st.session_state.chat_history.append(("bot", first_question))
        st.markdown(f"""
        <div style='text-align: left; clear: both; padding: 5px 0;'>
            <div style='display: inline-block; background-color: #f1f3f4; color: black;
                        padding: 10px 15px; border-radius: 15px; max-width: 70%;'>
                {first_question}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Show chat input only if the chat is not closed
if not st.session_state.chat_closed:
    user_input = st.chat_input("Your message...")
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_input))
        st.markdown(f"""
        <div style='text-align: right; clear: both; padding: 5px 0;'>
            <div style='display: inline-block; background-color: #0083B8; color: white;
                        padding: 10px 15px; border-radius: 15px; max-width: 70%;'>
                {user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if is_information_request(user_input):
            info_prompt = f"""
You are a medical assistant. Please provide a clear, friendly, and informative answer to the following health-related question:

User: "{user_input}"
"""
            info_response = llm.invoke(info_prompt)
            info_content = info_response.content if hasattr(info_response, "content") else str(info_response)

            last_question = ""
            if st.session_state.last_feature:
                last_question = random.choice(feature_questions[st.session_state.last_feature])

            combined_response = f"{info_content.strip()}\n\nBy the way, could you also answer this for me?\n➡️ {last_question}"

            st.session_state.chat_history.append(("bot", combined_response))
            st.rerun()



        # Get the last question asked to provide context
        last_bot_message = ""
        for sender, msg in reversed(st.session_state.chat_history):
            if sender == "bot":
                last_bot_message = msg
                break

        extraction_result = agent.extract_data_with_confidence(
            question_asked=last_bot_message,
            user_response=user_input,
            expected_feature=st.session_state.last_feature
        )

        
        if not st.session_state.prediction_done:
            bot_response = agent.generate_response(extraction_result, st.session_state.last_feature)
            st.session_state.chat_history.append(("bot", bot_response))

        if st.session_state.prediction_done and not st.session_state.prediction_results:
            try:
                result_dict = debate_with_llm_and_ml_models_v2(st.session_state.user_data)
                st.session_state.prediction_results = result_dict["final_decision"]
                st.session_state.prediction_explanation = result_dict["llm_final_explanation"]

                st.session_state.chat_history.append(("bot", f"""
### 🧠 Prediction Summary:
- SVM says: **{result_dict['svm_result']}**
- Random Forest says: **{result_dict['rf_result']}**
- LLM Final Decision: **{result_dict['llm_model_result']}**

Click below to see the reasoning:
"""))

                st.session_state.chat_closed = True

            except Exception as e:
                st.session_state.chat_history.append(("bot", f"Sorry, there was an error processing your data: {str(e)}"))

        st.rerun()


else:
    # Show a message that the chat is closed
    st.info("📝 The assessment is complete and the chat session is now closed.")
    if st.session_state.prediction_results is not None:
        display_prediction_results(
            st.session_state.prediction_results,
            st.session_state.prediction_explanation,
            st.session_state.user_data
        )


# Display parsed features on the left
with st.sidebar:
    st.markdown("### 🧾 Extracted Data So Far")
    for feature, value in st.session_state.user_data.items():
        if value is not None:
            st.write(f"**{feature.capitalize()}**: {value}")
    
    # Show completion progress
    completed = sum(1 for v in st.session_state.user_data.values() if v is not None)
    total = len(st.session_state.user_data)
    progress = completed / total
    st.progress(progress)

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




