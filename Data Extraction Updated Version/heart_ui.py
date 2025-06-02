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

llm = ChatOllama(model="mistral:latest", temperature=0.1)

class HeartDiseaseDataExtractionAgent:
    """Enhanced agent for extracting heart disease risk data with confidence scoring"""
    
    def __init__(self, llm):
        self.llm = llm
        self.required_features = required_features
        self.feature_descriptions = feature_descriptions
    
    def parse_user_response_manually(self, user_response):
        """
        Manual parsing for common patterns before using LLM
        This provides a fallback and improves reliability
        """
        response_lower = user_response.lower().strip()
        extracted_data = {}
        
        # Gender detection (1=female, 2=male for heart disease)
        gender_male = [
            'male', 'man', 'guy', 'boy', 'gentleman', 'he/him', 'he him',
            "i am a man", "i'm a man", "identify as male", "identify as a man",
            "he is male", "he is a man", "my gender is male"
        ]

        gender_female = [
            'female', 'woman', 'girl', 'lady', 'she/her', 'she her',
            "i am a woman", "i'm a woman", "identify as female", "identify as a woman",
            "she is female", "she is a woman", "my gender is female"
        ]
        
        if any(word in response_lower for word in gender_male):
            extracted_data['gender'] = {'value': 2, 'confidence': 9}
        if any(word in response_lower for word in gender_female):
            extracted_data['gender'] = {'value': 1, 'confidence': 9}
        
        # Age detection 
        age_patterns = [
            r"\b(?:i'?m|i\s+am|im)\s+(\d{1,3})(?:\s+now)?\b",           # I'm 65, I am 65, im 65
            r"\b(\d{1,3})\s+(?:years?|yrs?)\s+old\b",                   # 65 years old, 65 yrs old
            r"\bage\s*[:\-]?\s*(\d{1,3})\b",                            # age: 65, age - 65
            r"\b(\d{1,3})\s*yo\b",                                      # 65yo
            r"\bin\s+my\s+(\d{2})'?s\b",                                # in my 60s, in my 20's
            r"●\s*age\s*[:\-]?\s*(\d{1,3})",                            # ● Age: 65
            r"\bturn(?:ing|s)?\s+(\d{1,3})\b",                          # turning 65, turns 65
            r"\bjust\s+turned\s+(\d{1,3})\b",                           # just turned 65
            r"\b(\d{1,3})\s+year\s+old\b",                              # 65 year old
            r"\bat\s+age\s+(\d{1,3})\b",                                # at age 65
            r"\b(\d{1,3})\b\s*(?:yo\b|y/o\b|yrs?\b)",                   # 65 yo, 65 y/o, 65 yrs
            r"\breached\s+age\s+of\s+(\d{1,3})\b",                      # reached age of 65
            r"\b(\d{1,3})\s+anniversar(?:y|ies)\s+of\s+birth\b",        # 65 anniversary of birth
            r"\b(\d{1,3})[-\s]?year[-\s]?old\b",
        ]
        
        for pattern in age_patterns:
            matches = re.finditer(pattern, response_lower)
            for match in matches:
                age = int(match.group(1))
                if 10 <= age <= 120:  # Reasonable age range
                    extracted_data['age'] = {'value': age, 'confidence': 9}
                    break  # Take first valid match
        
        # Height 
        height_patterns = [
            r'(\d+)\s*(cm|centimeters?)',                     # 170 cm
            r'height\s*:?\s*(\d+)\s*(cm|centimeters?)?',      # Height: 170 cm
            r'●\s*height\s*:?\s*(\d+)\s*(cm|centimeters?)?',  # ● Height: 170 cm
            r'i\s+am\s+(\d+)\s*(cm|centimeters?)',            # I am 170 cm
            r'my\s+height\s+is\s+(\d+)\s*(cm|centimeters?)?', # my height is 170 cm
            r'(\d+)\s*cm\s+tall',                             # 170 cm tall
            r'(\d+)\s+centimeters?\s+tall',                   # 170 centimeters tall
            r'i\'?m\s+(\d+)\s*(cm|centimeters?)',             # I'm 170 cm
            # Feet and inches patterns (convert to cm)
            r'(\d+)\s*[\'\']\s*(\d+)\s*[\""]',                # 5'8"
            r'(\d+)\s+feet?\s+(\d+)\s+inch(?:es)?',           # 5 feet 8 inches
            r'(\d+)\s*ft\s*(\d+)\s*in',                       # 5ft 8in
        ]
        
        for pattern in height_patterns:
            match = re.search(pattern, response_lower)
            if match:
                if "'" in pattern or "feet" in pattern or "ft" in pattern:
                    # Convert feet/inches to cm
                    feet = int(match.group(1))
                    inches = int(match.group(2)) if len(match.groups()) > 1 else 0
                    height_cm = int((feet * 12 + inches) * 2.54)
                    if 100 <= height_cm <= 250:
                        extracted_data['height'] = {'value': height_cm, 'confidence': 9}
                        break
                else:
                    height = int(match.group(1))
                    if 100 <= height <= 250:
                        extracted_data['height'] = {'value': height, 'confidence': 9}
                        break

        # Weight 
        weight_patterns = [
            r'(\d+(?:\.\d+)?)\s*(kg|kilograms?)',             # 70 kg, 70.5 kg
            r'weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kilograms?)?', # Weight: 70 kg
            r'●\s*weight\s*:?\s*(\d+(?:\.\d+)?)\s*(kg|kilograms?)?', # ● Weight: 70 kg
            r'i\s+weigh\s+(\d+(?:\.\d+)?)\s*(kg|kilograms?)?', # I weigh 70 kg
            r'my\s+weight\s+is\s+(\d+(?:\.\d+)?)\s*(kg|kilograms?)?', # my weight is 70 kg
            r'weigh\s+(\d+(?:\.\d+)?)\s*(kg|kilograms?)',     # weigh 70 kg
            r'i\'?m\s+(\d+(?:\.\d+)?)\s*(kg|kilograms?)',     # I'm 70 kg
            # Pounds patterns (convert to kg)
            r'(\d+(?:\.\d+)?)\s*(lbs?|pounds?)',              # 155 lbs
            r'weight\s*:?\s*(\d+(?:\.\d+)?)\s*(lbs?|pounds?)', # Weight: 155 lbs
            r'i\s+weigh\s+(\d+(?:\.\d+)?)\s*(lbs?|pounds?)',   # I weigh 155 lbs
        ]
        
        for pattern in weight_patterns:
            match = re.search(pattern, response_lower)
            if match:
                weight_val = float(match.group(1))
                unit = match.group(2) if len(match.groups()) > 1 else ''
                
                if 'lb' in unit or 'pound' in unit:
                    # Convert pounds to kg
                    weight_kg = int(weight_val * 0.453592)
                else:
                    weight_kg = int(weight_val)
                    
                if 30 <= weight_kg <= 300:
                    extracted_data['weight'] = {'value': weight_kg, 'confidence': 9}
                    break

        # Blood Pressure
        bp_patterns = [
            r'(\d{2,3})\s*[/over]\s*(\d{2,3})',                    # 120/80, 120 over 80
            r'blood\s+pressure\s*:?\s*(\d{2,3})\s*[/over]\s*(\d{2,3})', # Blood pressure: 120/80
            r'●\s*blood\s+pressure\s*:?\s*(\d{2,3})\s*[/over]\s*(\d{2,3})', # ● Blood pressure: 120/80
            r'bp\s*:?\s*(\d{2,3})\s*[/over]\s*(\d{2,3})',          # BP: 120/80
            r'my\s+blood\s+pressure\s+is\s+(\d{2,3})\s*[/over]\s*(\d{2,3})', # my blood pressure is 120/80
            r'pressure\s+is\s+(\d{2,3})\s*[/over]\s*(\d{2,3})',    # pressure is 120/80
            r'systolic\s+(\d{2,3}).*diastolic\s+(\d{2,3})',       # systolic 120 diastolic 80
            r'(\d{2,3})\s+systolic.*(\d{2,3})\s+diastolic',       # 120 systolic 80 diastolic
            r'top\s+number\s+(\d{2,3}).*bottom\s+number\s+(\d{2,3})', # top number 120 bottom number 80
            r"my\s+blood\s+pressure\s+is\s+(\d{2,3})\s+over\s+(\d{2,3})",
        ]
        
        for pattern in bp_patterns:
            match = re.search(pattern, response_lower)
            if match:
                systolic = int(match.group(1))
                diastolic = int(match.group(2))
                if 70 <= systolic <= 250 and 40 <= diastolic <= 150:
                    extracted_data['ap_hi'] = {'value': systolic, 'confidence': 9}
                    extracted_data['ap_lo'] = {'value': diastolic, 'confidence': 9}
                    break

        # Cholesterol detection
        cholesterol_normal = [
            "normal cholesterol", "cholesterol is normal", "cholesterol normal",
            "good cholesterol", "cholesterol is good", "cholesterol levels are normal",
            "healthy cholesterol", "no cholesterol issues", "cholesterol is fine"
        ]

        cholesterol_above = [
            "high cholesterol", "elevated cholesterol", "cholesterol is high",
            "above normal cholesterol", "cholesterol above normal", "slightly high cholesterol",
            "borderline cholesterol", "cholesterol is elevated"
        ]

        cholesterol_well_above = [
            "very high cholesterol", "extremely high cholesterol", "cholesterol is very high",
            "well above normal cholesterol", "dangerously high cholesterol",
            "severely elevated cholesterol", "cholesterol way above normal"
        ]
        
        if any(phrase in response_lower for phrase in cholesterol_well_above):
            extracted_data['cholesterol'] = {'value': 3, 'confidence': 8}
        if any(phrase in response_lower for phrase in cholesterol_above):
            extracted_data['cholesterol'] = {'value': 2, 'confidence': 8}
        if any(phrase in response_lower for phrase in cholesterol_normal):
            extracted_data['cholesterol'] = {'value': 1, 'confidence': 8}
        
        # Glucose/Blood sugar detection
        glucose_normal = [
            "normal glucose", "glucose is normal", "normal blood sugar", "blood sugar is normal",
            "glucose normal", "good glucose", "healthy glucose", "no diabetes",
            "glucose levels are normal", "blood sugar levels are normal"
        ]

        glucose_above = [
            "high glucose", "elevated glucose", "high blood sugar", "elevated blood sugar",
            "glucose is high", "blood sugar is high", "above normal glucose",
            "prediabetic", "borderline diabetes", "slightly high glucose"
        ]

        glucose_well_above = [
            "very high glucose", "extremely high glucose", "very high blood sugar",
            "diabetic", "diabetes", "well above normal glucose", "severely elevated glucose",
            "glucose way above normal", "uncontrolled diabetes"
        ]
        
        if any(phrase in response_lower for phrase in glucose_well_above):
            extracted_data['gluc'] = {'value': 3, 'confidence': 8}
        if any(phrase in response_lower for phrase in glucose_above):
            extracted_data['gluc'] = {'value': 2, 'confidence': 8}
        if any(phrase in response_lower for phrase in glucose_normal):
            extracted_data['gluc'] = {'value': 1, 'confidence': 8}
        
        # Smoking detection 
        smoking_negative = [
            "don't smoke", "do not smoke", "never smoked", "non-smoker", "non smoker",
            "quit smoking", "stopped smoking", "gave up smoking", "used to smoke",
            "smoked in the past", "i have never smoked", "i am not a smoker",
            "not a smoker", "never been a smoker"
        ]

        smoking_positive = [
            "i smoke", "i'm a smoker", "i am a smoker", "smoker", "smoking",
            "smoke regularly", "smoke a lot", "smoke cigarettes", "smoke tobacco",
            "cigarette", "tobacco", "nicotine", "can't quit smoking", "cannot quit smoking",
            "chain smoker", "i have been smoking", "pack a day"
        ]
        
        # Check negative first (more specific)
        smoking_found = False
        for phrase in smoking_negative:
            if phrase in response_lower:
                if any(past in response_lower for past in ["used to", "when i was", "back then", "in the past"]):
                    extracted_data['smoke'] = {'value': 0, 'confidence': 8} 
                else:
                    extracted_data['smoke'] = {'value': 0, 'confidence': 9}
                smoking_found = True
                break
        
        if not smoking_found:
            for phrase in smoking_positive:
                if phrase in response_lower:
                    extracted_data['smoke'] = {'value': 1, 'confidence': 8}
                    break
        
        # Alcohol detection
        alcohol_positive = [
            "i drink alcohol", "i drink beer", "i drink wine", "drink alcohol",
            "drinking alcohol", "i have a drink", "social drinking", "wine with dinner",
            "enjoy wine", "enjoy beer", "regularly drink", "often drink", "i drink",
            "consume alcohol", "alcohol consumption"
        ]

        alcohol_negative = [
            "don't drink", "do not drink", "never drink", "no alcohol", "i'm sober",
            "avoid alcohol", "abstain from alcohol", "no alcohol use", "teetotal",
            "don't consume alcohol", "alcohol free","i don't drink alcohol", "i do not drink alcohol", "i never drink alcohol",
    "never drink alcohol", "don’t drink alcohol", "i don’t consume alcohol", "i do not consume alcohol",
        ]
        
        alcohol_found = False
        for phrase in alcohol_negative:
            if phrase in response_lower:
                extracted_data['alco'] = {'value': 0, 'confidence': 8}
                alcohol_found = True
                break
        
        if not alcohol_found:
            for phrase in alcohol_positive:
                if phrase in response_lower:
                    if any(word in response_lower for word in ['occasionally', 'sometimes', 'rarely', 'social']):
                        extracted_data['alco'] = {'value': 1, 'confidence': 7}
                    else:
                        extracted_data['alco'] = {'value': 1, 'confidence': 8}
                    break
        
        # Physical activity detection
        active_positive = [
            "i exercise", "i work out", "i'm active", "i am active", "physically active",
            "regular exercise", "go to gym", "play sports", "run regularly", "walk daily",
            "active lifestyle", "exercise regularly", "stay active", "very active",
            "workout routine", "fitness routine", "i train"
        ]

        active_negative = [
            "don't exercise", "do not exercise", "sedentary", "not active", "inactive",
            "don't work out", "no exercise", "couch potato", "sit all day",
            "no physical activity", "avoid exercise", "hate exercise", "lazy",
            "not very physically active", "not really active", "low activity level",
    "i’m not active", "i am not active", "i’m inactive", "i am inactive"
        ]
        
        activity_found = False
        for phrase in active_negative:
            if phrase in response_lower:
                extracted_data['active'] = {'value': 0, 'confidence': 8}
                activity_found = True
                break
        
        if not activity_found:
            for phrase in active_positive:
                if phrase in response_lower:
                    extracted_data['active'] = {'value': 1, 'confidence': 8}
                    break
        
        return extracted_data

    def extract_data_with_confidence(self, question_asked, user_response, expected_feature=None):
        """
        Enhanced extraction method that combines manual parsing with LLM analysis
        if any required feature is missing after manual extraction.
        """

        # First try manual parsing
        # manual_extraction = self.parse_user_response_manually(user_response)
        manual_extraction = {}
        # Create the result structure with default values
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

        # If any required feature is still missing AND manual extraction didn't find much, invoke LLM
        missing_features = [
            f for f, v in result["extracted_features"].items()
            if v["value"] is None
        ]
        
        # Only use LLM if manual extraction found very little and there are missing features
        if missing_features and len(manual_extraction) < 2:
            try:
                llm_result = self._llm_extract_data(question_asked, user_response, expected_feature)
                if llm_result and isinstance(llm_result, dict):
                    # Merge LLM results with manual results (manual takes priority)
                    for feature, llm_data in llm_result.get("extracted_features", {}).items():
                        if (feature in result["extracted_features"] and 
                            result["extracted_features"][feature]["value"] is None and
                            llm_data.get("value") is not None):
                            result["extracted_features"][feature] = llm_data
                            if feature not in result["analysis"]["other_features_provided"]:
                                result["analysis"]["other_features_provided"].append(feature)
                    
                    # Update analysis if LLM found the expected feature
                    if (expected_feature and 
                        llm_result.get("extracted_features", {}).get(expected_feature, {}).get("value") is not None):
                        result["analysis"]["expected_feature_provided"] = True
                        
            except Exception as e:
                print(f"LLM extraction failed: {e}")
                # Continue with manual extraction results
                pass

        return result

    def _llm_extract_data(self, question_asked, user_response, expected_feature=None):
        """
        Fallback LLM extraction with improved error handling
        """
        
        prompt = f"""Extract health information from this response and return it in the EXACT JSON format below.

User was asked: "{question_asked}"
User responded: "{user_response}"

Return ONLY valid JSON in this exact format (no other text):
{{
  "extracted_features": {{
    "age": {{"value": null, "confidence": 0}},
    "gender": {{"value": null, "confidence": 0}},
    "height": {{"value": null, "confidence": 0}},
    "weight": {{"value": null, "confidence": 0}},
    "ap_hi": {{"value": null, "confidence": 0}},
    "ap_lo": {{"value": null, "confidence": 0}},
    "cholesterol": {{"value": null, "confidence": 0}},
    "gluc": {{"value": null, "confidence": 0}},
    "smoke": {{"value": null, "confidence": 0}},
    "alco": {{"value": null, "confidence": 0}},
    "active": {{"value": null, "confidence": 0}}
  }},
  "analysis": {{
    "expected_feature_provided": false,
    "other_features_provided": [],
    "needs_clarification": false,
    "clarification_reason": ""
  }}
}}

RULES:
- For gender: 1=female, 2=male, null if unclear
- For age: exact number, null if unclear
- For height: value in centimeters, null if unclear
- For weight: value in kilograms, null if unclear
- For ap_hi: systolic blood pressure (higher number), null if unclear
- For ap_lo: diastolic blood pressure (lower number), null if unclear
- For cholesterol: 1=normal, 2=above normal, 3=well above normal, null if unclear
- For gluc: 1=normal, 2=above normal, 3=well above normal, null if unclear
- For smoke: 1=yes, 0=no, null if unclear
- For alco: 1=yes, 0=no, null if unclear
- For active: 1=active, 0=inactive, null if unclear
- Confidence: 1-10 (10=very certain)
- Return ONLY the JSON, no explanations"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean the content - remove any markdown formatting
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Try to find JSON within the content
            json_patterns = [
                r'\{[\s\S]*\}',  # Anything between braces
                r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Nested braces pattern
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    break
            
            if not json_str:
                json_str = content
            
            # Try to parse JSON
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Attempted to parse: {json_str[:200]}...")
                return None
                
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return None
    
    def generate_response(self, extraction_result, expected_feature):
        """Generate appropriate bot response based on extraction results"""
        
        analysis = extraction_result["analysis"]
        extracted_features = extraction_result["extracted_features"]
        
        # Count how many features were extracted with high confidence
        high_confidence_features = []
        for feature, data in extracted_features.items():
            if data["value"] is not None and data["confidence"] >= 7:
                high_confidence_features.append(feature)
        
        # If we got multiple high-confidence features, acknowledge them
        if len(high_confidence_features) > 1:
            return self._handle_multiple_features(high_confidence_features, extracted_features)
        
        # If we got the expected feature with high confidence
        elif expected_feature and extracted_features[expected_feature]["value"] is not None:
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
        
        # If we got one other feature with high confidence
        elif len(high_confidence_features) == 1:
            feature = high_confidence_features[0]
            st.session_state.user_data[feature] = extracted_features[feature]["value"]
            return self._acknowledge_and_redirect(feature, expected_feature)
        
        # Default case - ask for clarification
        else:
            return self._ask_for_clarification(expected_feature, "not_understood")
    
    def _handle_multiple_features(self, features, extracted_features):
        """Handle when multiple features are extracted from one response"""
        
        # Save all the extracted data
        saved_features = []
        for feature in features:
            st.session_state.user_data[feature] = extracted_features[feature]["value"]
            saved_features.append(feature.replace('_', ' '))
        
        # Create acknowledgment message
        if len(saved_features) <= 3:
            ack_msg = f"Thank you! I've noted your {', '.join(saved_features)} information."
        else:
            ack_msg = f"Thank you! I've noted information about {len(saved_features)} health factors."
        
        # Find next question
        next_feature = self._get_next_missing_feature()
        if next_feature:
            st.session_state.last_feature = next_feature
            question = random.choice(feature_questions[next_feature])
            return f"{ack_msg} Now, {question.lower()}"
        else:
            return self._initiate_prediction()
    
    def _acknowledge_and_redirect(self, noted_feature, expected_feature):
        """Acknowledge one feature and redirect to expected question"""
        
        feature_name = noted_feature.replace('_', ' ')
        ack_msg = f"I noted your {feature_name} information."
        
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
        return "Perfect! I have all the information I need. Let me analyze your heart disease risk now..."
# Initialize the agent
agent = HeartDiseaseDataExtractionAgent(llm)

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

def is_information_request(message: str) -> bool:
    """
    Check if the user message is an information-seeking question.
    Improved logic to better distinguish between questions and data-providing responses.
    """
    message = message.lower().strip()
    
    # Direct question patterns that clearly indicate information seeking
    direct_question_starters = [
        "what is", "what are", "what does", "what do",
        "why is", "why are", "why does", "why do",
        "how is", "how are", "how does", "how do", "how can",
        "when is", "when are", "when does", "when do",
        "where is", "where are", "where does", "where do",
        "which is", "which are", "which does", "which do",
        "can you explain", "could you explain", "please explain",
        "tell me about", "what's the difference", "what is the difference"
    ]
    
    # Check if message starts with direct question patterns
    for starter in direct_question_starters:
        if message.startswith(starter):
            return True
    
    # Check if it's a simple question (ends with ? and is relatively short)
    if message.endswith("?") and len(message.split()) <= 15:
        return True
    
    # If the message is long (indicating detailed response) and doesn't start with question words, it's likely data
    if len(message.split()) > 20:
        return False
    
    # Check for very specific information-seeking patterns
    info_seeking_patterns = [
        r"^(what|why|how|when|where|which)\s+.*\?$",
        r"^(can|could|should|would)\s+you\s+(tell|explain|help)",
        r"^(i\s+want\s+to\s+know|i\s+need\s+to\s+understand|i'm\s+curious)"
    ]
    
    import re
    for pattern in info_seeking_patterns:
        if re.search(pattern, message):
            return True
    
    return False

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

        # Check if it's a genuine information request
        is_info_request = is_information_request(user_input)
        
        # Additional check: If user is providing data (contains personal info indicators), prioritize data extraction
        data_indicators = ['i am', 'i have', 'i do', 'i don\'t', 'i smoke', 'my age', 'years old', 'my blood pressure', 'my weight', 'my height', 'male', 'female', 'i exercise', 'i work out']
        contains_data = any(indicator in user_input.lower() for indicator in data_indicators)
        
        if is_info_request and not contains_data:
            # Handle information request
            info_prompt = f"""
You are a medical assistant. Please provide a clear, friendly, and informative answer to the following health-related question:

User: "{user_input}"
"""
            info_response = llm.invoke(info_prompt)
            info_content = info_response.content if hasattr(info_response, "content") else str(info_response)

            # Get the last question asked to continue the assessment
            last_question = ""
            if st.session_state.last_feature:
                last_question = random.choice(feature_questions[st.session_state.last_feature])

            combined_response = f"{info_content.strip()}\n\nNow, let's continue with your assessment:\n➡️ {last_question}"

            st.session_state.chat_history.append(("bot", combined_response))
            st.rerun()
        else:
            # Handle data extraction (normal flow)
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