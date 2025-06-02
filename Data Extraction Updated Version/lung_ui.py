import streamlit as st
import json
import random
import re
from langchain_ollama import ChatOllama
from tool_loader import svm_lung_predictor, rf_lung_predictor



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
    "gender", "age", "smoking", "yellow_fingers", "anxiety", "peer_pressure",
    "chronic_disease", "fatigue", "allergy", "wheezing", "alcohol_consuming",
    "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain"
]

feature_questions = {
    "gender": ["What's your gender? (Male/Female)", "Are you male or female?", "Could you tell me your gender?"],
    "age": ["How old are you?", "May I ask your age?", "What's your age in years?"],
    "smoking": ["Do you smoke?", "Are you a smoker?", "Do you have a smoking habit?"],
    "yellow_fingers": ["Do you have yellow fingers?", "Have you noticed yellowing of your fingers?", "Are your fingers yellowed?"],
    "anxiety": ["Do you experience anxiety?", "Do you suffer from anxiety?", "Are you anxious frequently?"],
    "peer_pressure": ["Do you feel peer pressure?", "Are you influenced by peer pressure?", "Do you experience pressure from peers?"],
    "chronic_disease": ["Do you have any chronic diseases?", "Are you diagnosed with chronic conditions?", "Do you suffer from chronic illness?"],
    "fatigue": ["Do you experience fatigue?", "Do you feel tired frequently?", "Are you often fatigued?"],
    "allergy": ["Do you have allergies?", "Are you allergic to anything?", "Do you suffer from allergies?"],
    "wheezing": ["Do you wheeze?", "Do you experience wheezing sounds when breathing?", "Have you noticed wheezing?"],
    "alcohol_consuming": ["Do you consume alcohol?", "Do you drink alcohol?", "Are you a drinker?"],
    "coughing": ["Do you cough frequently?", "Do you have a persistent cough?", "Are you coughing often?"],
    "shortness_of_breath": ["Do you experience shortness of breath?", "Do you feel breathless?", "Do you have difficulty breathing?"],
    "swallowing_difficulty": ["Do you have difficulty swallowing?", "Is swallowing difficult for you?", "Do you experience swallowing problems?"],
    "chest_pain": ["Do you experience chest pain?", "Do you have chest pain?", "Are you experiencing pain in your chest?"]
}

# Feature descriptions for better understanding
feature_descriptions = {
    "gender": "gender (1 for male, 0 for female)",  # Değiştirildi: M/F yerine 1/0
    "age": "age in years (exact number)",
    "smoking": "smoking status (0=no, 1=yes)",
    "yellow_fingers": "yellowing of fingers (0=no, 1=yes)",
    "anxiety": "anxiety levels (0=no, 1=yes)",
    "peer_pressure": "peer pressure influence (0=no, 1=yes)",
    "chronic_disease": "presence of chronic diseases (0=no, 1=yes)",
    "fatigue": "experiencing fatigue (0=no, 1=yes)",
    "allergy": "having allergies (0=no, 1=yes)",
    "wheezing": "wheezing sounds when breathing (0=no, 1=yes)",
    "alcohol_consuming": "alcohol consumption (0=no, 1=yes)",
    "coughing": "frequent coughing (0=no, 1=yes)",
    "shortness_of_breath": "shortness of breath (0=no, 1=yes)",
    "swallowing_difficulty": "difficulty swallowing (0=no, 1=yes)",
    "chest_pain": "chest pain (0=no, 1=yes)"
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

class LungCancerDataExtractionAgent:
    """Enhanced agent for extracting lung cancer risk data with confidence scoring"""
    
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
        
        # Gender detection
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
            extracted_data['gender'] = {'value': 1, 'confidence': 9}
        if any(word in response_lower for word in gender_female):
            extracted_data['gender'] = {'value': 0, 'confidence': 9}
        
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
        ]
        
        for pattern in age_patterns:
            matches = re.finditer(pattern, response_lower)
            for match in matches:
                age = int(match.group(1))
                if 10 <= age <= 120:  # Reasonable age range
                    extracted_data['age'] = {'value': age, 'confidence': 9}
                    break  # Take first valid match
        
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
                    extracted_data['smoking'] = {'value': 0, 'confidence': 8} 
                else:
                    extracted_data['smoking'] = {'value': 0, 'confidence': 9}
                smoking_found = True
                break
        
        if not smoking_found:
            for phrase in smoking_positive:
                if phrase in response_lower:
                    extracted_data['smoking'] = {'value': 1, 'confidence': 8}
                    break
        
        # Yellow fingers
        yellow_finger_positive = [
            "yellow finger", "yellowing finger", "yellow fingers", "yellowing fingers",
            "fingers are yellow", "fingers look yellow", "fingers turn yellow",
            "yellow stain", "yellow stains", "stained finger", "stained fingers",
            "nicotine stain", "nicotine marks", "yellow marks on fingers",
            "finger discoloration", "fingers discolored"
        ]

        yellow_finger_negative = [
            "no yellow finger", "no yellowing finger", "no yellow fingers", "no yellow stains",
            "fingers are not yellow", "fingers look normal", "no stains on fingers",
            "fingers are clean", "no discoloration", "no nicotine stains",
            "fingers not discolored", "no yellow marks on fingers"
        ]

        if any(keyword in response_lower for keyword in yellow_finger_positive):
            extracted_data['yellow_fingers'] = {'value': 1, 'confidence': 8}
        elif any(keyword in response_lower for keyword in yellow_finger_negative):
            extracted_data['yellow_fingers'] = {'value': 0, 'confidence': 8}

        # Anxiety
        anxiety_positive = [
            "anxious", "i feel anxious", "i am anxious", "i get anxious",
            "i have anxiety", "anxiety issues", "anxiety problems", "anxiety symptoms",
            "suffering from anxiety", "dealing with anxiety", "panic attacks",
            "nervous", "worried", "on edge", "panic", "can't relax", "cannot relax"
        ]

        anxiety_negative = [
            "no anxiety", "not anxious", "feeling calm", "i am calm", "feeling relaxed",
            "am relaxed", "don't feel anxious", "do not feel anxious",
            "i have no anxiety", "calm state", "peace of mind", "no sign of anxiety",
            "not have anxiety", "no anxiety issues", "no anxiety problems"
        ]
        
        anxiety_found = False
        for phrase in anxiety_negative:
            if phrase in response_lower:
                extracted_data['anxiety'] = {'value': 0, 'confidence': 8}
                anxiety_found = True
                break
        
        if not anxiety_found:
            for phrase in anxiety_positive:
                if phrase in response_lower:
                    extracted_data['anxiety'] = {'value': 1, 'confidence': 8}
                    break
        
        # Peer pressure
        peer_pressure_positive = [
            "peer pressure", "pressure from peers", "under pressure", "peer influence", "under peer pressure",
            "influenced by friends", "influenced by peers", "pressure to drink",
            "pressure to smoke", "social pressure", "friends made me", "friends pressured me",
            "friends wanted me to", "everyone else", "felt pressured"
        ]

        peer_pressure_negative = [
            "no peer pressure", "not under pressure", "no pressure from peers", "not under peer pressure",
            "no social pressure", "independent decision", "independent choice",
            "wasn't influenced", "decided on my own", "my own decision", "it was my choice"
        ]
        
        pressure_found = False
        for phrase in peer_pressure_negative:
            if phrase in response_lower:
                extracted_data['peer_pressure'] = {'value': 0, 'confidence': 8}
                pressure_found = True
                break
        
        if not pressure_found:
            for phrase in peer_pressure_positive:
                if phrase in response_lower:
                    extracted_data['peer_pressure'] = {'value': 1, 'confidence': 8}
                    break
        
        # Chronic disease
        chronic_positive = [
            "chronic disease", "chronic illness", "chronic condition", "ongoing condition",
            "medical condition", "heart problem", "heart condition", "diabetes", "diabetic",
            "hypertension", "high blood pressure", "copd", "living with chronic",
            "have a chronic condition", "suffering from chronic", "diagnosed with diabetes",
            "diagnosed with hypertension"
        ]

        chronic_negative = [
            "no chronic issues", "no chronic conditions", "no medical conditions",
            "don't have chronic", "do not have chronic", "not chronically ill",
            "no health problems", "healthy", "in good health", "free of disease",
            "never had a chronic condition"
        ]
        
        chronic_found = False
        for phrase in chronic_negative:
            if phrase in response_lower:
                extracted_data['chronic_disease'] = {'value': 0, 'confidence': 8}
                chronic_found = True
                break
        
        if not chronic_found:
            for phrase in chronic_positive:
                if phrase in response_lower:
                    extracted_data['chronic_disease'] = {'value': 1, 'confidence': 8}
                    break
        
        # Fatigue
        fatigue_positive = [
            "tired", "fatigue", "exhausted", "weary", "sluggish", "drained", "worn out",
            "feel tired", "feeling tired", "always tired", "constantly tired",
            "wake up tired", "lacking energy", "low energy", "can't stay awake",
            "get tired easily", "no energy", "sleepy all the time"
        ]

        fatigue_negative = [
            "not tired", "not fatigued", "not exhausted", "no fatigue",
            "don't feel tired", "do not feel tired", "energetic", "feel energetic",
            "full of energy", "well rested", "good energy levels", "active and alert"
        ]
        
        fatigue_found = False
        for phrase in fatigue_negative:
            if phrase in response_lower:
                extracted_data['fatigue'] = {'value': 0, 'confidence': 8}
                fatigue_found = True
                break
        
        if not fatigue_found:
            for phrase in fatigue_positive:
                if phrase in response_lower:
                    extracted_data['fatigue'] = {'value': 1, 'confidence': 8}
                    break
        
        # Allergy
        allergy_positive = [
            "allergy", "allergies", "allergic", "i have allergies", "i am allergic",
            "allergic to", "allergic reaction", "allergy symptoms", "suffer from allergies"
        ]

        allergy_negative = [
            "no allergy", "no allergies", "no known allergies", "don't have allergies",
            "do not have allergies", "not allergic", "i am not allergic",
            "without any allergies"
        ]
        
        allergy_found = False
        for phrase in allergy_negative:
            if phrase in response_lower:
                extracted_data['allergy'] = {'value': 0, 'confidence': 8}
                allergy_found = True
                break
        
        if not allergy_found:
            for phrase in allergy_positive:
                if phrase in response_lower:
                    extracted_data['allergy'] = {'value': 1, 'confidence': 8}
                    break
        
        # Wheezing
        wheeze_positive = [
            "wheeze", "wheezing", "wheezing sound", "wheezing attack", "noisy breathing",
            "whistling sound", "whistling breath", "chest tightness and wheezing",
            "breathing sound", "breathing issues", "breathing problems"
        ]

        wheeze_negative = [
            "no wheeze", "no wheezing", "don't wheeze", "breathing is clear", "doesn't wheeze", "does not wheeze",
            "clear lungs", "no noisy breathing", "no abnormal breathing", "do not wheeze"
        ]
        
        wheeze_found = False
        for phrase in wheeze_negative:
            if phrase in response_lower:
                extracted_data['wheezing'] = {'value': 0, 'confidence': 8}
                wheeze_found = True
                break
        
        if not wheeze_found:
            for phrase in wheeze_positive:
                if phrase in response_lower:
                    extracted_data['wheezing'] = {'value': 1, 'confidence': 8}
                    break
        
        # Alcohol
        alcohol_positive = [
            "i drink alcohol", "i drink beer", "i drink wine", "drink alcohol",
            "drinking alcohol", "i have a drink", "social drinking", "wine with dinner",
            "enjoy wine", "enjoy beer", "regularly drink", "often drink"
        ]

        alcohol_negative = [
            "don't drink", "do not drink", "never drink", "no alcohol", "i'm sober",
            "avoid alcohol", "abstain from alcohol", "no alcohol use", "teetotal"
        ]
        
        alcohol_found = False
        for phrase in alcohol_negative:
            if phrase in response_lower:
                extracted_data['alcohol_consuming'] = {'value': 0, 'confidence': 8}
                alcohol_found = True
                break
        
        if not alcohol_found:
            for phrase in alcohol_positive:
                if phrase in response_lower:
                    if any(word in response_lower for word in ['occasionally', 'sometimes', 'rarely', 'social']):
                        extracted_data['alcohol_consuming'] = {'value': 1, 'confidence': 7}
                    else:
                        extracted_data['alcohol_consuming'] = {'value': 1, 'confidence': 8}
                    break
        
        # Coughing
        cough_positive = [
            "cough", "coughing", "i have a cough", "i cough", "persistent cough",
            "dry cough", "wet cough", "chronic cough", "frequent coughing",
            "can't stop coughing", "bouts of coughing"
        ]

        cough_negative = [
            "no cough", "no coughing", "don't cough", "do not cough", "not coughing",
            "without any cough", "cough resolved", "cough has gone", "do not have coughing",
            "don't have coughing"
        ]
        
        cough_found = False
        for phrase in cough_negative:
            if phrase in response_lower:
                extracted_data['coughing'] = {'value': 0, 'confidence': 8}
                cough_found = True
                break
        
        if not cough_found:
            for phrase in cough_positive:
                if phrase in response_lower:
                    if 'persistent' in response_lower or 'chronic' in response_lower:
                        extracted_data['coughing'] = {'value': 1, 'confidence': 9}
                    else:
                        extracted_data['coughing'] = {'value': 1, 'confidence': 8}
                    break
        
        # Shortness of breath
        breath_positive = [
            "shortness of breath", "short of breath", "out of breath", "breathless",
            "difficulty breathing", "trouble breathing", "hard to breathe",
            "can't breathe", "unable to breathe", "breathing problems", "struggle to breathe",
            "experience shortness of breath", "breathing difficulty", "breathing issues"
        ]

        breath_negative = [
            "no shortness of breath", "not breathless", "breathing is fine",
            "breathing is normal", "can breathe normally", "no breathing problems",
            "not experience shortness of breath", "not have shortness of breath"
        ]
        
        breath_found = False
        for phrase in breath_negative:
            if phrase in response_lower:
                extracted_data['shortness_of_breath'] = {'value': 0, 'confidence': 8}
                breath_found = True
                break
        
        if not breath_found:
            for phrase in breath_positive:
                if phrase in response_lower:
                    extracted_data['shortness_of_breath'] = {'value': 1, 'confidence': 8}
                    break
        
        # Swallowing difficulty 
        swallow_positive = [
            "difficulty swallowing", "trouble swallowing", "problems swallowing",
            "hard to swallow", "can't swallow", "swallowing problem", "swallowing hurts",
            "swallowing is hard", "swallowing trouble"
        ]

        swallow_negative = [
            "no difficulty swallowing", "no trouble swallowing", "swallowing is normal",
            "swallowing is fine", "can swallow normally", "no swallowing problems", "not experiencing swallowing",
            "not have swallowing", "swallowing is easy", "swallowing is clear"
        ]
        
        swallow_found = False
        for phrase in swallow_negative:
            if phrase in response_lower:
                extracted_data['swallowing_difficulty'] = {'value': 0, 'confidence': 8}
                swallow_found = True
                break
        
        if not swallow_found:
            for phrase in swallow_positive:
                if phrase in response_lower:
                    extracted_data['swallowing_difficulty'] = {'value': 1, 'confidence': 8}
                    break
        
        # Chest pain
        chest_positive = [
            "chest pain", "chest ache", "chest discomfort", "chest tightness",
            "chest pressure", "chest hurts", "pain in chest", "tight chest",
            "chest feels tight", "chest feels heavy", "stabbing chest pain",
            "burning chest pain", "sharp chest pain"
        ]

        chest_negative = [
            "no chest pain", "no chest discomfort", "chest feels fine", "chest is fine",
            "not experiencing chest pain", "chest is clear", "don't have chest pain", "don' feel chest pain",
            "do not feel chest pain", "no tightness in chest", "no pressure in chest", "no ache in chest"
        ]
        
        chest_found = False
        for phrase in chest_negative:
            if phrase in response_lower:
                extracted_data['chest_pain'] = {'value': 0, 'confidence': 8}
                chest_found = True
                break
        
        if not chest_found:
            for phrase in chest_positive:
                if phrase in response_lower:
                    extracted_data['chest_pain'] = {'value': 1, 'confidence': 8}
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
    "gender": {{"value": null, "confidence": 0}},
    "age": {{"value": null, "confidence": 0}},
    "smoking": {{"value": null, "confidence": 0}},
    "yellow_fingers": {{"value": null, "confidence": 0}},
    "anxiety": {{"value": null, "confidence": 0}},
    "peer_pressure": {{"value": null, "confidence": 0}},
    "chronic_disease": {{"value": null, "confidence": 0}},
    "fatigue": {{"value": null, "confidence": 0}},
    "allergy": {{"value": null, "confidence": 0}},
    "wheezing": {{"value": null, "confidence": 0}},
    "alcohol_consuming": {{"value": null, "confidence": 0}},
    "coughing": {{"value": null, "confidence": 0}},
    "shortness_of_breath": {{"value": null, "confidence": 0}},
    "swallowing_difficulty": {{"value": null, "confidence": 0}},
    "chest_pain": {{"value": null, "confidence": 0}}
  }},
  "analysis": {{
    "expected_feature_provided": false,
    "other_features_provided": [],
    "needs_clarification": false,
    "clarification_reason": ""
  }}
}}

RULES:
- For gender: 1=male, 0=female, null if unclear
- For age: exact number, null if unclear  
- For others: 1=yes, 0=no, yes=1, no=0 null if unclear
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
        return "Perfect! I have all the information I need. Let me analyze your lung cancer risk now..."

# Initialize the agent
agent = LungCancerDataExtractionAgent(llm)

def ask_next_question():
    """Find the next unanswered feature and return a question for it"""
    for f, v in st.session_state.user_data.items():
        if v is None:
            q = random.choice(feature_questions[f])
            st.session_state.last_feature = f
            return q
    return None



def debate_with_llm_and_ml_models_v2_lung(features: dict):

    svm_result = svm_lung_predictor(features)
    rf_result = rf_lung_predictor(features)
    llm_result, llm_explanation = llm_predict_lung_cancer(features)

    votes = [svm_result, rf_result, llm_result]
    final_decision = 1 if votes.count(1) >= 2 else 0

    prompt = f"""
You are a lung cancer prediction expert.

Three independent models gave the following results:
- SVM: {svm_result}
- Random Forest: {rf_result}
- LLM (based on clinical reasoning): {llm_result}

Patient data:
{json.dumps(features, indent=2)}

The final decision was made by majority voting (at least 2 out of 3 must agree). Your task is to:
1. Analyze and explain why this decision makes sense or not.
2. Evaluate which model seems most reliable in this specific case.
3. Explain if any model seems questionable and why.
4. Provide your final validation with the statement: FINAL_DECISION: {final_decision}
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


def llm_predict_lung_cancer(features, model_name="gemma2:latest", temperature=0.1, top_p=0.95, repeat_penalty=1.2):
    """Predict lung cancer risk using LLM and feature-based risk scoring"""
    row = features
    risk_score = 0
    
    # Calculate risk score based on features
    if row.get('age') and row['age'] > 60:
        risk_score += 2
    elif row.get('age') and row['age'] > 50:
        risk_score += 1
        
    if row.get('smoking') and row['smoking'] == 1:
        risk_score += 3  # Smoking is a major risk factor
    if row.get('yellow_fingers') and row['yellow_fingers'] == 1:
        risk_score += 2  # Often associated with heavy smoking
    if row.get('chronic_disease') and row['chronic_disease'] == 1:
        risk_score += 1
    if row.get('coughing') and row['coughing'] == 1:
        risk_score += 1
    if row.get('shortness_of_breath') and row['shortness_of_breath'] == 1:
        risk_score += 1
    if row.get('chest_pain') and row['chest_pain'] == 1:
        risk_score += 1
    if row.get('wheezing') and row['wheezing'] == 1:
        risk_score += 1
    if row.get('fatigue') and row['fatigue'] == 1:
        risk_score += 1
    if row.get('swallowing_difficulty') and row['swallowing_difficulty'] == 1:
        risk_score += 1
    
    prompt = f"""You are a medical expert analyzing lung cancer risk. 
Your task is to predict whether the patient has lung cancer risk (0 = low risk, 1 = high risk) based on the data provided.

Analyze the patient's full clinical profile carefully, considering all these factors:

- Gender: Male patients typically have higher lung cancer risk than females.
- Age: Risk increases significantly with age, especially after 50-60 years.
- Smoking: The most significant risk factor for lung cancer.
- Yellow fingers: Often indicates heavy smoking or nicotine staining.
- Anxiety: Can be related to health concerns or symptoms.
- Peer pressure: May influence smoking or other risky behaviors.
- Chronic disease: Pre-existing conditions may increase risk.
- Fatigue: Can be an early symptom of lung cancer.
- Allergy: May indicate respiratory sensitivity.
- Wheezing: Respiratory symptom that may indicate lung problems.
- Alcohol consumption: Can compound other risk factors.
- Coughing: Persistent cough is a key lung cancer symptom.
- Shortness of breath: Important respiratory symptom.
- Swallowing difficulty: Can indicate advanced lung cancer or related conditions.
- Chest pain: Important symptom that may indicate lung problems.

Additional calculated risk factors for this patient:
- Overall risk score: {risk_score}/15 (Based on key risk factors)

Follow this step-by-step reasoning process:

1. Assess smoking status - the most critical risk factor for lung cancer.
2. Consider age - older patients have significantly higher risk.
3. Evaluate respiratory symptoms (coughing, wheezing, shortness of breath).
4. Check for physical signs (yellow fingers, chest pain).
5. Consider systemic symptoms (fatigue, swallowing difficulty).
6. Evaluate the combination of factors to determine overall risk.
7. Make your final prediction.

Patient's data:
- Gender: {row.get('gender', 'Unknown')} (M = male, F = female)
- Age: {row.get('age', 'Unknown')} years
- Smoking: {row.get('smoking', 'Unknown')} (0 = no, 1 = yes)
- Yellow fingers: {row.get('yellow_fingers', 'Unknown')} (0 = no, 1 = yes)
- Anxiety: {row.get('anxiety', 'Unknown')} (0 = no, 1 = yes)
- Peer pressure: {row.get('peer_pressure', 'Unknown')} (0 = no, 1 = yes)
- Chronic disease: {row.get('chronic_disease', 'Unknown')} (0 = no, 1 = yes)
- Fatigue: {row.get('fatigue', 'Unknown')} (0 = no, 1 = yes)
- Allergy: {row.get('allergy', 'Unknown')} (0 = no, 1 = yes)
- Wheezing: {row.get('wheezing', 'Unknown')} (0 = no, 1 = yes)
- Alcohol consuming: {row.get('alcohol_consuming', 'Unknown')} (0 = no, 1 = yes)
- Coughing: {row.get('coughing', 'Unknown')} (0 = no, 1 = yes)
- Shortness of breath: {row.get('shortness_of_breath', 'Unknown')} (0 = no, 1 = yes)
- Swallowing difficulty: {row.get('swallowing_difficulty', 'Unknown')} (0 = no, 1 = yes)
- Chest pain: {row.get('chest_pain', 'Unknown')} (0 = no, 1 = yes)
- Risk score: {risk_score}/15

Provide thorough step-by-step medical reasoning explaining your assessment. On the final line of your response, write "FINAL_PREDICTION: [0 or 1]" where 0 = low lung cancer risk, 1 = high lung cancer risk.
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

# ////////////////////////////
def create_user_friendly_summary(prediction, explanation, features):
    """Create a user-friendly summary of the lung cancer risk prediction"""
    
    # Create a short summary with key points
    summary = f"**Lung Cancer Risk Assessment**\n\n"
    
    if prediction == 1:
        summary += "⚠️ **Result: Higher risk of lung cancer detected**\n\n"
    else:
        summary += "✅ **Result: Lower risk of lung cancer detected**\n\n"
    
    summary += "**Key health metrics:**\n"
    if features.get('gender'):
        summary += f"- Gender: {features['gender']}\n"
    if features.get('age'):
        summary += f"- Age: {features['age']} years\n"
    if features.get('smoking') is not None:
        summary += f"- Smoking: {'Yes' if features['smoking'] == 1 else 'No'}\n"
    
    # Count symptoms
    symptoms = ['coughing', 'shortness_of_breath', 'chest_pain', 'wheezing', 'fatigue', 'swallowing_difficulty']
    symptom_count = sum(1 for s in symptoms if features.get(s) == 1)
    summary += f"- Respiratory/related symptoms: {symptom_count}/6\n"
    
    if features.get('yellow_fingers') == 1:
        summary += "- Yellow fingers: Yes\n"
    if features.get('chronic_disease') == 1:
        summary += "- Chronic disease: Yes\n"
    
    # Extract a concise explanation from the full explanation
    key_sentences = []
    important_patterns = [
        r"(?:smoking|age|symptoms|cough|breath|chest)[^.]*\.",
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
    
    # Add important disclaimer
    summary += "\n**Important Note:**\n"
    summary += "This assessment is for informational purposes only and should not replace professional medical diagnosis. "
    summary += "If you have concerns about lung cancer risk, please consult with a healthcare professional.\n"
    
    # Add a final closing message
    summary += "\n**Thank you for completing the lung cancer risk assessment.**\n"
    summary += "This chat session is now closed."
    
    return summary

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
        st.warning("⚠️ Higher risk of lung cancer detected")
    else:
        st.success("✅ Lower risk of lung cancer detected")

    st.metric("Prediction", "High Risk" if pred == 1 else "Low Risk")

    with st.expander("View detailed explanation"):
        st.write(explanation)
    
    # Add a visual indicator that the chat is closed
    st.info("📝 The chat session is now closed. Thank you for completing the assessment.")

# UI starts
st.title("🫁 Lung Cancer Risk Assessment Chat")

# Always show the welcome message
st.markdown("""
<div style="background-color: #eef1f5; border-left: 6px solid #FF4B4B;
            padding: 15px; margin-bottom: 20px; border-radius: 10px;
            font-size: 16px;">
<strong>Hello!</strong> Welcome to the lung cancer risk assessment.<br>
I'll ask you some health questions to evaluate your risk of lung cancer.<br>
This assessment considers various factors including lifestyle, symptoms, and medical history.<br>
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
        data_indicators = ['i am', 'i have', 'i do', 'i don\'t', 'i smoke', 'i cough', 'my age', 'years old', 'male', 'female']
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
                    result_dict = debate_with_llm_and_ml_models_v2_lung(st.session_state.user_data)
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