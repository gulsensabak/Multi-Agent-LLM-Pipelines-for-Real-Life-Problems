import pandas as pd
import streamlit as st
import re
import json
from langchain_ollama import ChatOllama

# Required health features
required_features = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

def extract_features_from_text(text, model_name="mistral:latest"):
    # First provide clear guidance on expected value formats
    prompt = f"""
You are an intelligent medical assistant extracting health data from text. Extract these features:

- age (in years, integer)
- gender (1 = female, 2 = male, integer)
- height (in cm, integer)
- weight (in kg, integer)
- ap_hi (systolic blood pressure, integer)
- ap_lo (diastolic blood pressure, integer)
- cholesterol (1 = normal, 2 = above normal, 3 = well above normal, integer)
- gluc (1 = normal, 2 = above normal, 3 = well above normal, integer)
- smoke (0 = no, 1 = yes, integer)
- alco (0 = no, 1 = yes, integer)
- active (0 = no, 1 = yes, integer)

User text: "{text}"

RULES:
1. Infer values where possible, but do NOT make wild guesses
2. Use null for any values not provided or that cannot be reasonably inferred
3. Convert text descriptions to appropriate numerical values (e.g., "high cholesterol" = 3)
4. Ensure all values are in the correct format (integers for all fields)
5. If blood pressure is given as a single value like "130/85", extract ap_hi=130, ap_lo=85

Respond EXCLUSIVELY with a valid JSON object in this exact format:
{{
  "age": integer or null,
  "gender": integer or null,
  "height": integer or null,
  "weight": integer or null,
  "ap_hi": integer or null,
  "ap_lo": integer or null,
  "cholesterol": integer or null,
  "gluc": integer or null,
  "smoke": integer or null,
  "alco": integer or null,
  "active": integer or null
}}

NO explanations, NO additional text, ONLY the JSON object.
"""

    llm = ChatOllama(
        model=model_name,
        temperature=0.1,
        top_p=0.95,
        repeat_penalty=1.2
    )

    response = llm.invoke(prompt)
    
    # Convert AIMessage to string if needed
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)
    
    # Enhanced JSON extraction
    # First, try to extract JSON using regex
    import re
    import json
    
    # Clean the response text to help with JSON parsing
    cleaned_text = response_text.strip()
    
    # Try multiple approaches to extract JSON
    try:
        # First try: direct JSON parsing if the whole response is valid JSON
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
            
        # Second try: regex to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
            
        # Third try: if we can't find proper JSON, return empty dict
        st.error(f"Failed to extract valid JSON from LLM response. Raw response: {cleaned_text[:100]}...")
        return {}
        
    except Exception as e:
        st.error(f"Error processing features extraction: {str(e)}")
        return {}
        
def validate_extracted_features(features):
    """Validates and normalizes the extracted features"""
    validated = {}
    
    # Define valid ranges and defaults for each feature
    validations = {
        "age": {"min": 1, "max": 120},
        "gender": {"valid_values": [1, 2]},
        "height": {"min": 50, "max": 250},  # cm
        "weight": {"min": 20, "max": 300},  # kg
        "ap_hi": {"min": 70, "max": 250},   # systolic
        "ap_lo": {"min": 40, "max": 150},   # diastolic
        "cholesterol": {"valid_values": [1, 2, 3]},
        "gluc": {"valid_values": [1, 2, 3]},
        "smoke": {"valid_values": [0, 1]},
        "alco": {"valid_values": [0, 1]},
        "active": {"valid_values": [0, 1]}
    }
    
    for feature, value in features.items():
        # Skip null values
        if value is None:
            validated[feature] = None
            continue
            
        try:
            # Convert to integer if possible
            value = int(value)
            
            # Validate against range or valid values
            if "min" in validations.get(feature, {}):
                if value < validations[feature]["min"] or value > validations[feature]["max"]:
                    st.warning(f"Warning: {feature} value {value} is outside normal range. Please verify.")
                    
            if "valid_values" in validations.get(feature, {}):
                if value not in validations[feature]["valid_values"]:
                    st.warning(f"Warning: {feature} value {value} is not in valid values {validations[feature]['valid_values']}.")
                    
            validated[feature] = value
        except (ValueError, TypeError):
            validated[feature] = None
            st.warning(f"Warning: {feature} value could not be converted to integer.")
    
    return validated

def get_bmi_display(bmi):
    """Helper function to safely format BMI value and category"""
    if bmi is None:
        return "Unknown", "Unknown"
        
    bmi_value = f"{bmi:.1f}"
    
    # BMI categories
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
        
    return bmi_value, bmi_category
    
def llm_predict_cardiovascular(features, model_name="mistral:latest", temperature=0.1, top_p=0.95, repeat_penalty=1.2):
    row = features
    risk_score = 0
    
    # Calculate BMI if possible
    bmi = None
    if row.get('height') and row.get('weight') and row['height'] > 0:
        height_m = row['height'] / 100
        bmi = row['weight'] / (height_m * height_m)
    
    # Age risk (older patients have higher risk)
    if row.get('age') and row['age'] > 55:
        risk_score += 2
    
    # Blood pressure risk
    if (row.get('ap_hi') and row['ap_hi'] >= 140) or (row.get('ap_lo') and row['ap_lo'] >= 90):
        risk_score += 2
    
    # Cholesterol risk
    if row.get('cholesterol'):
        if row['cholesterol'] == 3:
            risk_score += 2
        elif row['cholesterol'] == 2:
            risk_score += 1
    
    # BMI risk
    if bmi is not None and bmi >= 30:
        risk_score += 1
    
    # Glucose risk
    if row.get('gluc') and row['gluc'] == 3:
        risk_score += 1
    
    # Smoking risk
    if row.get('smoke') and row['smoke'] == 1:
        risk_score += 2
        
    # Physical inactivity risk
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
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)

    match = re.search(r"FINAL_PREDICTION:\s*([01])", response_text)
    if match:
        prediction = int(match.group(1))
    else:
        prediction = 0

    return prediction, response_text

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

def main():
    st.set_page_config(page_title="Cardiovascular Risk Predictor", page_icon="🫀")
    st.title("🫀 Cardiovascular Disease Risk Predictor")
    
    # Add sample inputs as guidance for users
    with st.expander("✨ Sample inputs - click for examples"):
        st.markdown("""
        Try using inputs like these:
        - "I'm a 45 year old man, 178cm tall and weigh 85kg. My blood pressure is 130/85. I don't smoke and I exercise regularly. My cholesterol and glucose are normal."
        - "Female, 67 years old with high blood pressure (145/95). I'm 160cm and 70kg. High cholesterol, normal glucose. I don't smoke or drink but I'm not very active."
        - "52 year old male, 175cm, 95kg. BP 155/100. High cholesterol and glucose. Smoker, drinks occasionally, sedentary lifestyle."
        """)

    st.markdown("### Enter your health information in natural language")
    st.markdown("Include details like age, gender, height, weight, blood pressure, cholesterol level, glucose level, smoking/drinking habits, and activity level.")

    user_input = st.text_area("Describe yourself:", height=120)

    col1, col2 = st.columns(2)
    
    with col1:
        model_selection = st.selectbox("Select Ollama model:", ["mistral:latest", "llama3", "phi3:latest", "gemma:latest"], index=0)

    with col2:
        with st.expander("Advanced Model Parameters"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
            top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)
            repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.2, 0.1)

    if st.button("Predict Cardiovascular Risk", type="primary"):
        if not user_input.strip():
            st.error("Please enter your health information before predicting.")
            return
            
        with st.spinner("Extracting data and predicting..."):
            features = extract_features_from_text(user_input, model_selection)
            
            # Validate and normalize the extracted features
            features = validate_extracted_features(features)
            
            # Check for missing features
            missing_features = [feature for feature in required_features if feature not in features or features[feature] is None]
            
            if missing_features:
                st.warning("⚠️ Some information is missing, which may affect prediction accuracy")
                
                # Display missing features in a more user-friendly format
                st.markdown("### Missing Information")
                
                missing_features_display = {
                    "age": "Age (in years)",
                    "gender": "Gender",
                    "height": "Height (in cm)",
                    "weight": "Weight (in kg)",
                    "ap_hi": "Systolic blood pressure",
                    "ap_lo": "Diastolic blood pressure",
                    "cholesterol": "Cholesterol level",
                    "gluc": "Glucose level",
                    "smoke": "Smoking status",
                    "alco": "Alcohol consumption",
                    "active": "Physical activity level"
                }
                
                for feature in missing_features:
                    st.markdown(f"- {missing_features_display.get(feature, feature)}")
                
                st.markdown("### Suggestion")
                st.markdown("For a more accurate prediction, consider providing the missing information above.")
                
                # Show what was successfully extracted
                with st.expander("View extracted features"):
                    st.json(features)
                    
                # If we have enough critical features, still offer to run the prediction
                critical_features = ["age", "weight", "ap_hi", "ap_lo", "cholesterol"]
                critical_missing = [f for f in critical_features if f in missing_features]
                
                if len(critical_missing) <= 2 and "age" not in critical_missing:
                    # We can still try to predict with some missing data
                    if st.button("Continue with incomplete data"):
                        pred, explanation = llm_predict_cardiovascular(features, model_selection, temperature, top_p, repeat_penalty)
                        display_prediction_results(pred, explanation, features)
            else:
                # We have all required features
                pred, explanation = llm_predict_cardiovascular(features, model_selection, temperature, top_p, repeat_penalty)
                display_prediction_results(pred, explanation, features)

if __name__ == "__main__":
    main()