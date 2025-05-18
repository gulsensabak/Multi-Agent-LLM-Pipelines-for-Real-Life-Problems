from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma2:latest", temperature=0.1)

def detect_problem_area(user_input):
    prompt = f'''
You are a medical assistant. Decide whether the user's complaint is related to the heart or lungs.

Only respond with one word: "heart" or "lung".

Complaint: "{user_input}"
Answer:'''
    response = llm.invoke(prompt)
    answer = response.content.strip().lower()
    if "heart" in answer:
        return "heart"
    elif "lung" in answer:
        return "lung"
    return "heart"