# tool_loader.py (güncellenmiş hali)
from langchain.agents import Tool
from lung_tools import svm_lung_predictor, rf_lung_predictor
from heart_tools import svm_heart_predictor, rf_heart_predictor

svm_tool_lung = Tool(
    name="svm_lung_predictor",
    func=svm_lung_predictor,
    description="Use this tool to predict lung cancer using SVM with patient features."
)

rf_tool_lung = Tool(
    name="rf_lung_predictor",
    func=rf_lung_predictor,
    description="Use this tool to predict lung cancer using Random Forest with patient features."
)

svm_tool_heart = Tool(
    name="svm_heart_predictor",
    func=svm_heart_predictor,
    description="Use this tool to predict cardiovascular disease using SVM with patient features."
)

rf_tool_heart = Tool(
    name="rf_heart_predictor",
    func=rf_heart_predictor,
    description="Use this tool to predict cardiovascular disease using Random Forest with patient features."
)

lung_tools = [svm_tool_lung, rf_tool_lung]
heart_tools = [svm_tool_heart, rf_tool_heart]
