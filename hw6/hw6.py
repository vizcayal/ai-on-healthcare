import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
load_dotenv()

# Standard / CoT Output Schema
class StrokePrediction(BaseModel):
    reasoning: str = Field(description="Clinical reasoning explaining the patient's stroke risk.")
    prediction: int = Field(description="Final binary prediction: 1 (high stroke risk) or 0 (low risk).")

# Tree of Thoughts (ToT) Output Schema
class StrokePredictionToT(BaseModel):
    branch_a_high_risk: str = Field(description="Branch A: Evidence supporting High Risk hypothesis.")
    branch_b_low_risk: str = Field(description="Branch B: Evidence supporting Low Risk / Protective hypothesis.")
    branch_c_interactions: str = Field(description="Branch C: Analysis of factor interactions & confounding markers.")
    consensus_reasoning: str = Field(description="Final synthesis comparing all branches to arrive at the prediction.")
    prediction: int = Field(description="Final binary prediction: 1 (high stroke risk) or 0 (low risk).")

# Graph of Thoughts (GoT) Output Schema
class StrokePredictionGoT(BaseModel):
    node1_cardiovascular_analysis: str = Field(description="Node 1: Independent Cardiovascular & Age risk sub-graph analysis.")
    node2_metabolic_analysis: str = Field(description="Node 2: Independent Metabolic & Lifestyle risk sub-graph analysis.")
    node3_graph_aggregation: str = Field(description="Node 3: Convergence node merging Node 1 and Node 2 for cross-system interaction.")
    consensus_verdict: str = Field(description="Node 4: Final graph synthesis and clinical decision.")
    prediction: int = Field(description="Final binary prediction: 1 (high stroke risk) or 0 (low risk).")

# Function to serialize patient data into natural language
def serialize_patient(row: pd.Series) -> str:
    """Convert a dataset row into a natural-language patient profile."""
    return f"""
    Patient Profile:
    - Age: {row['age']} years
    - Gender: {row['gender']}
    - BMI (Body Mass Index): {row['bmi']}
    - Average glucose level: {row['avg_glucose_level']} mg/dL
    - Hypertension: {'Yes' if row['hypertension'] == 1 else 'No'}
    - Heart Disease: {'Yes' if row['heart_disease'] == 1 else 'No'}
    - Marital status (Ever married): {row['ever_married']}
    - Work type: {row['work_type']}
    - Residence type: {row['Residence_type']}
    - Smoking status: {row['smoking_status']}
    """

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "data", "healthcare-dataset-stroke-data.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join("hw6", "data", "healthcare-dataset-stroke-data.csv")

df = pd.read_csv(csv_path)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Separate positive and negative patients
pos_patients = df[df['stroke'] == 1].copy()
neg_patients = df[df['stroke'] == 0].copy()

# Sample 10 positive and 10 negative exemplars for Few-Shot prompts
pos_exemplars = pos_patients.sample(n=10, random_state=42)
neg_exemplars = neg_patients.sample(n=10, random_state=42)

# Exclude exemplars from pool to prevent data leakage
pos_test_pool = pos_patients.drop(pos_exemplars.index)
neg_test_pool = neg_patients.drop(neg_exemplars.index)

# Sample 25 positive and 25 negative test patients for balanced evaluation (50 total)
pos_test = pos_test_pool.sample(n=25, random_state=100)
neg_test = neg_test_pool.sample(n=25, random_state=100)

test_set = pd.concat([pos_test, neg_test]).sample(frac=1.0, random_state=77).reset_index()

# ---------------------------------------------------------
# PROMPT CONSTRUCTIONS FOR THE 5 METHODS
# ---------------------------------------------------------

# Method 1: Zero-Shot System Prompt
ZERO_SHOT_PROMPT = (
    "You are an expert medical analyst. Assess the risk of the patient suffering a stroke. "
    "Return a binary prediction: 1 for high stroke risk, 0 for low risk, along with clinical reasoning."
)

# Method 2: Standard Few-Shot Prompt (10 Positive + 10 Negative Exemplars)
std_few_shot_list = ["Here are 20 clinical reference examples:"]
for i, (_, row) in enumerate(neg_exemplars.iterrows(), 1):
    std_few_shot_list.append(
        f"--- Example {i} (Low Risk) ---\n"
        f"{serialize_patient(row).strip()}\n"
        f"Clinical Rationale: Low baseline risk factors.\n"
        f"Prediction: 0\n"
    )
for i, (_, row) in enumerate(pos_exemplars.iterrows(), 1):
    std_few_shot_list.append(
        f"--- Example {i+10} (High Risk) ---\n"
        f"{serialize_patient(row).strip()}\n"
        f"Clinical Rationale: High stroke risk due to compounding patient conditions.\n"
        f"Prediction: 1\n"
    )
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + "\n\n" + "\n".join(std_few_shot_list)

# Method 3: Chain-of-Thought (CoT) Few-Shot Prompt
cot_few_shot_list = ["Here are 20 step-by-step Chain-of-Thought clinical reference examples:"]
for i, (_, row) in enumerate(neg_exemplars.iterrows(), 1):
    cot_few_shot_list.append(
        f"--- Example {i} (Low Risk - 0) ---\n"
        f"{serialize_patient(row).strip()}\n"
        f"Chain-of-Thought Reasoning:\n"
        f"- Step 1 (Demographics): Age {row['age']}.\n"
        f"- Step 2 (Cardiovascular): Hypertension={'Yes' if row['hypertension']==1 else 'No'}, Heart Disease={'Yes' if row['heart_disease']==1 else 'No'}.\n"
        f"- Step 3 (Metabolic): Glucose {row['avg_glucose_level']} mg/dL, BMI {row['bmi']}.\n"
        f"- Step 4 (Synthesis): Overall clinical profile does not meet high risk threshold.\n"
        f"Prediction: 0\n"
    )
for i, (_, row) in enumerate(pos_exemplars.iterrows(), 1):
    cot_few_shot_list.append(
        f"--- Example {i+10} (High Risk - 1) ---\n"
        f"{serialize_patient(row).strip()}\n"
        f"Chain-of-Thought Reasoning:\n"
        f"- Step 1 (Demographics): Age {row['age']} (advanced age is a major stroke driver).\n"
        f"- Step 2 (Cardiovascular): Hypertension={'Yes' if row['hypertension']==1 else 'No'}, Heart Disease={'Yes' if row['heart_disease']==1 else 'No'}.\n"
        f"- Step 3 (Metabolic): Glucose {row['avg_glucose_level']} mg/dL, BMI {row['bmi']}.\n"
        f"- Step 4 (Synthesis): Multiple compounding vascular/metabolic risks indicate high stroke probability.\n"
        f"Prediction: 1\n"
    )
COT_PROMPT = (
    "You are an expert medical analyst. Think step-by-step before making a diagnosis.\n"
    "Analyze risk in 4 steps:\n"
    "Step 1: Non-modifiable demographics (age, gender).\n"
    "Step 2: Cardiovascular conditions (hypertension, heart disease).\n"
    "Step 3: Metabolic & lifestyle factors (glucose, BMI, smoking).\n"
    "Step 4: Clinical synthesis to decide binary prediction (1 for high risk, 0 for low risk).\n\n"
    + "\n".join(cot_few_shot_list)
)

# Method 4: Tree of Thoughts (ToT) Few-Shot Prompt
TOT_PROMPT = (
    "You are an expert clinical diagnostician using a Tree of Thoughts (ToT) decision framework.\n"
    "To evaluate the patient's stroke risk, systematically explore 3 reasoning branches before reaching a verdict:\n\n"
    "1. Branch A (High-Risk Hypothesis): Evaluate all factors supporting high risk (e.g. age > 60, hypertension, glucose > 200, heart disease, obesity).\n"
    "2. Branch B (Low-Risk / Protective Hypothesis): Evaluate all protective factors supporting low risk (e.g. young age, normal BMI < 25, normal glucose, absence of vascular disease).\n"
    "3. Branch C (Confounding & Factor Interaction Path): Analyze interactions between contradictory markers (e.g. high glucose with young age or normal BMI with hypertension).\n"
    "4. Consensus Synthesis: Compare the strength of evidence across Branch A, B, and C to output the final binary prediction (1 for high risk, 0 for low risk).\n\n"
    + "\n".join(cot_few_shot_list)
)

# Method 5: Graph of Thoughts (GoT) Few-Shot Prompt
GOT_PROMPT = (
    "You are an expert medical diagnostician using a Graph of Thoughts (GoT) framework.\n"
    "Process patient stroke risk through a 4-node Directed Graph:\n\n"
    "- Node 1 (Cardiovascular Sub-Graph): Independently evaluate Age, Hypertension, and Heart Disease.\n"
    "- Node 2 (Metabolic Sub-Graph): Independently evaluate Glucose level, BMI, and Smoking status.\n"
    "- Node 3 (Graph Aggregation Node): Merge outputs from Node 1 and Node 2. Evaluate how vascular and metabolic factors compound each other.\n"
    "- Node 4 (Consensus Verdict Node): Integrate the full graph network to decide final binary prediction (1 for high risk, 0 for low risk).\n\n"
    + "\n".join(cot_few_shot_list)
)

# OpenAI Client Setup
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set the OPENAI_API_KEY environment variable in your .env file")

client = OpenAI(api_key=api_key)
MODEL_NAME = "gpt-5.4-mini"

# Evaluation Runner
def evaluate_method(method_key: str):
    if method_key == 'zero_shot':
        mode_name = "1. Zero-Shot"
        system_content = ZERO_SHOT_PROMPT
        schema_class = StrokePrediction
    elif method_key == 'few_shot':
        mode_name = "2. Few-Shot (Standard)"
        system_content = FEW_SHOT_PROMPT
        schema_class = StrokePrediction
    elif method_key == 'cot':
        mode_name = "3. Chain of Thoughts (CoT)"
        system_content = COT_PROMPT
        schema_class = StrokePrediction
    elif method_key == 'tot':
        mode_name = "4. Tree of Thoughts (ToT)"
        system_content = TOT_PROMPT
        schema_class = StrokePredictionToT
    elif method_key == 'got':
        mode_name = "5. Graph of Thoughts (GoT)"
        system_content = GOT_PROMPT
        schema_class = StrokePredictionGoT

    print(f"\n>>> Running [{mode_name}] on 50 balanced patients (25 stroke / 25 non-stroke) with {MODEL_NAME}...\n")
    
    results = []
    for idx, (_, patient_row) in enumerate(test_set.iterrows(), 1):
        patient_text = serialize_patient(patient_row)
        actual_stroke = int(patient_row['stroke'])
        patient_id = patient_row['index']
        
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": patient_text},
                ],
                response_format=schema_class,
                temperature=0.0,
            )
            result = completion.choices[0].message.parsed
            pred_val = result.prediction
        except Exception as e:
            print(f"Error for sample {idx}: {e}")
            pred_val = 0
            
        results.append({
            "sample_num": idx,
            "patient_id": patient_id,
            "actual": actual_stroke,
            "pred": pred_val
        })
        print(f"[{mode_name}] Sample {idx}/50 (ID {patient_id}): Actual={actual_stroke}, Pred={pred_val}")
        
    actuals = [r["actual"] for r in results]
    preds = [r["pred"] for r in results]
    
    tp = sum(1 for a, p in zip(actuals, preds) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(actuals, preds) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(actuals, preds) if a == 0 and p == 0)
    fn = sum(1 for a, p in zip(actuals, preds) if a == 1 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(actuals) if len(actuals) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "mode": mode_name,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

# Execute All 5 Methods
m1_zero = evaluate_method('zero_shot')
m2_few = evaluate_method('few_shot')
m3_cot = evaluate_method('cot')
m4_tot = evaluate_method('tot')
m5_got = evaluate_method('got')

# Print Final 5-Way Comparison Table
print("\n" + "="*115)
print("FINAL 5-METHOD COMPARATIVE EVALUATION SUMMARY (50 Patients: 25 Stroke / 25 Non-Stroke)")
print(f"Model: {MODEL_NAME} | Comparing 5 Core Prompting Strategies (Including GoT)")
print("="*115)
print(f"{'Metric':<22} | {'1. Zero-Shot':<12} | {'2. Few-Shot':<12} | {'3. CoT':<12} | {'4. ToT':<12} | {'5. GoT':<14}")
print("-" * 115)
print(f"{'True Positives (TP)':<22} | {m1_zero['tp']:<12} | {m2_few['tp']:<12} | {m3_cot['tp']:<12} | {m4_tot['tp']:<12} | {m5_got['tp']:<14}")
print(f"{'False Positives (FP)':<22} | {m1_zero['fp']:<12} | {m2_few['fp']:<12} | {m3_cot['fp']:<12} | {m4_tot['fp']:<12} | {m5_got['fp']:<14}")
print(f"{'True Negatives (TN)':<22} | {m1_zero['tn']:<12} | {m2_few['tn']:<12} | {m3_cot['tn']:<12} | {m4_tot['tn']:<12} | {m5_got['tn']:<14}")
print(f"{'False Negatives (FN)':<22} | {m1_zero['fn']:<12} | {m2_few['fn']:<12} | {m3_cot['fn']:<12} | {m4_tot['fn']:<12} | {m5_got['fn']:<14}")
print("-" * 115)
print(f"{'Accuracy':<22} | {m1_zero['accuracy']:<12.2%} | {m2_few['accuracy']:<12.2%} | {m3_cot['accuracy']:<12.2%} | {m4_tot['accuracy']:<12.2%} | {m5_got['accuracy']:<14.2%}")
print(f"{'Precision':<22} | {m1_zero['precision']:<12.2%} | {m2_few['precision']:<12.2%} | {m3_cot['precision']:<12.2%} | {m4_tot['precision']:<12.2%} | {m5_got['precision']:<14.2%}")
print(f"{'Recall':<22} | {m1_zero['recall']:<12.2%} | {m2_few['recall']:<12.2%} | {m3_cot['recall']:<12.2%} | {m4_tot['recall']:<12.2%} | {m5_got['recall']:<14.2%}")
print(f"{'F1 Score':<22} | {m1_zero['f1']:<12.2%} | {m2_few['f1']:<12.2%} | {m3_cot['f1']:<12.2%} | {m4_tot['f1']:<12.2%} | {m5_got['f1']:<14.2%}")
print("="*115)