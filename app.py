from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
	page_title="Diabetes Risk Check",
	page_icon="D",
	layout="centered",
	initial_sidebar_state="collapsed",
)


MODEL_PATH = Path(__file__).with_name("model.pkl")
FEATURES = [
	"gender",
	"age",
	"hypertension",
	"heart_disease",
	"smoking_history",
	"bmi",
	"HbA1c_level",
	"blood_glucose_level",
]

GENDER_CODES = {"Female": 0, "Male": 1}
SMOKING_CODES = {
	"No Info": 0,
	"current": 1,
	"ever": 2,
	"former": 3,
	"never": 4,
	"not current": 5,
}


@st.cache_resource
def load_model():
	return joblib.load(MODEL_PATH)


st.markdown(
	"""
	<style>
		@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

		:root {
			--ink: #17211f;
			--muted: #64736e;
			--paper: #f5f7f2;
			--card: #ffffff;
			--line: #dce5df;
			--teal: #147d72;
			--teal-dark: #0d5e56;
			--coral: #e46f5c;
		}

		.stApp {
			background: radial-gradient(circle at 8% 0%, #dcefe8 0, transparent 34%),
						linear-gradient(135deg, var(--paper) 0%, #eef4ef 100%);
			color: var(--ink);
			font-family: 'DM Sans', sans-serif;
		}

		[data-testid="stHeader"] { background: transparent; }
		.block-container { max-width: 920px; padding: 3.5rem 1.25rem 3rem; }

		.eyebrow {
			color: var(--teal);
			font-size: 0.75rem;
			font-weight: 700;
			letter-spacing: 0.12em;
			text-transform: uppercase;
			margin-bottom: 0.7rem;
		}

		h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
		h1 { font-size: clamp(2.4rem, 7vw, 4.7rem); line-height: 0.98; letter-spacing: 0; margin: 0; }
		.intro { color: var(--muted); font-size: 1.05rem; line-height: 1.6; max-width: 650px; margin: 1.2rem 0 2rem; }

		.form-shell {
			background: rgba(255, 255, 255, 0.82);
			border: 1px solid rgba(220, 229, 223, 0.95);
			border-radius: 8px;
			box-shadow: 0 18px 55px rgba(34, 67, 59, 0.09);
			padding: 1.25rem 1.5rem 1.5rem;
		}

		.section-label {
			color: var(--teal-dark);
			font-family: 'Space Grotesk', sans-serif;
			font-size: 1.05rem;
			font-weight: 700;
			margin: 0.8rem 0 0.35rem;
		}

		[data-testid="stForm"] { border: 0; padding: 0; }
		label { color: var(--ink) !important; font-weight: 600 !important; }
		[data-baseweb="input"], [data-baseweb="select"] { border-radius: 6px; }
		.stButton > button, [data-testid="stFormSubmitButton"] button {
			background: var(--teal); border: 0; border-radius: 6px; color: white;
			font-weight: 700; min-height: 3rem; transition: background 160ms ease, transform 160ms ease;
		}
		.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
			background: var(--teal-dark); color: white; transform: translateY(-1px);
		}
		.result-good, .result-alert {
			border-radius: 8px; padding: 1.2rem 1.35rem; margin-top: 1.5rem;
		}
		.result-good { background: #e1f2e9; border: 1px solid #b9dfca; }
		.result-alert { background: #fff0e9; border: 1px solid #f2c2b3; }
		.result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; }
		.result-copy { color: #4c5d57; margin-top: 0.35rem; }
		.footnote { color: var(--muted); font-size: 0.82rem; line-height: 1.5; margin-top: 1.5rem; }
	</style>
	""",
	unsafe_allow_html=True,
)


st.markdown('<div class="eyebrow">Personal health snapshot</div>', unsafe_allow_html=True)
st.title("Diabetes risk check")
st.markdown(
	'<p class="intro">Enter the health details below for a quick prediction from your trained machine-learning model.</p>',
	unsafe_allow_html=True,
)

try:
	model = load_model()
except Exception as error:
	st.error(f"The model could not be loaded from {MODEL_PATH.name}: {error}")
	st.stop()

with st.container(border=True):
	with st.form("prediction_form"):
		st.markdown('<div class="section-label">About you</div>', unsafe_allow_html=True)
		personal_col1, personal_col2 = st.columns(2)
		with personal_col1:
			gender = st.selectbox("Gender", list(GENDER_CODES))
			age = st.number_input("Age", min_value=1, max_value=120, value=40, step=1)
		with personal_col2:
			smoking_history = st.selectbox("Smoking history", list(SMOKING_CODES))
			bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1, format="%.1f")

		st.markdown('<div class="section-label">Medical measures</div>', unsafe_allow_html=True)
		health_col1, health_col2 = st.columns(2)
		with health_col1:
			hypertension = st.selectbox("Hypertension", ["No", "Yes"])
			heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
		with health_col2:
			hba1c_level = st.number_input("HbA1c level", min_value=2.0, max_value=20.0, value=5.5, step=0.1, format="%.1f")
			blood_glucose_level = st.number_input("Blood glucose level", min_value=40, max_value=500, value=120, step=1)

		submitted = st.form_submit_button("Check prediction", use_container_width=True)

if submitted:
	input_data = pd.DataFrame(
		[[
			GENDER_CODES[gender],
			age,
			int(hypertension == "Yes"),
			int(heart_disease == "Yes"),
			SMOKING_CODES[smoking_history],
			bmi,
			hba1c_level,
			blood_glucose_level,
		]],
		columns=FEATURES,
	)
	prediction = int(model.predict(input_data)[0])
	probability = float(model.predict_proba(input_data)[0, 1])

	if prediction == 1:
		st.markdown(
			f'<div class="result-alert"><div class="result-title">Higher diabetes risk indicated</div><div class="result-copy">Model probability: <strong>{probability:.1%}</strong>. Please discuss these results with a qualified healthcare professional.</div></div>',
			unsafe_allow_html=True,
		)
	else:
		st.markdown(
			f'<div class="result-good"><div class="result-title">Lower diabetes risk indicated</div><div class="result-copy">Model probability: <strong>{probability:.1%}</strong>. This is a prediction, not a diagnosis.</div></div>',
			unsafe_allow_html=True,
		)

st.markdown(
	'<div class="footnote">For educational use only. This tool does not replace medical advice, testing, or diagnosis.</div>',
	unsafe_allow_html=True,
)
