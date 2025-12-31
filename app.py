import streamlit as st
from spam_detection import train_model
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Background */
body {
    background: radial-gradient(circle at top, #1e3c72, #0f0f14);
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    animation: fadeIn 0.8s ease-in-out;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #b5b5c3;
    margin-bottom: 30px;
}

/* Icon */
.icon {
    font-size: 50px;
    text-align: center;
    margin-bottom: 10px;
    transition: transform 0.4s ease;
}
.icon:hover {
    transform: scale(1.15) rotate(5deg);
}

/* Textarea */
textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    background-color: rgba(255,255,255,0.05) !important;
    color: #ffffff !important;
}

textarea:focus {
    border: 1px solid #6c63ff !important;
    box-shadow: 0 0 12px rgba(108,99,255,0.6) !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #4facfe, #6c63ff);
    color: white;
    border-radius: 14px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 10px 25px rgba(108,99,255,0.4);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(108,99,255,0.6);
}

/* Result Card */
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 16px;
    animation: fadeInUp 0.6s ease;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
}

.success {
    background: rgba(0, 200, 150, 0.15);
    color: #00e6a8;
    border: 1px solid rgba(0, 230, 168, 0.4);
}

.error {
    background: rgba(255, 70, 70, 0.15);
    color: #ff4d4d;
    border: 1px solid rgba(255, 77, 77, 0.4);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 35px;
    font-size: 13px;
    color: #8a8aa3;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.96); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model, vectorizer = train_model()

# ---------------- UI ----------------
st.markdown("<div class='icon'>🛡️</div>", unsafe_allow_html=True)
st.markdown("<div class='title'>Email Spam Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning Based Email Classifier</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

email_text = st.text_area(
    "📧 Enter Email Message",
    placeholder="Paste or type the email content here...",
    height=160
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("🔍 Predict Spam", use_container_width=True)

if predict:
    with st.spinner("Analyzing email with ML model..."):
        time.sleep(1.2)

    if email_text.strip() == "":
        st.warning("⚠️ Please enter an email message.")
    else:
        vec = vectorizer.transform([email_text])
        pred = model.predict(vec)

        if pred[0] == 1:
            st.markdown(
                "<div class='result error'>🚫 SPAM EMAIL DETECTED</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result success'>✅ THIS EMAIL IS NOT SPAM</div>",
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Developed by Sandip Shihire | ML Project</div>", unsafe_allow_html=True)
