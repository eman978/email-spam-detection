import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import random

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 20 Spam & 20 Ham Examples ─────────────────────────────────────────────────
SPAM_EXAMPLES = [
    "WINNER!! You have been selected to receive a £1000 cash prize! Call 09061701461 to claim NOW. T&Cs apply.",
    "Congratulations! You've won a FREE iPhone 15! Click here to claim your reward before it expires: www.freeprize.net",
    "URGENT: Your bank account has been suspended. Verify your details immediately at www.secure-bank-login.com",
    "You have been specially selected for a FREE holiday to Maldives! Call 0800-FREE-NOW to claim. Limited offer!",
    "Make $5000 per week working from home! Guaranteed income. No experience needed. Click NOW: www.easymoney.biz",
    "FREE entry! Text WIN to 88888 and win a brand new car! Only 10 winners. Don't miss out!",
    "ALERT: Your PayPal account has been compromised. Reset your password NOW: www.paypal-secure-reset.com",
    "Exclusive deal! Buy 1 get 5 FREE! Limited stock. Order NOW and get 90% discount. Offer expires midnight!",
    "You've been pre-approved for a £50,000 loan! No credit check. Apply NOW: www.instant-loans-uk.com",
    "Dear customer, your Amazon account will be CLOSED unless you verify your info here: www.amazon-verify.net",
    "HOT SINGLES IN YOUR AREA want to meet you! Click here FREE: www.dating-adults.net. Unsubscribe anytime.",
    "CLAIM YOUR PRIZE! You are our 1,000,000th visitor! Click to collect your $500 gift card immediately!",
    "Lose 30 pounds in 30 days GUARANTEED! Doctor-approved pill. Buy 2 get 1 FREE. Limited time only!",
    "Your parcel could not be delivered. Pay £2.99 redelivery fee here: www.royal-mail-delivery-uk.com",
    "LAST CHANCE: Your subscription expires TODAY. Renew now and get 3 months FREE: www.netflix-offer.net",
    "Earn UNLIMITED cash with our crypto investment plan! 300% returns guaranteed. Invest $100 get $400 back!",
    "SIX FIGURE INCOME from home! Our proven system made 2000+ people rich. Start today FREE: www.richfast.com",
    "IMPORTANT NOTICE: You owe £450 in unpaid taxes. Pay now to avoid legal action: www.hmrc-payment.com",
    "FREE adult content unlocked for YOU! Visit www.xxx-freezone.com — No credit card needed. Act fast!",
    "Congratulations! Your mobile number has won the weekly draw! Prize: £3000 cash. Reply CLAIM to 80085.",
]

HAM_EXAMPLES = [
    "Hey, are you free for lunch tomorrow? I was thinking we could try that new Italian place near the office.",
    "Hi Mum, just letting you know I arrived safely. Call me when you're free this evening.",
    "Can you send me the report before the meeting? I need to review it beforehand. Thanks!",
    "Don't forget we have football practice at 6pm today. Bring your boots!",
    "Hey, happy birthday! Hope you have an amazing day. Let's catch up soon.",
    "The meeting has been rescheduled to 3pm on Friday. Please update your calendar accordingly.",
    "I left my keys at your place yesterday. Can I come by and pick them up later today?",
    "Just finished the book you recommended — it was absolutely brilliant! What should I read next?",
    "Dinner is in the oven. Should be ready by 7. Let me know if you're running late.",
    "Can you pick up some milk and bread on your way home? Also we're out of eggs.",
    "Thanks for covering my shift last week. I really appreciate it, I'll return the favour soon.",
    "Your 2pm appointment with Dr. Ahmed is confirmed for Thursday 10th April. Please arrive 10 mins early.",
    "Hey, did you watch the match last night? That last-minute goal was absolutely insane!",
    "I've attached my CV as requested. Please let me know if you need any additional information.",
    "The project deadline has been extended to next Friday. Let's use the extra time to polish things up.",
    "Just checked — the train leaves at 8:47am. Platform 3. See you at the station!",
    "Hey, hope you're feeling better. Let me know if you need anything, happy to drop supplies off.",
    "Your order #45821 has been dispatched and will arrive within 2-3 working days. Track at royalmail.com",
    "Quick reminder: team standup is at 9:30am tomorrow. We'll be reviewing sprint goals.",
    "Great work on the presentation today! The client was really impressed with your data analysis.",
]

# ── Session State Init ────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "spam_index" not in st.session_state:
    st.session_state.spam_index = -1
if "ham_index" not in st.session_state:
    st.session_state.ham_index = -1

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg:            #0b0e14;
        --bg-panel:       #11151d;
        --bg-elevated:    #161b26;
        --border:         #232a38;
        --text-primary:   #eef1f6;
        --text-secondary: #a7b0c0;
        --text-muted:     #6b7385;
        --accent:         #4f7cff;
        --accent-hover:   #3f68e6;
        --danger:         #ef5a5a;
        --danger-bg:      rgba(239, 90, 90, 0.10);
        --danger-border:  rgba(239, 90, 90, 0.35);
        --success:        #2fbf8f;
        --success-bg:     rgba(47, 191, 143, 0.10);
        --success-border: rgba(47, 191, 143, 0.35);
        --warning:        #e0a326;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text-primary);
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-panel) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-primary);
    }
    [data-testid="stSidebar"] .hero-sub {
        color: var(--text-secondary) !important;
    }

    /* Sidebar navigation label */
    .sidebar-section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 4px 0 10px 0;
    }

    /* Radio group (page navigation) — fix disappearing text on selection */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 4px;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 9px 12px !important;
        margin: 0 !important;
        transition: background 0.12s, border-color 0.12s;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: var(--bg-elevated);
        border-color: var(--border);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span,
    [data-testid="stSidebar"] div[role="radiogroup"] label div {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 0.92rem;
        opacity: 1 !important;
    }
    /* Selected radio option */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(79, 124, 255, 0.14);
        border-color: rgba(79, 124, 255, 0.45);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
        color: var(--text-primary) !important;
    }
    /* Radio circle marker itself */
    [data-testid="stSidebar"] div[role="radiogroup"] label [data-baseweb="radio"] > div:first-child {
        border-color: var(--text-muted) !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) [data-baseweb="radio"] > div:first-child {
        border-color: var(--accent) !important;
        background: var(--accent) !important;
    }

    /* Alerts inside sidebar (success / warning boxes) */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: var(--text-primary) !important;
    }

    hr, [data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
    }

    /* ── Typography ──────────────────────────────────────────────────────── */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.4rem;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-secondary);
        letter-spacing: 0.04em;
    }

    /* ── Result Card ─────────────────────────────────────────────────────── */
    .result-card {
        border-radius: 12px;
        padding: 26px 28px;
        margin: 16px 0;
        border: 1px solid var(--border);
        background: var(--bg-elevated);
    }
    .spam-card {
        background: var(--danger-bg);
        border-color: var(--danger-border) !important;
    }
    .ham-card {
        background: var(--success-bg);
        border-color: var(--success-border) !important;
    }
    .result-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .result-value {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        line-height: 1;
        letter-spacing: -0.01em;
    }
    .spam-value { color: var(--danger); }
    .ham-value  { color: var(--success); }

    .conf-bar-wrap {
        background: rgba(255,255,255,0.06);
        border-radius: 99px;
        height: 6px;
        margin-top: 12px;
        overflow: hidden;
    }
    .conf-bar-fill-spam {
        height: 100%;
        border-radius: 99px;
        background: var(--danger);
    }
    .conf-bar-fill-ham {
        height: 100%;
        border-radius: 99px;
        background: var(--success);
    }

    /* ── Metric tiles ────────────────────────────────────────────────────── */
    .metric-tile {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        min-width: 0;
        overflow: hidden;
    }
    .metric-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text-primary);
        white-space: nowrap;
    }
    .metric-label {
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
        white-space: normal;
        line-height: 1.3;
    }

    /* ── Inputs ──────────────────────────────────────────────────────────── */
    .stTextArea textarea {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.88rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button {
        background: var(--accent) !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 11px 24px !important;
        width: 100% !important;
        letter-spacing: 0.01em !important;
        transition: background 0.12s !important;
    }
    .stButton > button:hover {
        background: var(--accent-hover) !important;
    }
    .stButton > button:focus:not(:active) {
        box-shadow: 0 0 0 2px rgba(79, 124, 255, 0.4) !important;
    }

    /* Secondary example buttons — visually distinct from primary CTA */
    div[data-testid="column"] .stButton > button {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: #1c2230 !important;
        border-color: var(--accent) !important;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    .tag-chip {
        display: inline-block;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
    }

    /* ── Tabs ────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
    }

    /* ── Misc chrome ─────────────────────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'won', 'prize', 'cash', 'claim', 'urgent',
    'click', 'offer', 'guaranteed', 'congratulations', 'selected', 'reward',
    'call now', 'limited', 'discount', 'deal', 'exclusive', 'account',
    'verify', 'suspended', 'alert', 'immediately',
]

def extract_features(text: str) -> dict:
    words   = text.split()
    caps    = sum(1 for w in words if w.isupper() and len(w) > 1)
    links   = len(re.findall(r'http[s]?://\S+|www\.\S+', text))
    nums    = len(re.findall(r'\b\d+\b', text))
    special = len(re.findall(r'[!$£€%@#*]', text))
    kws     = [kw for kw in SPAM_KEYWORDS if kw.lower() in text.lower()]
    return {
        'length'  : len(text),
        'words'   : len(words),
        'caps'    : caps,
        'links'   : links,
        'numbers' : nums,
        'special' : special,
        'keywords': kws,
    }


@st.cache_resource
def load_models():
    """Load pickled models. Returns None values if files not found."""
    files = {
        'tfidf'  : 'tfidf_vectorizer.pkl',
        'scaler' : 'scaler.pkl',
        'model'  : None,
        'le'     : 'label_encoder.pkl',
    }
    model_candidates = [f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pkl')]
    if model_candidates:
        files['model'] = model_candidates[0]

    loaded = {}
    for key, fname in files.items():
        if fname and os.path.exists(fname):
            with open(fname, 'rb') as f:
                loaded[key] = pickle.load(f)
        else:
            loaded[key] = None
    return loaded


def predict(text: str, assets: dict):
    if not all(assets.get(k) for k in ['tfidf', 'model', 'le']):
        return None, None

    tfidf  = assets['tfidf']
    scaler = assets['scaler']
    model  = assets['model']
    le     = assets['le']

    X = tfidf.transform([text])
    if scaler:
        X = scaler.transform(X)

    pred  = model.predict(X)[0]
    label = le.inverse_transform([pred])[0]

    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)[0]
    elif hasattr(model, 'decision_function'):
        raw = model.decision_function(X)[0]
        prob_spam = 1 / (1 + np.exp(-raw))
        prob = [1 - prob_spam, prob_spam]

    return label, prob


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Spam Detector")
    st.markdown("<div class='hero-sub'>ML-Powered Email &amp; SMS Shield</div>", unsafe_allow_html=True)
    st.divider()

    assets = load_models()
    model_loaded = all(assets.get(k) for k in ['tfidf', 'model', 'le'])

    if model_loaded:
        model_files = [f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pkl')]
        model_name  = model_files[0].replace('best_model_', '').replace('.pkl', '').replace('_', ' ').title() if model_files else 'Unknown'
        st.success(f"Model loaded: {model_name}")
    else:
        st.warning("Model files not found. Run the training notebook first to generate and save the model.")
        st.markdown("""
        **Required files in the same folder as `app.py`:**
        - `tfidf_vectorizer.pkl`
        - `scaler.pkl`
        - `label_encoder.pkl`
        - `best_model_*.pkl`
        """)

    st.divider()
    st.markdown("<div class='sidebar-section-label'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Predict", "Dataset Stats", "About"], label_visibility="collapsed")


# ── Page: Predict ─────────────────────────────────────────────────────────────
if page == "Predict":

    col_h, _ = st.columns([3, 1])
    with col_h:
        st.markdown("<div class='hero-title'>Is It Spam?</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>Paste any message and get an instant classification</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Input Message</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:4px'>Quick Examples</div>", unsafe_allow_html=True)
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            if st.button("Spam Example"):
                available = [i for i in range(len(SPAM_EXAMPLES)) if i != st.session_state.spam_index]
                st.session_state.spam_index = random.choice(available)
                st.session_state.input_text = SPAM_EXAMPLES[st.session_state.spam_index]
        with ex_col2:
            if st.button("Ham Example"):
                available = [i for i in range(len(HAM_EXAMPLES)) if i != st.session_state.ham_index]
                st.session_state.ham_index = random.choice(available)
                st.session_state.input_text = HAM_EXAMPLES[st.session_state.ham_index]

        user_input = st.text_area(
            label="message",
            value=st.session_state.input_text,
            placeholder="Paste your email or SMS here...\n\ne.g. 'Congratulations! You've won a FREE iPhone. Click here NOW!'",
            height=200,
            label_visibility="collapsed",
            key="input_text",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Analyze Message")

    with col_right:
        st.markdown("<div class='section-header'>Result</div>", unsafe_allow_html=True)

        if run and st.session_state.input_text.strip():
            text_to_analyse = st.session_state.input_text
            feats = extract_features(text_to_analyse)
            label, prob = predict(text_to_analyse, assets)

            if label:
                is_spam  = label.lower() == 'spam'
                conf     = (prob[1] if prob is not None else 0.95) if is_spam else (prob[0] if prob is not None else 0.95)
                conf_pct = f"{conf*100:.1f}%"

                card_cls = "spam-card" if is_spam else "ham-card"
                val_cls  = "spam-value" if is_spam else "ham-value"
                bar_cls  = "conf-bar-fill-spam" if is_spam else "conf-bar-fill-ham"
                verdict  = "SPAM" if is_spam else "HAM"

                st.markdown(f"""
                <div class='result-card {card_cls}'>
                    <div class='result-label'>Verdict</div>
                    <div class='result-value {val_cls}'>{verdict}</div>
                    <div class='result-label' style='margin-top:14px'>Confidence</div>
                    <div style='font-family:"IBM Plex Mono",monospace; font-size:1.3rem; color:var(--text-primary)'>{conf_pct}</div>
                    <div class='conf-bar-wrap'>
                        <div class='{bar_cls}' style='width:{conf*100:.0f}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='section-header' style='margin-top:20px'>Message Signals</div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-tile'><div class='metric-num'>{feats['words']}</div><div class='metric-label'>Words</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-tile'><div class='metric-num'>{feats['caps']}</div><div class='metric-label'>All-Caps</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-tile'><div class='metric-num'>{feats['special']}</div><div class='metric-label'>Special Chars</div></div>", unsafe_allow_html=True)

                if feats['keywords']:
                    st.markdown("<div class='section-header' style='margin-top:16px'>Spam Keywords Detected</div>", unsafe_allow_html=True)
                    chips = "".join([f"<span class='tag-chip'>{kw}</span>" for kw in feats['keywords']])
                    st.markdown(chips, unsafe_allow_html=True)

            else:
                st.info("Model not loaded. Please check the sidebar and ensure all .pkl files are present.")

        elif run and not st.session_state.input_text.strip():
            st.warning("Please enter a message first.")
        else:
            st.markdown("""
            <div style='color:var(--text-muted); font-family:"IBM Plex Mono",monospace; font-size:0.82rem; padding:40px 0; text-align:center'>
                Enter a message or click an example, then click Analyze
            </div>
            """, unsafe_allow_html=True)


# ── Page: Dataset Stats ───────────────────────────────────────────────────────
elif page == "Dataset Stats":
    st.markdown("<div class='hero-title'>Dataset Stats</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>SMS Spam Collection — Exploratory Overview</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    csv_candidates = [f for f in os.listdir('.') if f.endswith('.csv')]
    df = None
    if csv_candidates:
        try:
            raw = pd.read_csv(csv_candidates[0], encoding='latin-1')
            if 'v1' in raw.columns and 'v2' in raw.columns:
                df = raw[['v1', 'v2']].copy()
                df.columns = ['label', 'message']
            elif 'label' in raw.columns and 'message' in raw.columns:
                df = raw[['label', 'message']].copy()
            elif 'Category' in raw.columns and 'Message' in raw.columns:
                df = raw[['Category', 'Message']].copy()
                df.columns = ['label', 'message']
            else:
                df = raw.iloc[:, :2].copy()
                df.columns = ['label', 'message']

            df = df.drop_duplicates(subset='message').reset_index(drop=True)
            df['msg_length'] = df['message'].apply(len)
            df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    if df is not None:
        total        = len(df)
        n_spam       = (df['label'] == 'spam').sum()
        n_ham        = (df['label'] == 'ham').sum()
        avg_len_spam = df[df['label'] == 'spam']['msg_length'].mean()
        avg_len_ham  = df[df['label'] == 'ham']['msg_length'].mean()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(f"<div class='metric-tile'><div class='metric-num'>{total:,}</div><div class='metric-label'>Total Messages</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='metric-tile'><div class='metric-num'>{n_spam:,}</div><div class='metric-label'>Spam</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='metric-tile'><div class='metric-num'>{n_ham:,}</div><div class='metric-label'>Ham</div></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='metric-tile'><div class='metric-num'>{avg_len_spam:.0f}</div><div class='metric-label'>Avg Spam Length</div></div>", unsafe_allow_html=True)
        k5.markdown(f"<div class='metric-tile'><div class='metric-num'>{avg_len_ham:.0f}</div><div class='metric-label'>Avg Ham Length</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        plt.style.use('dark_background')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-header'>Class Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#161b26')
            ax.set_facecolor('#161b26')
            counts = df['label'].value_counts()
            bars   = ax.bar(counts.index, counts.values,
                            color=['#2fbf8f', '#ef5a5a'], width=0.5, edgecolor='none')
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25,
                        str(val), ha='center', color='#eef1f6', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', color='#a7b0c0')
            ax.tick_params(colors='#a7b0c0')
            ax.spines[:].set_visible(False)
            ax.set_yticks([])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("<div class='section-header'>Message Length Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#161b26')
            ax.set_facecolor('#161b26')
            ax.hist(df[df['label'] == 'ham']['msg_length'],  bins=50, alpha=0.85, color='#2fbf8f', label='Ham',  edgecolor='none')
            ax.hist(df[df['label'] == 'spam']['msg_length'], bins=50, alpha=0.85, color='#ef5a5a', label='Spam', edgecolor='none')
            ax.set_xlabel('Character Count', color='#a7b0c0')
            ax.tick_params(colors='#a7b0c0')
            ax.spines[:].set_visible(False)
            ax.legend(facecolor='#232a38', labelcolor='#eef1f6')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("<div class='section-header' style='margin-top:20px'>Sample Messages</div>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Spam Samples", "Ham Samples"])
        with tab1:
            spam_df = df[df['label'] == 'spam'][['message', 'msg_length']]
            st.dataframe(spam_df.sample(min(5, len(spam_df)), random_state=7), use_container_width=True, hide_index=True)
        with tab2:
            ham_df = df[df['label'] == 'ham'][['message', 'msg_length']]
            st.dataframe(ham_df.sample(min(5, len(ham_df)), random_state=7), use_container_width=True, hide_index=True)
    else:
        st.info("Place your `spam.csv` in the same folder as `app.py` to see dataset stats.")


# ── Page: About ───────────────────────────────────────────────────────────────
elif page == "About":
    st.markdown("<div class='hero-title'>About</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Spam Detector

        This app uses a machine learning model trained on the **SMS Spam Collection dataset**
        (approximately 5,500 real SMS messages) to classify text as spam or legitimate (ham).

        **Pipeline:**
        - TF-IDF Vectorization (unigrams + bigrams, top 5,000 features)
        - StandardScaler (sparse-safe, no data leakage)
        - Best model selected from 8 classifiers via GridSearchCV

        **Models evaluated:**
        Logistic Regression, Naive Bayes, Decision Tree, Random Forest,
        Gradient Boosting, AdaBoost, Linear SVC, KNN
        """)
    with col2:
        st.markdown("""
        ### Tech Stack

        | Layer | Tool |
        |-------|------|
        | Language | Python 3.10+ |
        | ML | scikit-learn |
        | App | Streamlit |
        | IDE | PyCharm |
        | Serialisation | Pickle |

        ### Required Files
        ```
        app.py
        spam.csv
        tfidf_vectorizer.pkl
        scaler.pkl
        label_encoder.pkl
        best_model_*.pkl
        requirements.txt
        ```
        """)
