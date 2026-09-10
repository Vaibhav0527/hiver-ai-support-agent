import streamlit as st
import json
import os
import yaml

# File paths
RAW_FILE = "data/golden/golden_set.jsonl"
LABELED_FILE = "data/golden/golden_set_labeled.jsonl"
INTENTS_CONFIG = "configs/intents.yaml"

@st.cache_data
def load_intents():
    with open(INTENTS_CONFIG, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return [i['name'] for i in config['intents']]

def load_data():
    if not os.path.exists(RAW_FILE):
        st.error(f"Raw dataset not found at {RAW_FILE}")
        return []
    
    # Load all items from raw
    raw_data = []
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
                
    # Check if labeled file exists to load previous work
    labeled_data = {}
    if os.path.exists(LABELED_FILE):
        with open(LABELED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    labeled_data[item['id']] = item
                    
    # Merge
    for item in raw_data:
        if item['id'] in labeled_data:
            # Overwrite with labeled data
            item.update(labeled_data[item['id']])
            
    return raw_data

def save_data(data):
    # We only save to LABELED_FILE
    with open(LABELED_FILE, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

st.set_page_config(page_title="Golden Set Labeler", layout="wide")

st.title("Golden Evaluation Set Labeling Interface")

# Initialize state
if 'data' not in st.session_state:
    st.session_state.data = load_data()
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

if not st.session_state.data:
    st.stop()

data = st.session_state.data
idx = st.session_state.current_idx
total = len(data)

intents = load_intents()

# Progress
labeled_count = sum(1 for item in data if item.get('source') == 'human_labeled')
st.progress(labeled_count / total if total > 0 else 0)
st.write(f"Progress: {labeled_count} / {total} Labeled (Viewing #{idx + 1})")

# Navigation buttons
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if st.button("⬅️ Previous", disabled=(idx == 0)):
        st.session_state.current_idx -= 1
        st.rerun()
with col2:
    if st.button("Next ➡️", disabled=(idx == total - 1)):
        st.session_state.current_idx += 1
        st.rerun()
with col4:
    if st.button("💾 Export/Save All"):
        save_data(data)
        st.success(f"Saved {len(data)} records to {LABELED_FILE}")

st.divider()

item = data[idx]

# Display data
st.subheader("Customer Message")
st.info(item.get('customer_message', ''))

st.subheader("Conversation Context")
context = item.get('conversation_context', [])
if not context:
    st.write("*No previous context (First message)*")
else:
    for c in context:
        st.write(f"**{c.get('role', 'unknown').title()}:** {c.get('text', '')}")

st.divider()

# Form for labeling
with st.form(key=f"label_form_{idx}"):
    st.subheader("Assign Labels")
    
    # Intent
    current_intent = item.get('intent', intents[0])
    # If the draft intent is invalid (e.g. general_complaint wasn't in list?), default to 0
    intent_idx = intents.index(current_intent) if current_intent in intents else 0
    selected_intent = st.selectbox("Intent", intents, index=intent_idx)
    
    # Escalation
    should_escalate = st.checkbox("Should Escalate?", value=item.get('should_escalate', False))
    escalation_reason = st.text_input("Escalation Reason (if yes)", value=item.get('escalation_reason', ''))
    
    # Characteristics
    # The JSON stores as list, we'll edit as comma separated string
    chars_list = item.get('expected_reply_characteristics', [])
    chars_str = ", ".join(chars_list)
    reply_chars = st.text_area("Expected Reply Characteristics (comma separated)", value=chars_str)
    
    submitted = st.form_submit_button("Save & Next")
    
    if submitted:
        # Update item
        item['intent'] = selected_intent
        item['should_escalate'] = should_escalate
        item['escalation_reason'] = escalation_reason
        item['expected_reply_characteristics'] = [x.strip() for x in reply_chars.split(',') if x.strip()]
        item['source'] = 'human_labeled'
        
        # Save changes to file immediately
        save_data(data)
        
        # Advance
        if idx < total - 1:
            st.session_state.current_idx += 1
            st.rerun()
        else:
            st.success("You have reached the end of the dataset!")
