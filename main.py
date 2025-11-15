import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# ============================================================
# ⚡ 1. Initialize HuggingFace Client (FIXED ENDPOINT)
# ============================================================
@st.cache_resource
def get_hf_client():
    return InferenceClient(
        api_key=st.secrets["HF_TOKEN"],
        base_url="https://router.huggingface.co/hf-inference/v1"
    )

# ============================================================
# ⚡ 2. Load Prompts CSV
# ============================================================
@st.cache_data
def load_prompts():
    df = pd.read_csv('prompts.csv')
    return df

# ============================================================
# ⚡ 3. TF-IDF Precomputation
# ============================================================
@st.cache_resource
def compute_tfidf_vectors(_df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    vectors = vectorizer.fit_transform(_df['act'].fillna(''))
    return vectorizer, vectors

# ============================================================
# ⚡ 4. Best Match Finder
# ============================================================
def find_best_match(user_query, vectorizer, tfidf_matrix, df):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    return df.iloc[best_idx]

# ============================================================
# ⚡ 5. HuggingFace Chat Completion
# ============================================================
def generate_prompt(client, messages):
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# ⚡ 6. Streamlit UI
# ============================================================
st.title("🎯 Meta Prompt Generator")
st.caption("Generate expert prompts using AI-powered prompt engineering")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated_prompt" not in st.session_state:
    st.session_state.generated_prompt = None
if "is_first_message" not in st.session_state:
    st.session_state.is_first_message = True

# Load data
try:
    df = load_prompts()
    vectorizer, tfidf_matrix = compute_tfidf_vectors(df)
    client = get_hf_client()
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# ⚡ Chat Input
# ============================================================
if user_input := st.chat_input("Describe the prompt you need..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # --------------------------------------------------------
    # First Message Case
    # --------------------------------------------------------
    if st.session_state.is_first_message:
        matched = find_best_match(user_input, vectorizer, tfidf_matrix, df)

        reference_context = (
            f"Act: {matched['act']}\n"
            f"Reference Prompt: {matched['prompt']}"
        )

        system_prompt = """
You are an expert prompt engineer. Your job is to create prompts that users can give to AI assistants like ChatGPT or Claude.

Based on the user's need and the reference prompt provided, generate a complete, ready-to-use prompt that the user can copy and paste into an AI chat.

The prompt should:
- Start with "I want you to act as..."
- Be detailed and specific
- Include clear instructions
- Be immediately usable

Output ONLY the prompt itself - no explanations, no meta-commentary, no markdown formatting.
"""

        api_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"User Request: {user_input}\n\n"
                    f"{reference_context}\n\n"
                    "Generate the prompt:"
                )
            }
        ]

        st.session_state.is_first_message = False

    # --------------------------------------------------------
    # Refinement Messages
    # --------------------------------------------------------
    else:
        system_prompt = """
You are an expert prompt engineer specializing in dramatic prompt transformations.

The user wants you to SIGNIFICANTLY improve their prompt based on their feedback. Don't just add a word - TRANSFORM the entire prompt to match their request.

If they say "make it wonderful":
- Add vivid, powerful language
- Include creative examples
- Make it inspiring and engaging
- Expand the scope with exciting details

If they say "make it shorter":
- Cut everything non-essential

If they say "more technical":
- Use technical jargon
- Add methodologies
- Make it engineering-grade

Keep the "I want you to act as..." structure but REWRITE everything else.

Output ONLY the transformed prompt - no explanations, no markdown formatting.
"""

        api_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"=== PREVIOUS PROMPT ===\n"
                    f"{st.session_state.generated_prompt}\n\n"
                    f"=== REQUESTED CHANGES ===\n"
                    f"{user_input}\n\n"
                    f"=== OUTPUT THE IMPROVED PROMPT BELOW ==="
                )
            }
        ]

    # Generate output
    with st.chat_message("assistant"):
        with st.spinner("Generating prompt..."):
            generated = generate_prompt(client, api_messages)
            st.markdown(generated)
            st.code(generated, language=None)

    # Save state
    st.session_state.generated_prompt = generated
    st.session_state.messages.append({"role": "assistant", "content": generated})


# ============================================================
# ⚡ Sidebar
# ============================================================
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
1. **Describe** the prompt you need  
2. **Refine** it with small instructions  
3. **Copy** the final version  
    """)

    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        st.session_state.generated_prompt = None
        st.session_state.is_first_message = True
        st.rerun()

    st.divider()
    st.caption("Powered by Qwen3-Coder-30B-A3B-Instruct")
