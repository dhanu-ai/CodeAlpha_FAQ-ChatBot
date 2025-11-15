import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# Initialize HuggingFace client
@st.cache_resource
def get_hf_client():
    return InferenceClient(
        provider="hf-inference",
        api_key=st.secrets["HF_TOKEN"],
    )

# Load and preprocess prompts data
@st.cache_data
def load_prompts():
    df = pd.read_csv('prompts.csv')
    return df

# Precompute TF-IDF vectors
@st.cache_resource
def compute_tfidf_vectors(_df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    vectors = vectorizer.fit_transform(_df['act'].fillna(''))
    return vectorizer, vectors

# Find best matching prompt
def find_best_match(user_query, vectorizer, tfidf_matrix, df):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    return df.iloc[best_idx]

# Generate prompt using Qwen
def generate_prompt(client, messages):
    try:
        # Convert to the format expected by InferenceClient
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # For system messages, we'll prepend to the first user message or handle separately
                continue
            formatted_messages.append(msg)
        
        # Combine system message with the first user message if system exists
        system_content = None
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
                break
        
        if system_content:
            # Prepend system instruction to the first user message
            if formatted_messages and formatted_messages[0]["role"] == "user":
                formatted_messages[0]["content"] = f"{system_content}\n\n{formatted_messages[0]['content']}"
        
        # Generate using the chat completion method
        response = client.chat_completion(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",  # Updated to a more available model
            messages=formatted_messages,
            max_tokens=1500,
            temperature=0.7,
            stream=False
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        # Fallback response
        return f"I apologize, but I encountered an error. Please try again. Error details: {str(e)}"

# Alternative generation function using text generation
def generate_prompt_alternative(client, prompt_text):
    try:
        response = client.text_generation(
            prompt=prompt_text,
            max_new_tokens=1500,
            temperature=0.7,
            do_sample=True
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🎯 Meta Prompt Generator")
st.caption("Generate expert prompts using AI-powered prompt engineering")

# Initialize session state
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

# User input
if user_input := st.chat_input("Describe the prompt you need..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Prepare messages for API call
    if st.session_state.is_first_message:
        # First message: use matched reference prompt
        matched = find_best_match(user_input, vectorizer, tfidf_matrix, df)
        reference_context = f"Act: {matched['act']}\nReference Prompt: {matched['prompt']}"
        
        system_prompt = """You are an expert prompt engineer. Your job is to create prompts that users can give to AI assistants like ChatGPT or Claude.

Based on the user's need and the reference prompt provided, generate a complete, ready-to-use prompt that the user can copy and paste into an AI chat.

The prompt should:
- Start with "I want you to act as..."
- Be detailed and specific
- Include clear instructions
- Be immediately usable

Output ONLY the prompt itself - no explanations, no meta-commentary, no markdown formatting."""
        
        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Request: {user_input}\n\n{reference_context}\n\nGenerate the prompt:"}
        ]
        st.session_state.is_first_message = False
    else:
        # Subsequent messages: refine previous prompt
        system_prompt = """You are an expert prompt engineer specializing in dramatic prompt transformations.

The user wants you to SIGNIFICANTLY improve their prompt based on their feedback. Don't just add a word - TRANSFORM the entire prompt to match their request.

If they say "make it wonderful":
- Add vivid, powerful language
- Include creative examples
- Make it inspiring and engaging
- Expand the scope with exciting details

If they say "make it shorter":
- Cut everything non-essential
- Keep only the core requirements

If they say "more technical":
- Add technical specifications
- Include industry jargon
- Add specific methodologies

Keep the "I want you to act as..." structure but BOLDLY rewrite everything else to match their vision.

Output ONLY the transformed prompt - no explanations, no markdown formatting."""
        
        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"=== PREVIOUS PROMPT ===\n{st.session_state.generated_prompt}\n\n=== REQUESTED CHANGES ===\n{user_input}\n\n=== OUTPUT THE IMPROVED PROMPT BELOW ==="}
        ]
    
    # Generate prompt
    with st.chat_message("assistant"):
        with st.spinner("Generating prompt..."):
            generated = generate_prompt(client, api_messages)
            st.markdown(generated)
            
            # Add copy button
            st.code(generated, language=None)
    
    # Save to session state
    st.session_state.generated_prompt = generated
    st.session_state.messages.append({"role": "assistant", "content": generated})

# Sidebar with instructions
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    1. **First Message**: Describe what kind of prompt you need
       - Example: *"I need a code review helper"*
       - Example: *"I want an advertiser for my AI agency"*
    
    2. **Refinements**: Request changes to improve the prompt
       - Example: *"make it more technical"*
       - Example: *"add security focus"*
       - Example: *"make it shorter"*
       - Example: *"make it wonderful"*
    
    3. **Copy**: Use the code block to copy your prompt
    """)
    
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        st.session_state.generated_prompt = None
        st.session_state.is_first_message = True
        st.rerun()
    
    st.divider()
    st.caption("Powered by Qwen2.5-Coder-32B-Instruct")