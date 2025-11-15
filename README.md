# 🎯 Meta Prompt Generator

An AI-powered prompt engineering tool that generates high-quality prompts for AI assistants like ChatGPT and Claude. Built with Streamlit and powered by Qwen 2.5-72B-Instruct.

## 🚀 Features

- **Smart Matching**: Uses TF-IDF vectorization and cosine similarity to find the best reference prompt from 160+ curated examples
- **AI-Powered Generation**: Leverages Qwen 2.5-72B-Instruct to create tailored, production-ready prompts
- **Iterative Refinement**: Refine generated prompts with natural language feedback
- **Copy-Ready Output**: Clean, formatted prompts ready to paste into any AI chat interface
- **Chat Interface**: Intuitive Streamlit chat UI for seamless interaction

## 📋 How It Works

### Architecture

```
User Input → TF-IDF Matching → Reference Prompt → Qwen API → Generated Prompt
                ↓
         (First Message)
                
User Feedback → Previous Prompt → Qwen API → Refined Prompt
                ↓
         (Subsequent Messages)
```

### Context Flow

**First Message:**
- User describes their need (e.g., "I need a code review helper")
- System matches against `act` column in `prompts.csv`
- Extracts matched `act + prompt` as reference context
- Sends to Qwen with user request + reference
- Generates complete prompt starting with "I want you to act as..."

**Refinement Messages:**
- User requests changes (e.g., "make it more technical", "add security focus")
- System uses previously generated prompt as context
- Sends to Qwen with old prompt + user feedback
- Generates improved version

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- HuggingFace API token ([Get one here](https://huggingface.co/settings/tokens))

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd meta-prompt-generator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Streamlit secrets**

Create a `.streamlit/secrets.toml` file in your project directory:
```bash
mkdir .streamlit
nano .streamlit/secrets.toml
```

Add your HuggingFace token:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

**Note:** Never commit this file to git. Add `.streamlit/` to your `.gitignore`.

4. **Add prompts data**

Place your `prompts.csv` file in the project root. The CSV should have this structure:
```csv
act,prompt
"An Ethereum Developer","Imagine you are an experienced Ethereum developer..."
"SEO Prompt","Using WebPilot, create an outline..."
```

5. **Run the app**
```bash
streamlit run app.py
```

## 📦 Project Structure

```
meta-prompt-generator/
├── app.py              # Main Streamlit application
├── prompts.csv         # Reference prompts database (160+ prompts)
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .env               # Environment variables (create this)
```

## 📝 Usage Examples

### Example 1: Creating an Advertiser Prompt

**User Input:**
```
I want an advertiser to promote my AI agency, Caspell Agency
```

**Generated Prompt:**
```
I want you to act as an advertiser for Caspell Agency, an AI solution provider. 
Your task is to create a comprehensive advertising campaign to promote our brand. 
Identify the target audience, develop key messages and slogans, select the media 
channels for promotion, and decide on any additional activities needed to reach 
our goals...
```

**Refinement:**
```
make it wonderful
```

**Refined Prompt:**
```
I want you to act as an advertiser for Caspell Agency, an AI solution provider. 
Your task is to create a breathtaking advertising campaign that will captivate 
and transform the world of businesses. Envision a campaign that showcases the 
unparalleled efficiency of our AI solutions and ignites a sense of wonder...
```

### Example 2: Code Review Helper

**User Input:**
```
I need a code review assistant for Python projects
```

**Refinement:**
```
add security focus and make it more technical
```

## 🎨 Features in Detail

### Smart Matching
- Uses scikit-learn's TF-IDF vectorizer
- Matches on the `act` column (prompt titles)
- Returns the full prompt as reference context
- Cached for performance

### Prompt Generation
- Model: `Qwen/Qwen2.5-72B-Instruct`
- Temperature: 0.7 (creative but controlled)
- Max tokens: 1500 (comprehensive prompts)
- System prompts optimized for prompt engineering

### Refinement Types
- **Style**: "make it more professional", "make it casual"
- **Scope**: "make it shorter", "add more details"
- **Focus**: "add security aspects", "focus on beginners"
- **Tone**: "make it wonderful", "more technical"

## 🔧 Configuration

### Model Settings

Edit in `app.py`:
```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",  # Model name
    messages=messages,
    max_tokens=1500,                     # Max output length
    temperature=0.7                      # Creativity (0.0-1.0)
)
```

### TF-IDF Settings

Edit in `app.py`:
```python
vectorizer = TfidfVectorizer(
    stop_words='english',      # Remove common words
    max_features=500           # Feature dimension
)
```

## 📊 Requirements

```txt
streamlit
pandas
scikit-learn
huggingface-hub
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more reference prompts to `prompts.csv`
- [ ] Support for multiple languages
- [ ] Export prompts to file
- [ ] Prompt history/favorites
- [ ] A/B testing different refinements
- [ ] Advanced filtering by category

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🙏 Acknowledgments

- Reference prompts sourced from [awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts)
- Powered by [Qwen 2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) via HuggingFace
- Built with [Streamlit](https://streamlit.io/)


## 🔮 Roadmap

- **v1.1**: Category filtering and prompt templates
- **v1.2**: Multi-turn conversation export
- **v1.3**: Custom reference prompt upload
- **v2.0**: Web-hosted version with user accounts

---

**Made with ❤️ for the prompt engineering community**