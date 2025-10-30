import streamlit as st
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
@st.cache_resource
def load_model(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = TFT5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model
MODEL_PATH = "fine_tuned_t5_small"
tokenizer, model = load_model(MODEL_PATH)

st.set_page_config(page_title="New Summarizer", page_icon="#", layout="centered")

st.title(" New Summarization App")
st.write("Generate a summary using your fine-tuned **T5 TensorFlow model**.")

# User input
text_input = st.text_area(" Enter text to summarize:", height=200)

max_input_length = st.slider("Max input length", 64, 1024, 512, step=64)
max_output_length = st.slider("Max summary length", 20, 300, 150, step=10)

if st.button(" Generate Summary"):
    if text_input.strip():
        with st.spinner("Generating summary... please wait "):
            # Encode input
            input_text = "summarize: " + text_input
            input_ids = tokenizer.encode(
                input_text,
                return_tensors="tf",
                max_length=max_input_length,
                truncation=True,
                padding="max_length"
            )

            # Generate summary
            summary_ids = model.generate(
                input_ids,
                max_length=max_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            # Decode and display
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.success(" Generated Summary:")
            st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

# Footer
st.markdown("---")
st.caption("Built with  using Streamlit & Transformers (TensorFlow).")
