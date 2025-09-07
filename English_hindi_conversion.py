!pip install transformers sentencepiece gradio --quiet
from transformers import MarianMTModel, MarianTokenizer

# Define source and target languages
src_lang = "en"
tgt_lang = "hi"

# Load model
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
def translate(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
import gradio as gr

def translate_gradio(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=translate_gradio,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text here..."),
        gr.Dropdown(choices=["en","hi","fr","de","es"], label="Source Language"),
        gr.Dropdown(choices=["hi","en","fr","de","es"], label="Target Language"),
    ],
    outputs="text",
    title="Text-to-Text Translation Demo"
)

iface.launch()
