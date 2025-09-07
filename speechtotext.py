!pip install transformers sentencepiece torch gTTS SpeechRecognition --quiet

from transformers import MarianMTModel, MarianTokenizer
import torch
import gradio as gr
from gtts import gTTS
import tempfile
import speech_recognition as sr

# Load MarianMT model
src_lang = "en"
tgt_lang = "hi"
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def translate(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
    translated = model.generate(**inputs, max_length=100)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name

def speech_to_text(audio):
    r = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = r.record(source)
        try:
            return r.recognize_google(audio_data)
        except:
            return "Could not recognize speech"

def translate_speech(input_audio, src_lang, tgt_lang):
    if input_audio is not None:
        text = speech_to_text(input_audio)
    else:
        return "", None
    
    translated_text = translate(text)
    
    tts_lang = "hi" if tgt_lang=="hi" else "en"
    speech_file = text_to_speech(translated_text, lang=tts_lang)
    
    return translated_text, speech_file

# Gradio Interface (4.x compatible)
iface = gr.Interface(
    fn=translate_speech,
    inputs=[
        gr.Audio(label="Upload Audio File", type="filepath"),  # upload instead of microphone
        gr.Dropdown(choices=["en","hi"], label="Source Language"),
        gr.Dropdown(choices=["en","hi"], label="Target Language"),
    ],
    outputs=[
        gr.Textbox(label="Translated Text"),
        gr.Audio(label="Translated Speech", type="filepath")
    ],
    title="Speech-to-Speech Translation (English ↔ Hindi)"
)

iface.launch()

