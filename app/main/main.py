import streamlit as st
import torch
from transformers.models.bark import BarkModel, BarkProcessor
import numpy as np
from langdetect import detect, DetectorFactory
import scipy.io.wavfile
import os
import base64

# Для обеспечения детерминированности langdetect
DetectorFactory.seed = 0

@st.cache_resource
def load_model(model_name="suno/bark-small"):
    """
    Загрузка модели Bark.
    Используем bark-small, чтобы сэкономить место и память.
    """
    model = BarkModel.from_pretrained(model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    processor = BarkProcessor.from_pretrained("suno/bark")
    return model, processor, device

def generate_speech(
        model,
        processor,
        device,
        text_prompt: str,
        semantic_temp: float = 0.7,
        temperature: float = 0.7
):
    """
    Генерация речи на основе текста. Язык определяется автоматически.
    """
    # Автоматическое определение языка
    try:
        detected_lang = detect(text_prompt)
    except:
        detected_lang = "unknown"

    if detected_lang == "ru":
        language = "Русский"
        preset = None  # нет официального ru_speaker
        # Добавляем тег для русского
        if not text_prompt.strip().startswith("<|ru-RU|>"):
            text_prompt = f"<|ru-RU|> {text_prompt}"
    elif detected_lang == "en":
        language = "English"
        preset = None  # всегда None, чтобы модель сама выбирала голос
        # Добавляем тег для английского
        if not text_prompt.strip().startswith("<|en-US|>"):
            text_prompt = f"<|en-US|> {text_prompt}"
    else:
        # По умолчанию, если язык не распознан или не поддерживается
        language = "English"
        preset = None
        if not text_prompt.strip().startswith("<|en-US|>"):
            text_prompt = f"<|en-US|> {text_prompt}"

    # Подготавливаем вход
    inputs = processor(
        text_prompt,
        voice_preset=preset,  # всегда None
        return_tensors="pt",
        return_attention_mask=True
    )

    # Перенос на устройство
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Генерация
    with torch.no_grad():
        speech_output = model.generate(
            **inputs,
            semantic_temperature=semantic_temp,
            temperature=temperature
        )

    # Возвращаем только первый сэмпл
    audio_array = speech_output[0].cpu().numpy()

    # Сохранение аудио в WAV файл
    wav_filename = "generated_audio.wav"
    scipy.io.wavfile.write(wav_filename, rate=model.generation_config.sample_rate, data=audio_array)

    return wav_filename

def get_binary_file_downloader_html(bin_file, file_label='Скачать'):
    """
    Создание HTML-ссылки для скачивания бинарного файла.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:audio/wav;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    return href

def main():
    st.title("Bark TTS (Русский / English)")

    # Загрузка модели
    model, processor, device = load_model()

    # Поле ввода текста
    text_prompt = st.text_area(
        "Введите текст для озвучивания",
        value="Привет, я говорю по-русски!"
    )

    if st.button("Сгенерировать речь"):
        if not text_prompt.strip():
            st.error("Пожалуйста, введите текст для озвучивания.")
        else:
            with st.spinner("Генерируем аудио..."):
                wav_filename = generate_speech(
                    model,
                    processor,
                    device,
                    text_prompt
                )
                # Проверка успешности генерации
                if not wav_filename or not os.path.exists(wav_filename):
                    st.error("Ошибка при генерации аудио. Пожалуйста, попробуйте снова.")
                else:
                    # Выводим аудио
                    st.audio(wav_filename, format='audio/wav')
                    # Ссылка для скачивания
                    st.markdown(get_binary_file_downloader_html(wav_filename, 'Скачать аудио'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
