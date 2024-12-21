
import streamlit as st
import torch
from transformers import BarkModel, AutoProcessor


@st.cache_resource
def load_model():
    # Загрузка модели
    model = BarkModel.from_pretrained("suno/bark-small")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    processor = AutoProcessor.from_pretrained("suno/bark")
    return model, processor, device


def generate_speech(model, processor, device, text_prompt: str, voice_preset: str = None):
    # Подготовка входных данных
    if voice_preset:
        inputs = processor(text_prompt, voice_preset=voice_preset)
    else:
        inputs = processor(text_prompt)

    # Перенос на девайс
    inputs = inputs.to(device)

    # Генерация
    speech_output = model.generate(**inputs)
    # Возвращаем только первый сэмпл (speech_output[0])
    audio_array = speech_output[0].cpu().numpy()
    return audio_array


def main():
    st.title("Bark TTS Streamlit App")

    model, processor, device = load_model()

    text_prompt = st.text_area("Введите текст для озвучивания", value="Привет! Я говорю через Bark.")
    voice_preset = st.selectbox(
        "Выберите голос (необязательно)",
        [
            None,  # по умолчанию
            "v2/en_speaker_3",
            "v2/en_speaker_6",
            "fr_speaker_3"
        ],
        index=0
    )

    if st.button("Сгенерировать речь"):
        with st.spinner("Генерируем аудио..."):
            audio_array = generate_speech(model, processor, device, text_prompt, voice_preset=voice_preset)
            # Выводим аудио
            st.audio(audio_array, sample_rate=model.generation_config.sample_rate)


if __name__ == "__main__":
    main()
