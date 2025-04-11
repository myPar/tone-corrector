import streamlit as st
import base64
import fasttext
import re
import torch
from model_wrapper.model_wrapper import ModelWrapper

st.set_page_config(
    page_title="detoxi.ai",
    page_icon="./mini_logo1.png",
    layout="centered"
)

# Кодируем логотип в base64 (для локальных файлов)
@st.cache_data
def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

@st.cache_resource  # Кэширование модели для ускорения работы
def load_model():
    return ModelWrapper()

model_wrapper= load_model()

bin_str = get_image_base64("./билли.png")
page_bg_img = '''
<style>
.stApp{
background-image: linear-gradient(rgba(255, 255, 255, 0.7), 
                        rgba(255, 255, 255, 0.7)),
                        url("data:image/png;base64,%s");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}
</style>
''' % bin_str
st.markdown(page_bg_img, unsafe_allow_html=True)

logo_base64 = get_image_base64("./top_logo1.png")

# Используем HTML для вставки логотипа в заголовок
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{logo_base64}" width="400">
    </div>
    """,
    unsafe_allow_html=True
)

# Описание
st.write("""<p style='text-align: center; font-size: 24px;'>Это приложение сделает твою речь менее токсичной. 
И даже не придётся платить 300 bucks.</p>""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("""О приложении""")
    st.write("""
    Это приложение, созданно для сдачи задания по ML.
    Оно показывает, чему мы научились за эту домашку:
    - Благославлять создателей hugging face
    - Писать прототипы приложений с помощью библиотеки Streamlit
    - Дружно работать в команде
    """, unsafe_allow_html=True)

st.write("""<p style='text-align: center;'>Введите текст ниже, и приложение определит токсичность твоего предложения.</p>""", unsafe_allow_html=True)

user_input = st.text_area('',height=200)

model_type = st.radio(
    "Выберите модель",
    ("fasttext", "ru-BERT","FRIDA")
)

def highlight_obscene_words(text, model_type):
    if model_type=="FRIDA":
        result=model_wrapper(text, model_type)
        result=result.predictions
        for item in result:
            if item.label=="non-toxic":
                st.markdown(
                    "<span style='background:#47916B;'>{}:приемлемо</span>".format(item.text), 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<span style='background:#ffcccc;'>{}:токсично</span>".format(item.text), 
                    unsafe_allow_html=True
                )
    else:    
        label, prob=model_wrapper(text.lower(),model_type)
        if label=='__label__positive':
            st.markdown(
                "<span style='background:#47916B;'>{}:приемлемо</span>".format(text), 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<span style='background:#ffcccc;'>{}:токсично</span>".format(text), 
                unsafe_allow_html=True
            )
    
if st.button("Проверить текст"):
    if user_input.strip():
        st.subheader("Результат:")
        result = re.split(r'[.\n!?]+', user_input)
        result = [part for part in result if part.strip() != ""]
        if model_type=="FRIDA":
            highlight_obscene_words(result,model_type)
        else:
            if result!=[]:
                for text in result:
                    highlight_obscene_words(text,model_type)
    else:
        st.warning("Пожалуйста, введите текст для проверки")