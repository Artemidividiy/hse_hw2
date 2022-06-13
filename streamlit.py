import pandas as pd
import streamlit as st
import streamlit_folium as st_folium
import folium
import io
import time
import requests
import base64

data_adress = "./data_cleaned.csv"

@st.cache
def load():  
    data = pd.read_csv(data_adress)
    return data

@st.cache
def get_currency(amount):
    api = f"https://api.apilayer.com/exchangerates_data/convert?to=RUB&from=THB&amount={amount}"
    try:
        response = requests.get(url=api, headers={"apikey" : "OGJmUfgUguQbQcSPUxPZJIGWnLweOV2j"})
        if(response.status_code == 429) : return "Достигнут лимит на запросы"
        return response.json()["result"]
    except: 
        return "что-то пошло не так"

@st.cache
def get_monser():
    api = "https://app.pixelencounter.com/api/basic/monsters/random"
    return requests.get(url=api).content.decode()

def render_svg():
    svg = get_monser()
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

st.title("Анализ апартаментов Бангкока", anchor="top")
with st.sidebar:
    st.title("Оглавление")
    st.header("[🔙В начало](#top)")
    st.subheader("[📊Данные](#1)")
    st.subheader("[💱Курс валюты](#2)")
    st.subheader("[🗺Карта](#3)")
    st.subheader("[👨‍🎓Обучаем](#4)")
    st.subheader("[👾Монстр](#5)")

data_load_state = st.text("⚙️загружаем данные...")
data = load()

data_load_state.text("✅Готово")

st.subheader("📊Данные о квартирах", anchor="1")
st.write(data)

st.header("💱Текущий курс местной валюты", anchor="2")
st.text("В Тайланде действующей валютой является Тайский бат")
amount = st.number_input("количество", value=1)
st.write(f"{amount} тайских бат будут стоить {get_currency(amount)}")

map = folium.Map(location=[13.75, 100.50], zoom_start=8)
for i in range(0,len(data)):
    folium.Marker(
        location=[data.iloc[[i]]["latitude"], data.iloc[[i]]['longitude']],popup=data.iloc[[i]]["name"].to_string(), icon=folium.Icon("green")
    ).add_to(map)
st.subheader("🗺Квартиры на карте", anchor="3")
st_folium.st_folium(map, width=700)

st.header("👨‍🎓Обучаем", anchor="4")
buf = io.StringIO()
data.info(buf=buf)
st.text(buf.getvalue())
graph_loading_state = st.text("⚙️учимся...")
st.text("RMSE test: 36418.04\n r2_score test: 0.5056552322775255")
st.subheader("Linear Regression")
time.sleep(1)
st.image("linear_regression.png")
st.subheader('alpha=0.0001')
time.sleep(1)
st.image("random_state.png")
st.text("RMSE test: 36418.04\n r2_score test: 0.5056554986886532\n RMSE test: 36417.95\n r2_score test: 0.5056578947375542")
st.subheader('alpha=0.001')
time.sleep(1)
st.image("random_state_alpha_0.001.png")
st.subheader('alpha=0.01')
time.sleep(1)
st.image("random_state_alpha_0.01.png")
st.text("RMSE test: 36417.07\n r2_score test: 0.5056816927610259\n RMSE test: 36408.87\n r2_score test: 0.505904375084538")
st.subheader('alpha=0.1')
time.sleep(1)
st.image("random_state_alpha_0.1.png")
st.subheader('alpha=1')
time.sleep(1)
st.image("random_state_alpha_1.png")
st.text("RMSE test: 36361.52\n r2_score test: 0.507188524838547\n RMSE test: 36350.14\n r2_score test: 0.5074969823710151")
st.subheader('alpha=10')
time.sleep(1)
st.image("random_state_alpha_10.png")
st.subheader('alpha=100')
time.sleep(1)
st.image("random_state_alpha_100.png")
st.text("RMSE test: 36662.94\n r2_score test: 0.4989843293557179\n RMSE test: 39708.62\n r2_score test: 0.4122856576396434")
st.subheader('alpha=1000')
time.sleep(1)
st.image("random_state_alpha_1000.png")
st.image("magnitudes_of_the_coefficients.png")
time.sleep(5)
st.text("RMSE test: 30450.87\n r2_score test: 0.6543822032680188\n RMSE test: 30450.87\n r2_score test: 0.6543822032680188")
st.text("random forest")
st.image("random_forest.png")
st.image("to_random_forst_pipeline.png")
time.sleep(5)
st.text("RMSE test: 30033.24\n r2_score test: 0.6637973026043756\n RMSE test: 30033.24\n r2_score test: 0.6637973026043756")
st.image("gradient_boosting_1.png")
st.image("gradient_boosting_2.png")
time.sleep(5)
st.text("RMSE test: 29937.48 \nr2_score test: 0.6659379297586393")
st.image("variable_importance.png")
graph_loading_state = st.text("✅Готово")

st.header("👾Получаем монстра", anchor=5)
if st.button("Запрос монстра"): 
   render_svg()
else: 
    st.write("Нажмите на кнопку")