import pandas as pd
import streamlit as st
import streamlit_folium as st_folium
import folium

data_adress = "./data_cleaned.csv"

@st.cache
def load():  
    data = pd.read_csv(data_adress)
    return data



st.title("Анализ апартаментов Бангкока")

data_load_state = st.text("⚙️загружаем данные...")
data = load()
data_load_state.text("✅Готово")
st.subheader("Данные о квартирах")
st.write(data)
st.subheader("Текущий курс местной валюты")
map = folium.Map(location=[13.75, 100.50], zoom_start=8)
for i in range(0,len(data)):
    folium.Marker(
        location=[data.iloc[[i]]["latitude"], data.iloc[[i]]['longitude']],popup=data.iloc[[i]]["name"].to_string(), icon=folium.Icon("green")
    ).add_to(map)
st.subheader("Квартиры на карте")
st_folium.st_folium(map, width=700)