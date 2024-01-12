import streamlit as st
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Visualization", page_icon="📈")

st.write("# Визуализация данных")


regression_data = pd.read_csv("data/regression_preprocessed_data")

# 1
st.write("# Датасет регрессии")
st.write("Этот график показывает взаимосвязь целевого признака - price - и наиболее коррелирующего - area.")

plot = sns.lmplot(x="area", y="price", data=regression_data)
st.pyplot(plot)

# 2
correlation_matrix = regression_data[['area', 'price', 'bedrooms', 'bathrooms', 'balcony']].corr()

st.write("Этот график показывает корреляцию целевого признака - price - и других признаков.")
plt.figure(figsize=(13,8))
sns.heatmap(correlation_matrix, annot=True, cmap= 'coolwarm')
plt.title("Корреляция между признаками")
st.pyplot(plt)


# 3
st.write("# Датасет классификации")

class_data = pd.read_csv("data/classification_preprocessed_data")
st.write("Выбраны наиболее коррелирующие и негативно коррелирующий признаки")

plt.figure(figsize=(13,8))

sns.pairplot(class_data[['Pressure', 'Humidity', 'Raw Ethanol','Fire Alarm']], hue='Fire Alarm')


st.pyplot(plt)


# 4
st.write("Выведем рапределение целевого признака с помощью гистрограмм")

class_data.hist('Fire Alarm')
st.pyplot(plt)
st.write("Наблюдаем значительный перевес в значении '1', то есть в случаях срабатывания сигнализации")

# с визуализациями зависимостей в наборе данных
# (визуализации Matplotlib, Seaborn, минимум 4 различных вида визуализаций);
