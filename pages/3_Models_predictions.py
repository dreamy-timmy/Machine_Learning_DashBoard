import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import joblib

st.set_page_config(page_title="Models predictions", page_icon="📈")


regression_data = pd.read_csv("data/regression_preprocessed_data")

X_reg = regression_data.drop(columns= 'price')
y_reg = regression_data['price']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size= .3)


st.title('Предсказания моделей регрессии')

first_model = joblib.load("models/MODEL_ONE_polynomial_best_regression_model.pkl")
fourth_model = joblib.load("models/MODEL_FOUR_bagging_regressor")
fiveth_model = joblib.load("models/MODEL_FIVE_stacking_regressor")
second_model = joblib.load("models/MODEL_TWO_gradient_boosting_regressor")


area = st.slider('Площадь дома', 503, 8000, 1500)

latitude = st.slider('Широта дома', min_value=18.873713, max_value=19.476239, step=0.001)

longitude = st.slider('Долгота дома', min_value=72.754080, max_value=73.197823, step=0.001)

bedrooms = st.number_input('Количество спален', min_value=2, step=1, max_value=10) # (2, 3, 4, 5, 6, 7, 8, 9, 10)
 
balcony = st.number_input('Количество балконов', min_value=0, step=1, max_value=8)

bathrooms = st.number_input('Количество ванн', min_value=0, step=1, max_value=10)

parking = st.radio('Количество парковочных мест', (0, 1, 2, 3, 4, 5, 6, 7, 8))

furnished_status = st.radio('Статус омеблированности дома', ("Нет мебели", "Частично", "Хорошо оснащено мебелью"))
if furnished_status == "Нет мебели": furnished_status = 0
elif furnished_status == "Частично": furnished_status = 0.5
else: furnished_status = 1

lift = st.number_input('Количество лифтов', min_value=0, step=1, max_value=8)

type_of_building = st.radio('Дом или квартира', ("Дом", "Квартира"))
if type_of_building == 'Дом': 
    type_of_building_House = 1
    type_of_building_Flat = 0
else:
    type_of_building_House = 0
    type_of_building_Flat = 1

status = st.radio('Статус дома', ("Готовый к заселению", "В процессе строения"))
if status == "Готовый к заселению":
    status_True = 1
    status_False = 0
else:
    status_True = 0
    status_False = 1

neworold = st.radio('Новый или перепродажа', ("Новый", "Перепродажа"))
if neworold == "Новый":
    neworold_True = 1
    neworold_False = 0
else:
    neworold_True = 0
    neworold_False = 1

# print(X_test)
test_data = pd.DataFrame(
    {
        'area': [area],
        'latitude': [latitude],
        'longitude': [longitude],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'balcony': [balcony],
        'parking': [parking],
        'furnished_status': [furnished_status],
        'lift': [lift],
        'type_of_building_Flat': [type_of_building_Flat],
        'type_of_building_Individual House': [type_of_building_House],
        'status_False': [status_False],
        'status_True': [status_True],
        'neworold_False': [neworold_False],
        'neworold_True': [neworold_True]
    }
)
if st.button('Предсказать стоимость дома'):
    first_prediction = int(first_model.predict(test_data)[0])
    st.text(f'Предсказаниe полиномиальной модели: {first_prediction}')
    four_prediction = int(fourth_model.predict(test_data)[0])
    st.text(f'Предсказаниe бэггинг модели: {four_prediction}')
    five_prediction = int(fiveth_model.predict(test_data)[0])
    st.text(f'Предсказаниe стэкинг модели: {five_prediction}')
    two_prediction = int(second_model.predict(test_data)[0])
    st.text(f'Предсказаниe модели градиентного бустинга: {two_prediction}')




# 2
classification_data = pd.read_csv("data/classification_preprocessed_data")

X_class = classification_data.drop(columns= 'Fire Alarm')
y_class = classification_data['Fire Alarm']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size= .3)




st.title('Предсказания моделей классификации')

sixth_model = tf.keras.models.load_model("models/MODEL_SIX_classification_NN")
third_model = joblib.load("models/MODEL_THREE_gradient_boosting_classifier")

Temperature = st.slider('Температура', -22.010000, 59.930000, 10.0)
Humidity = st.slider('Влажность', 10.740000, 75.200000, 10.0)
TVOC = st.slider('Индекс наличия Летучих Органических Соединений', 0.0, 60000.000000, 1000.0)
eCO2 = st.slider('Эквивалент углекислого газа', 400.000000, 60000.000000, 10000.0)
Raw_H2 = st.slider('Необработанный водород', 10668.000000, 13803.000000, 1500.0)
Raw_Ethanol = st.slider('Необработанный этанол', 15317.000000, 21410.000000, 1500.0)
Pressure = st.slider('Давление', 930.852000, 939.861000, 1500.0)
PM1_0 = st.slider('Ультрадисперсные частицы', 0.0, 14333.690000, 1500.0)
PM2_5 = st.slider('Мельчайшие частицы', 0.0, 45432.260000, 1500.0)
NC0_5 = st.slider('Коцентрация твёрдых частиц диаметром меньше 0.5 мкм', 0.0, 61482.030000, 3000.0)
NC1_0 = st.slider('Коцентрация твёрдых частиц диаметром меньше 1.0 мкм', 0.0, 51914.680000, 3000.0)
NC2_5 = st.slider('Коцентрация твёрдых частиц диаметром меньше 2.5 мкм', 0.0, 30026.438000, 1500.0)


test_data = pd.DataFrame(
    {
        'Temperature': [Temperature],
        'Humidity': [Humidity],
        'TVOC': [TVOC],
        'eCO2': [eCO2],
        'Raw H2': [Raw_H2],
        'Raw Ethanol': [Raw_Ethanol],
        'Pressure': [Pressure],
        'PM1.0': [PM1_0],
        'PM2.5': [PM2_5],
        'NC0.5': [NC0_5],
        'NC1.0': [NC1_0],
        'NC2.5': [NC2_5]
    }
)
if st.button('Предсказать результат пожарного датчика'):
    three_prediction = int(third_model.predict(test_data)[0])
    st.text(f'Предсказаниe модели градиентного бустинга: {three_prediction}')
    six_prediction = int(sixth_model.predict(test_data)[0])
    st.text(f'Предсказаниe модели нейросети: {six_prediction}')
    # five_prediction = int(fiveth_model.predict(test_data)[0])
    # st.text(f'Предсказаниe стэкинг модели: {five_prediction}')


# с помощью которой можно получить предсказание
# соответствующей модели ML (см. п. 1): реализовать загрузку файла в формате
# *.csv, сделать ввод соответствующих данных с использованием интерактивных
# компонентов и валидации.
