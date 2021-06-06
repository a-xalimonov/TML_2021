import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import f1_score
import country_converter as coco

@st.cache
def load_data():
    data = pd.read_csv('../worldcitiespop.csv', sep=",")
    data = data.drop("Population", axis=1).dropna()
    data = data.sample(n=20000, random_state=10)
    enc = LabelEncoder()
    data["Country_encoded"] = enc.fit_transform(data["Country"])
    return (data, enc)

st.header('Курсовой проект по дисциплине «Технологии машинного обучения»')

data_load_state = st.text('Загрузка данных...')
data, ecoder = load_data()
data_load_state.text('Данные готовы!')

# Head
st.subheader('Первые 5 значений')
st.write(data.head())
st.write('Количество строк:', data.shape[0])
st.write('Количество столбцов:', data.shape[1])
st.write('Типы данных:', data.dtypes)

st.subheader('Основные диаграммы')

# Цветовые обозначения
if st.checkbox('Добавить цветовые обозначения'):
    hue_ = "Country"
else:
    hue_ = None

# Парные диаграммы
fig1 = sns.pairplot(data, height=6, aspect=1.5, hue=hue_)
if (hue_):
    fig1.legend.remove()
st.pyplot(fig1)
# Карта
fig2, ax = plt.subplots(figsize=(25,15))
sns.scatterplot(ax=ax, x='Longitude', y='Latitude', hue=hue_, data=data, legend=False)
st.pyplot(fig2)

# Соотношение обучающей и тестовой выборки
st.subheader('Разбиение выборки на обучающую и тестовую')
test_size = st.slider('Соотношение:', min_value=0.05, max_value=0.95, value=0.3, step=0.01)

model_state = st.text('Обучение моделей и расчёт метрик...')
# Разделение выборки и обучение моделей
X_train, X_test, Y_train, Y_test = train_test_split(data[["Latitude", "Longitude"]], data["Country_encoded"], test_size=test_size, random_state=10)

models = {
    'Nearest Neighbors':KNeighborsClassifier(n_neighbors=5),
    'C-Support Vector': SVC(),
    'Desicion Tree  ':DecisionTreeClassifier(),
    'Random Forest  ':RandomForestClassifier(),
    'Bagging        ':BaggingClassifier(random_state=10)
    }


scores = {
    'values': [],
    'names': []
}

for model_name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores['values'].append(f1_score(Y_test, Y_pred, average="weighted"))
    scores['names'].append(model_name)


scores_data = pd.DataFrame({
    'index': scores['names'],
    'values': scores['values'],
}).set_index('index')

# Вывод графика с метриками
st.bar_chart(scores_data, height=550)
model_state.text('Готово!')

# Расчёт по координатам
st.subheader('Определение страны по координатам')
Longitude = st.number_input('Долгота')
Latitude = st.number_input('Широта')

for model_name, model in models.items():
    prediction = model.predict([[Longitude, Latitude]])
    st.write(model_name, '–', coco.convert(names=ecoder.inverse_transform(prediction), to="name_short"))