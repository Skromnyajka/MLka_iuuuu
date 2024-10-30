#%%

# Выгружаю библиотеки необходимые мне для очистки данных и их анализа
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

#%%

# Выгружаю датафрейм для работы с ним
df = pd.read_csv('df.csv')

#%%

# Провожу первичный просмотр датафрейма
print(df.head(5))
df.shape

#%%

# Проверяю данные на дубликаты и удаляю их
print(df['url'].value_counts()[:5])
df.drop_duplicates(subset='url', inplace=True)
df['url'].value_counts()[:5]

#%%

# Провожу очистку данных и классификацию столбцов
# Обрабатываю некоторые столбцы для дальнейшей работы с ними
df['living_meters'] = df['living_meters'].str.replace(r'(\s.*)', '', regex=True).replace(',', '.', regex=True).astype(float)
df['kitchen_meters'] = df['kitchen_meters'].str.replace(r'(\s.*)', '', regex=True).replace(',', '.', regex=True).astype(float)



df.dropna(subset=['total_meters', 'year_of_construction'], inplace=True)

def replace_invalid_decimal(value):
    try:
        float(value)
        return value
    except:
        return str(value.split('.')[0]) + '.0'

df['total_meters'] = df['total_meters'].apply(lambda x: replace_invalid_decimal(x)).astype(float)

def del_str(value):
    try:
        if int(value) <= 2024:
          return value
        else:
          return np.nan
    except:
        return np.nan

df['year_of_construction'] = df['year_of_construction'].apply(lambda x: del_str(x))

#%%

# Заполняю пропуски в living_meters и kitchen_meters
df['living_meters'] = df['living_meters'].fillna(df['total_meters']-df['kitchen_meters'])
df['kitchen_meters'] = df['kitchen_meters'].fillna(df['total_meters']-df['living_meters'])

#%%

# Заменяю все значения -1 на пустые для более точной очистки и анализа, а значения -1 rooms_count на 0
print(df.info())

df['rooms_count'].replace(-1, 0, inplace=True)

df.replace(-1, np.nan, inplace=True)
df.replace('-1', np.nan, inplace=True)

#%%

# Вывожу информацию о расположении пустых значений в датафрейме и их колличестве
print(df.info())

sns.heatmap(df.isnull(), cmap='cividis')

#%%

# Вывожу кол-во пустых значений в калонках
df.isnull().sum()

#%%

# Вывожу процент пустых значений в калонках
round(df.isnull().sum() * 100 / len(df), 2).to_frame(name='percent_missing')

#%%

# Удаляю колонки: в которых большое кол-во пустых значений; в которых все значения одинаковые; которые не будут учавствовать в анализе
df.drop(['residential_complex', 'district', 'url', 'accommodation_type', 'phone', 'underground', 'house_number', 'deal_type', 'object_type', 'heating_type', 'house_material_type', 'finish_type', 'street'], axis = 1, inplace = True)
df.head()

#%%

# Удаляю строки, в которых есть пустые значения
df.dropna(how='all', inplace=True)
df.dropna(subset=['price', 'location', 'author', 'author_type', 'floors_count', 'rooms_count', 'living_meters', 'kitchen_meters', 'year_of_construction'], inplace=True)

#%%

# Проверяю результат очистки
print(df.isnull().any().any())
sns.heatmap(df.isnull(), cmap='cividis')
df.shape

#%%

# Классифицирую столбцы для дальнейшего анализа
df['year_of_construction'] = df['year_of_construction'].astype(int)
df['floor'] = df['floor'].astype(int)
df['floors_count'] = df['floors_count'].astype(int)
df['rooms_count'] = df['rooms_count'].astype(int)
df['price'] = df['price'].astype(int)

df.info()

#%%

# Очистка от выбросов с помощью Метода IQR и упорядочевую индексы
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1
df
df = df[(df['price'] < q3 + 1.5 * iqr) & (df['price'] > q1 - 1.5 * iqr)].reset_index()

df.drop(['index'], axis = 1, inplace = True)

df.shape

#%%

df.to_csv('new_clear_df.csv')

#%%

# Анализ данных
df.head(10)

#%%

df.describe(include='all')

# %%

# Популярность городов
location = df['location'].value_counts().nlargest(5)

plt.figure(figsize=(10, 10))

sns.set_palette("pastel")
plt.pie(location, 
        labels=location.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        wedgeprops={'edgecolor': 'black'})

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.legend(title="Города", loc="upper right", bbox_to_anchor=(1.2, 1))

plt.title("Популярность городов", fontsize=16, fontweight='bold')

plt.show()

#%%

# Популярность авторов объявлений
author = df['author'].value_counts().nlargest(5)

plt.figure(figsize=(10, 10))

sns.set_palette("pastel")
plt.pie(author, 
        labels=author.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        wedgeprops={'edgecolor': 'black'})

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.legend(title="Авторы объявлений", loc="upper right", bbox_to_anchor=(1.2, 1))

plt.title("Популярность авторов объявлений", fontsize=16, fontweight='bold')

plt.show()

#%%

# Популярность по году постройки здания
year_of_construction = df['year_of_construction'].value_counts().nlargest(5)

plt.figure(figsize=(10, 10))

sns.set_palette("pastel")
plt.pie(year_of_construction, 
        labels=year_of_construction.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        wedgeprops={'edgecolor': 'black'})

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.legend(title="Год", loc="upper right", bbox_to_anchor=(1.2, 1))

plt.title("Популярность по году постройки здания", fontsize=16, fontweight='bold')

plt.show()

#%%

# Популярность по типу авторов объявлений
author_type = df['author_type'].value_counts().nlargest(5)

plt.figure(figsize=(10, 10))

sns.set_palette("pastel")
plt.pie(author_type, 
        labels=author_type.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        wedgeprops={'edgecolor': 'black'})

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.legend(title="Типы авторов объявлений", loc="upper right", bbox_to_anchor=(1.2, 1))

plt.title("Популярность по типу авторов объявлений", fontsize=16, fontweight='bold')

plt.show()

#%%

# Тепловая карта отношения городов к авторам по средней цене
price_loc_author = df.loc[df['author'].isin(df['author'].value_counts().nlargest(10).keys().to_list())]

pivot_table = price_loc_author.pivot_table(values='price', index='location', columns='author', aggfunc='mean')

plt.figure(figsize=(15, 15))

sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlGnBu')

plt.title('Распределение средних цен по городам')
plt.xlabel('Автор')
plt.ylabel('Город')

plt.show()

#%%

# Средняя цена квартиры в зависимости от этажа
average_price_per_floor = df.groupby('floor')['price'].mean().reset_index()

plt.figure(figsize=(15, 15))

sns.barplot(x='floor', y='price', data=average_price_per_floor, palette='viridis')

plt.title('Средняя цена квартиры в зависимости от этажа')
plt.xlabel('Этаж')
plt.ylabel('Средняя цена квартиры')

plt.grid(True)
plt.show()

#%%

# Средняя цена квартиры в зависимости от общего кол-ва этажей в здании
average_price_per_floor = df.groupby('floors_count')['price'].mean().reset_index()

plt.figure(figsize=(15, 15))

sns.barplot(x='floors_count', y='price', data=average_price_per_floor, palette='viridis')

plt.title('Средняя цена квартиры в зависимости от общего кол-ва этажей в здании')
plt.xlabel('Общее кол-во этажей')
plt.ylabel('Средняя цена квартиры')

plt.grid(True)
plt.show()

#%%

# График распределения зависимости цены от общей площади квартиры
plt.figure(figsize=(15, 15))

sns.scatterplot(x='total_meters', y='price', data=df, marker='o')

plt.title('Зависимость цены от общей площади квартиры')
plt.xlabel('Общая площадь (м²)')
plt.ylabel('Цена')

plt.grid(True)
plt.show()

#%%

# Линейный график изменения средней цены квартир в зависимости от года постройки
average_price_per_year = df.groupby('year_of_construction')['price'].mean().reset_index()

plt.figure(figsize=(15, 8))

sns.lineplot(x='year_of_construction', y='price', data=average_price_per_year, marker='o')

plt.xticks(range(1600, 2025, 20))

plt.title('Изменение средней цены квартир в зависимости от года постройки')
plt.xlabel('Год постройки')
plt.ylabel('Средняя цена квартир')

plt.grid(True)
plt.show()

#%%

from sklearn import preprocessing

#Напишем функцию, которая принимает на вход наши данные, кодирует числовыми значениями категориальные признаки и возвращает обновленный данные и сами кодировщики
def number_encode_features(init_df):
    result = init_df.copy() #копируем нашу исходную таблицу
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == object: # np.object -- строковый тип / если тип столбца - строка, то нужно его закодировать
            encoders[column] = preprocessing.LabelEncoder() #для колонки column создаем кодировщик
            result[column] = encoders[column].fit_transform(result[column]) #применяем кодировщик к столбцу и перезаписываем столбец
    return result, encoders

encoded_data, encoders = number_encode_features(df) #Теперь encoded data содержит закодированные категориальные признаки
encoded_data.head() #проверяем

#%%

# Матрица корреляций
plt.subplots(figsize=(10,10))
encoded_data, encoders = number_encode_features(df)
sns.heatmap(encoded_data.corr(), square=True, annot=True)

#%%

encoded_data.hist(figsize=(20,15), color='red')

#%%

