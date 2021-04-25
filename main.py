# It is best to run this code using the Python Console in Pychram, line by
# line, so that you can see all the images. If you choose to run the
# entire program directly, you may not see many of the images.
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import webbrowser
from plotly.offline import init_notebook_mode
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

init_notebook_mode(connected=True)
# Put the data and code files in the same folder.
data = pd.read_csv('./accidents_2012_to_2014.csv', sep=',', encoding='utf-8')

pd.set_option('display.max_rows', None)  # Cancel row limit
pd.set_option('display.max_columns', None)  # Cancel column limit
pd.set_option('display.width', 1000)  # Increase line wide

data.shape
data.info()
data.isnull().sum()
data["date"] = pd.to_datetime(data.Date, format='%d/%m/%Y')
data['amount'] = pd.to_datetime(data.Date, format='%d/%m/%Y')
data.Day_of_Week.value_counts()
data.Year.value_counts()
data["time"] = pd.to_datetime(data.Time, format='%H:%M')
pd.DatetimeIndex(
    data["time"]).hour.value_counts().plot(
    kind="bar",
    title="Accident by DayHour")
pd.DatetimeIndex(
    data["date"]).month.value_counts().plot(
    kind="bar",
    title="Accident by Month")
pd.DatetimeIndex(
    data["date"]).weekday.value_counts().plot(
    kind="bar",
    title="Accident by Weekday")
data["Weather_Conditions"].value_counts().plot(
    kind="bar", title="The accident counts by Weather_Conditions", color="blue")
data["Light_Conditions"].value_counts().plot(
    kind="bar", title="The accident counts by Light_Conditions", color="blue")
data["Road_Type"].value_counts().plot(
    kind="bar", title="The accident counts by Road_Type", color="blue")

data_1 = data.set_index('date')
data_1.drop(['Date'], axis=1, inplace=True)
data_2 = data_1['amount'].groupby(
    [data_1.index.year, data_1.index.month, data_1.Accident_Severity]).count()
data_2 = data_2.reset_index(level=0, inplace=False)
data_2 = data_2.rename(index=str, columns={'date': "year"})
data_2 = data_2.reset_index(level=0, inplace=False)
data_2 = data_2.rename(index=str, columns={'date': "month"})
data_2 = data_2.reset_index(level=0, inplace=False)
data_3 = pd.pivot_table(
    data_2,
    index=['month'],
    columns=[
        'Accident_Severity',
        'year'],
    values=['amount'])
data_3
data_3['amount']['3'].plot(figsize=(10, 10), title='Slight')

ta_tmp = data.iloc[:30000, :]
ta_tmp['Number_of_Casualties'] = ta_tmp['Number_of_Casualties'].astype(
    np.float64)
max_amount = float(ta_tmp['Number_of_Casualties'].max())
hmap = folium.Map(location=[54, -3], zoom_start=5)
Layer = HeatMap(list(zip(ta_tmp.Latitude.values, ta_tmp.Longitude.values, ta_tmp.Number_of_Casualties.values)),
                min_opacity=0.2, radius=17, blur=15, max_zoom=1)
Layer.add_to(hmap)
hmap.save('hmap.html')
webbrowser.open('hmap.html')

cause_columns = [
    'Weather_Conditions',
    'Light_Conditions',
    'Road_Surface_Conditions',
    'Pedestrian_Crossing-Physical_Facilities',
    'Pedestrian_Crossing-Human_Control']
plt.figure(figsize=(10, 15))
counter = 0
for i in cause_columns:
    plt.subplot(5, 1, counter + 1)
    plt.xticks([])
    data[i].groupby(data[i]).count().plot(kind="bar")
    counter = counter + 1
plt.tight_layout()
plt.show()

predict_data = data_1['amount'].groupby(data_1.index.year).count()
X = predict_data.index.values
Y = predict_data.values
predict = np.sum(Y) / len(Y)
predict

df = data[['Date', 'Number_of_Casualties']]
df.to_csv('data_Casualties.csv')
dc = pd.read_csv('data_Casualties.csv')
dc = dc.iloc[:, 1:]
data_13 = dc[dc['Date'].astype('datetime64').dt.year == 2013]
temp = data_13.groupby('Date').agg({'Number_of_Casualties': 'sum'})
data_x = temp['Number_of_Casualties']
plt.plot(temp.index, data_x)
plot = temp.plot(kind='bar')
plot.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
fig = plot.get_figure()
fig.savefig("output.png")
plot_acf(temp)
plt.savefig("outputfirst.png")
print(u'ADF result is as follows:：', ADF(temp['Number_of_Casualties']))
plot_pacf(temp)
plt.savefig("outputsecond.png")
print(
    u'The white noise test result of the difference sequence is as follows:',
    acorr_ljungbox(
        temp,
        lags=1))
temp = temp.astype(float)
model1 = ARIMA(temp, (1, 1, 1)).fit()
model2 = ARIMA(temp, (1, 0, 1)).fit()
model1.summary2()
model2.summary2()
print(model1.forecast(5))
print(model2.forecast(5))
data_14 = dc[dc['Date'].astype('datetime64').dt.year == 2014]
real = data_14.groupby('Date').agg({'Number_of_Casualties': 'sum'})
