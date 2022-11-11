#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import missingno as mnso
import plotly.express as px
import os
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install gapminder
from gapminder import gapminder 
import folium
import geopandas as gpd
import folium
import geopandas as gpd
import streamlit as st
#!pip install streamlit-folium
import streamlit_folium as st_folium
from streamlit_folium import folium_static


# In[2]:


#pip show plotly


# In[3]:


gapminder = gapminder[["country", "continent"]]
#gapminder.head()


# In[4]:


gapminder.rename(columns ={'country':'Country'},inplace = True)
#gapminder.head()


# In[5]:


df_continent = pd.read_csv("covid 19 CountryWise.csv")
#df_continent


# In[6]:


df_continent = df_continent[["Country", "Region"]]


# In[7]:


df_death = pd.read_csv("cause_of_deaths.csv")


# In[8]:


df_death.rename(columns ={'Country/Territory':'Country'},inplace = True)


# In[9]:


df_death = df_death.merge(df_continent, left_on = 'Country', right_on = 'Country')
#df_death = df_death.merge(gapminder, on='Country')


# In[10]:


#mnso.matrix(df_death)


# In[11]:


#df_death.columns


# In[12]:


#df_death.quantile(0.75)


# In[13]:


#df_death_clean = df_death[(df_death['Meningitis'] <= 847.25) & (df_death["Alzheimer's Disease and Other Dementias"] <= 2456.25) & 
                          #(df_death["Parkinson's Disease"] <= 609.25) & (df_death['Nutritional Deficiencies'] <= 1167.25) & 
                          #(df_death["Malaria"] <= 393.00) & (df_death["Drowning"] <= 698.00) & 
                          #(df_death['Interpersonal Violence'] <= 877.00) & 
                          #(df_death["Maternal Disorders"] <= 734.00) & 
                          #(df_death["HIV/AIDS"] <= 1879.00) & (df_death['Drug Use Disorders'] <= 129.00) & 
                          #(df_death["Tuberculosis"] <= 2924.25) & (df_death["Cardiovascular Diseases"] <= 42546.50) & 
                          #(df_death['Lower Respiratory Infections'] <= 10161.25) & (df_death["Neonatal Disorders"] <= 7419.75) & 
                          #(df_death["Alcohol Use Disorders"] <= 316.00) & (df_death['Self-harm'] <= 1882.25) & 
                          #(df_death["Exposure to Forces of Nature"] <= 12.00) & (df_death["Diarrheal Diseases"] <= 3946.75) &
                          #(df_death['Environmental Heat and Cold Exposure'] <= 109.00) & (df_death["Neoplasms"] <= 20147.75) & 
                          #(df_death["Conflict and Terrorism"] <= 23.00) & (df_death['Diabetes Mellitus'] <= 2954.00) & 
                          #(df_death["Chronic Kidney Disease"] <= 2922.50) & (df_death["Poisonings"] <= 254.00) & 
                          #(df_death["Protein-Energy Malnutrition"] <= 1042.50) & 
                          #(df_death["Road Injuries"] <= 3435.25) & (df_death['Chronic Respiratory Diseases'] <= 5249.75) & 
                          #(df_death["Cirrhosis and Other Chronic Liver Diseases"] <= 3547.25) & (df_death["Digestive Diseases"] <= 6080.00) & 
                          #(df_death["Fire, Heat, and Hot Substances"] <= 450.00) & (df_death['Acute Hepatitis'] <= 160.00)]
#df_death_clean


# In[14]:


#df_death_clean = df_death[(df_death['Meningitis'] <= 2330.60) & (df_death["Alzheimer's Disease and Other Dementias"] <= 5209.45) & 
                          #(df_death["Parkinson's Disease"] <= 1241.15) & (df_death['Nutritional Deficiencies'] <= 3050.05) & 
                          #(df_death["Malaria"] <= 4412.25) & (df_death["Drowning"] <= 1294.45) & 
                          #(df_death['Interpersonal Violence'] <= 1869.60) & 
                          #(df_death["Maternal Disorders"] <= 1564.15) & 
                          #(df_death["HIV/AIDS"] <= 5603.00) & (df_death['Drug Use Disorders'] <= 244.00) & 
                          #(df_death["Tuberculosis"] <= 7225.15) & (df_death["Cardiovascular Diseases"] <= 78957.55) & 
                          #(df_death['Lower Respiratory Infections'] <= 18697.15) & (df_death["Neonatal Disorders"] <= 15208.30) & 
                          #(df_death["Alcohol Use Disorders"] <= 550.15) & (df_death['Self-harm'] <= 3394.30) & 
                          #(df_death["Exposure to Forces of Nature"] <= 38.00) & (df_death["Diarrheal Diseases"] <= 9947.75) &
                          #(df_death['Environmental Heat and Cold Exposure'] <= 160.00) & (df_death["Neoplasms"] <= 38410.85) & 
                          #(df_death["Conflict and Terrorism"] <= 162.00) & (df_death['Diabetes Mellitus'] <= 6362.60) & 
                          #(df_death["Chronic Kidney Disease"] <= 5512.90) & (df_death["Poisonings"] <= 435.00) & 
                          #(df_death["Protein-Energy Malnutrition"] <= 2950.10) & 
                          #(df_death["Road Injuries"] <= 7264.90) & (df_death['Chronic Respiratory Diseases'] <= 12647.80) & 
                          #(df_death["Cirrhosis and Other Chronic Liver Diseases"] <= 7009.20) & (df_death["Digestive Diseases"] <= 12191.65) & 
                          #(df_death["Fire, Heat, and Hot Substances"] <= 687.15) & (df_death['Acute Hepatitis'] <= 333.00)]
#df_death_clean


# In[15]:


df_death_box = df_death[['Meningitis',
       "Alzheimer's Disease and Other Dementias", "Parkinson's Disease",
       'Nutritional Deficiencies', 'Malaria', 'Drowning',
       'Interpersonal Violence', 'Maternal Disorders', 'HIV/AIDS',
       'Drug Use Disorders', 'Tuberculosis', 'Cardiovascular Diseases',
       'Lower Respiratory Infections', 'Neonatal Disorders',
       'Alcohol Use Disorders', 'Self-harm', 'Exposure to Forces of Nature',
       'Diarrheal Diseases', 'Environmental Heat and Cold Exposure',
       'Neoplasms', 'Conflict and Terrorism', 'Diabetes Mellitus',
       'Chronic Kidney Disease', 'Poisonings', 'Protein-Energy Malnutrition',
       'Road Injuries', 'Chronic Respiratory Diseases',
       'Cirrhosis and Other Chronic Liver Diseases', 'Digestive Diseases',
       'Fire, Heat, and Hot Substances', 'Acute Hepatitis']]


# In[16]:


fig = go.Figure()

for col in df_death_box:
    fig.add_trace(go.Box(y=df_death_box[col].values, name=df_death_box[col].name))

    fig.update_layout(title_text="Boxplots all diseases")
#fig.show()


# In[17]:


df_drowning = df_death[["Cardiovascular Diseases", "Diabetes Mellitus", "Tuberculosis","Region", "Country", "Year" ]]
fig = px.box(df_drowning, x="Region", y="Cardiovascular Diseases", color="Region")
#df_drowning.head()


# In[18]:


df2 = df_drowning.groupby('Region')['Cardiovascular Diseases'].sum()
#df2.head()


# In[19]:


#df2.head(10)


# In[20]:


wp = df_drowning.loc[(df_drowning['Region'] == "Western Pacific")]
#wp.head()


# In[21]:


df = px.data.gapminder()
fig2 = px.scatter_geo(df_death, locations="Code", color="Region", hover_name="Country", size="Cardiovascular Diseases",
               animation_frame="Year", projection="natural earth")
#fig.show()


# In[22]:


#plt.figure(figsize=(10,10))
#plt.ticklabel_format(useOffset = True, style = 'plain')
#sns.barplot(data = df_death, x= 'Region', y='Meningitis',ci = None, palette = 'Set2' )
#plt.xticks(rotation = 45, size=12)
#plt.yticks(size = 12)
#plt.xlabel('Region',fontsize = 18)
#plt.ylabel('Total Cases', fontsize = 18)
#plt.title ('Total cases meningitis per region',fontsize = 18)


# In[23]:


df_death_hist = df_death[['Year', 'Country', 'Meningitis',
       "Alzheimer's Disease and Other Dementias", "Parkinson's Disease",
       'Nutritional Deficiencies', 'Malaria', 'Drowning',
       'Interpersonal Violence', 'Maternal Disorders', 'HIV/AIDS',
       'Drug Use Disorders', 'Tuberculosis', 'Cardiovascular Diseases',
       'Lower Respiratory Infections', 'Neonatal Disorders',
       'Alcohol Use Disorders', 'Self-harm', 'Exposure to Forces of Nature',
       'Diarrheal Diseases', 'Environmental Heat and Cold Exposure',
       'Neoplasms', 'Conflict and Terrorism', 'Diabetes Mellitus',
       'Chronic Kidney Disease', 'Poisonings', 'Protein-Energy Malnutrition',
       'Road Injuries', 'Chronic Respiratory Diseases',
       'Cirrhosis and Other Chronic Liver Diseases', 'Digestive Diseases',
       'Fire, Heat, and Hot Substances', 'Acute Hepatitis', 'Region']]


# In[52]:


df_death_hist = df_death_hist.groupby(["Year", 'Region'])['Cardiovascular Diseases'].mean().reset_index(name="Cardiovascular Diseases")

df_death_hist["Year"] = df_death_hist["Year"].astype(str)
#df_death_hist.head()


# In[53]:


wp_c = wp.groupby(["Year", "Country"])['Cardiovascular Diseases'].mean().reset_index(name="Cardiovascular Diseases")

wp_c["Year"] = wp_c["Year"].astype(str)
#wp_c


# In[54]:


fig = px.histogram(wp_c, x="Year", y="Cardiovascular Diseases", color="Country")
#fig.show()


# In[27]:


wp_2 = wp.loc[(wp['Country'] == "China")].sort_values("Year")
#wp_2.head()


# In[28]:


fig = px.line(wp_2, x="Diabetes Mellitus", y="Cardiovascular Diseases", symbol = "Country", hover_name="Year")
fig.update_xaxes(autorange = True)
#fig.show()


# In[29]:


fig = px.scatter_matrix(wp_2,
    dimensions=["Cardiovascular Diseases", "Diabetes Mellitus", "Tuberculosis"], hover_name="Year")
#fig.show()


# In[30]:


dfc = df_death[["Cardiovascular Diseases", "Diabetes Mellitus", "Tuberculosis","Region", "Country", "Year", "Code"]]
wp = dfc.loc[(dfc['Region'] == "Western Pacific")]
#wp.head()


# In[31]:


wpn = wp.drop(columns=["Country", "Code", 'Region',"Year"])
df1 = wpn.max(axis=1)
df2 = wpn.idxmax(axis=1)
wp["Top Count"] = df1
wp['Top Cause'] = df2
#wp.head()


# In[32]:


fig = px.choropleth(wp,               
              locations="Code",               
              color="Top Cause",
              hover_name="Country",
              hover_data=["Top Count"],
              animation_frame="Year",
              width=800      
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig.show()


# In[33]:


wp_2 = dfc.loc[(dfc['Country'] == "China")].sort_values("Year")
#wp_2.head()


# In[34]:


wp_3 = gpd.read_file('WP.json')
#wp.head()


# In[35]:


#lege folium map voor referentie
m = folium.Map(location=[12.5657, 104.9910], zoom_start=2, tiles = 'openstreetmap')

folium.Choropleth(
            geo_data=wp_3,
            data=wp,
            columns=['Country', 'Cardiovascular Diseases'],  
            key_on='feature.properties.admin', 
            fill_color='YlOrRd',
            nan_fill_color="White", 
            fill_opacity=1,
            line_opacity=0.2,
            legend_name='Cardiovascular Diseases) ', 
            highlight=True,
            line_color='black').add_to(m) 
#m


# In[36]:


df = pd.read_csv('cause_of_deaths.csv')
df_continent_2 = pd.read_csv("covid 19 CountryWise.csv")


# In[37]:


df_continent_2 = df_continent[["Country", "Region"]]


# In[38]:


df.rename(columns ={'Country/Territory':'Country'},inplace = True)
df2 = df.merge(df_continent, left_on = 'Country', right_on = 'Country')
df3 = df2[["Cardiovascular Diseases", "Diabetes Mellitus", "Tuberculosis","Region", "Country", "Year", "Code"]]
wp = df3.loc[(df3['Region'] == "Western Pacific")]


# In[39]:


wp_china = df3.loc[(df3['Country'] == "China")].sort_values("Year")
#wp_c['Year'] = pd.to_datetime(wp_c['Year'], format= '%Y')
#wp_c.head()


# In[40]:


fig = plt.figure()
sns.regplot(data = wp, y='Cardiovascular Diseases',x = 'Diabetes Mellitus', ci = None )
#plt.show()


# In[41]:


fig = plt.figure()
sns.regplot(data = wp, y='Cardiovascular Diseases',x = 'Tuberculosis', ci = None )
#plt.show()


# In[42]:


#for country, ax in zip(set(wp['Country'])):
#    countries = wp[wp['Country'] == country]
#    sns.regplot(x= countrys['Cardiovascular Diseases'],y = countries['Diabetes Mellitus'], color= 'red',ax = ax).set_title(country)
#plt.tight_layout()
#plt.show()


# In[43]:


modeldata = wp_china[['Year','Cardiovascular Diseases']]


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[45]:


target = wp_c['Cardiovascular Diseases']
features = wp_c[wp_c.columns.difference(['Cardiovascular Diseases','Year'])]

x_train, x_test, y_train, y_test = train_test_split(pd.get_dummies(features), target, test_size = 0.3)
lr = LinearRegression()
lr2 = lr.fit(x_train,y_train)
score = lr.score(x_test, y_test)
#print(score)


# In[46]:


#De coefficient voor bevestiging van wortel van R is dichtbij 1, model is fitted om een voorspelling te doen


# In[47]:


x = modeldata.iloc[:, :1].values 
y = modeldata.iloc[:, 1:2].values 


# In[48]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.3, random_state=0) 
regressor = LinearRegression() 
regressor.fit(x_train2, y_train2) 
y_pred = regressor.predict(x_test2)


# In[49]:


fig = plt.scatter(x_train2, y_train2 ,color='r') 
plt.plot(x_test2, y_pred, color='k') 


# In[50]:


#y_pred


# In[51]:


###### CODE STREAMLIT #######


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1602102001667-29c495d72fe3?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2832&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

pages = st.sidebar.selectbox('Pagina',('Home','Wereld','West Pacific', 'China'))

if pages == 'Home':
    
    st.header("**Doodsoorzaken wereld**")
    st.markdown("Met dit dashboard worden de meest voorkomende doodsoorzaken laten zien over de hele wereld van 1990 tot 2019.")
    
    st.markdown("Bronnen: https://www.trouw.nl/nieuws/670-000-chinezen-sterven-jaarlijks-door-gevolgen-smog~b42aa97b/")

elif pages == 'Wereld':
    
    st.subheader('Alle doodsoorzaken over alle landen van de wereld (1990 - 2019)')
    st.markdown("Hieronder zijn allereerst boxplots weergeven van alle doodsoorzaken. Het opvallende aan de rij met boxplots is dat hart- en vaatziekten er meteen bovenuit springt. Om deze reden gaan we hart- en vaatziekten verder uitvergroten en onderzoeken waar dit vandaan komt.")
    
    fig = go.Figure()
    for col in df_death_box:
        fig.add_trace(go.Box(y=df_death_box[col].values, name=df_death_box[col].name))
        fig.update_layout(title_text="Boxplots all diseases")
    st.plotly_chart(fig) 
    
    st.markdown("Op de map hieronder de verdeling van hart- en vaatziekten over de hele wereld te zien tussen 1990 - 2019.")
    
    fig2 = px.scatter_geo(df_death, locations="Code", color="Region", hover_name="Country", size="Cardiovascular Diseases",
         animation_frame="Year", projection="natural earth")
    st.plotly_chart(fig2)
    
elif pages == 'West Pacific':
    
    st.subheader('Hart- en vaatziekten West Pacific (1990 - 2019)')
    st.markdown("In de onderstaande map is te zien dat China in grote getallen domineerd in het aantal doodsoorzaken door hart- en vaatziekten. China telt 4,5 miljoen doden door hart- en vaatziekten. Een reden dat deze doodsoorzaak zo hoog is in China kan komen doordat de bevolking veel blootgesteld wordt aan luchtvervuiling, ook wel smog.")
    #number = st.number_input('Voeg een voertuig ID in', min_value=1, max_value=200, value=1, step=1)
    
    map = folium.Map(location=[12.5657, 104.9910], zoom_start=2, tiles = 'openstreetmap')
    folium.Choropleth(
            geo_data=wp_3,
            data=wp,
            columns=['Country', 'Cardiovascular Diseases'],  
            key_on='feature.properties.admin', 
            fill_color='YlOrRd',
            nan_fill_color="White", 
            fill_opacity=1,
            line_opacity=0.2,
            legend_name='Cardiovascular Diseases) ', 
            highlight=True,
            line_color='black').add_to(map)
    folium_static(map)
    
    st.markdown("Hieronder is een histogram weergeven van de verdeling van de doodsoorzaak hart- en vaatziektes over de landen in de west pacific. In het histogram is, net zoals in de map hierboven, duidelijk te zien dat China de meeste doden heeft als gevolg van hart- en vaatziekten.")
                
    fig3 = px.histogram(wp_c, x="Year", y="Cardiovascular Diseases", color="Country")
    st.plotly_chart(fig3)
    
elif pages == 'China':
    
    st.subheader("Hart- en vaatziekten China (1990 - 2019)")
    st.markdown("Hieronder is een lijndiagram te zien van het verloop van doden door hart- en vaatziekten in China over de jaren heen. Wat meteen opvalt is dat na 2005 het aantal doden afneemt en na 2006 minimaal stijgt.")
    
    fig4 = px.line(wp_2, x="Diabetes Mellitus", y="Cardiovascular Diseases", symbol = "Country", hover_name="Year")
    #fig.update_xaxes(autorange = True)
    st.plotly_chart(fig4)
    
    fig5 = plt.figure()
    sns.regplot(data = wp, y='Cardiovascular Diseases',x = 'Diabetes Mellitus', ci = None )
    st.pyplot(fig5)
    
    fig6 = plt.figure()
    sns.regplot(data = wp, y='Cardiovascular Diseases',x = 'Tuberculosis', ci = None )
    st.pyplot(fig6)
    
    fig7 = plt.figure()
    plt.scatter(x_train2, y_train2 ,color='r') 
    plt.plot(x_test2, y_pred, color='k') 
    st.pyplot(fig7)
    

