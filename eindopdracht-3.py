#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#import missingno as mnso
import plotly.express as px
import os
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install gapminder
from gapminder import gapminder 
import streamlit as st


# In[2]:


gapminder = gapminder[["country", "continent"]]
#gapminder.head()


# In[3]:


gapminder.rename(columns ={'country':'Country'},inplace = True)
#gapminder.head()


# In[4]:


df_continent = pd.read_csv("covid 19 CountryWise.csv")
#df_continent


# In[5]:


df_continent = df_continent[["Country", "Region"]]


# In[6]:


df_death = pd.read_csv("cause_of_deaths.csv")


# In[7]:


df_death.rename(columns ={'Country/Territory':'Country'},inplace = True)


# In[8]:


df_death = df_death.merge(df_continent, left_on = 'Country', right_on = 'Country')
#df_death = df_death.merge(gapminder, on='Country')


# In[9]:


#mnso.matrix(df_death)


# In[10]:


#df_death.columns


# In[11]:


#df_death.quantile(0.75)


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:

def interactive_boxplot(dataframe):
    fig = go.Figure()

    for col in df_death_box:
        fig.add_trace(go.Box(y=df_death_box[col].values, name=df_death_box[col].name))

        fig.update_layout(title_text="Boxplots all diseases")

        return dataframe

# In[16]:

df_drowning = df_death[["Cardiovascular Diseases", "Diabetes Mellitus", "Tuberculosis","Region", "Country", "Year" ]]
fig = px.box(df_drowning, x="Region", y="Cardiovascular Diseases", color="Region")
    

# In[17]:


df2 = df_drowning.groupby('Region')['Cardiovascular Diseases'].sum()


# In[18]:


#df2.head(10)


# In[19]:


wp = df_drowning.loc[(df_drowning['Region'] == "Western Pacific")]


# In[20]:

def interactive_map(mapcirkel):
    df = px.data.gapminder()
    fig1 = px.scatter_geo(df_death, locations="Code", color="Region", hover_name="Country", size="Cardiovascular Diseases",
         animation_frame="Year", projection="natural earth")
    
    return mapcirkel


# In[21]:


plt.figure(figsize=(10,10))
plt.ticklabel_format(useOffset = True, style = 'plain')
sns.barplot(data = df_death, x= 'Region', y='Meningitis',ci = None, palette = 'Set2' )
plt.xticks(rotation = 45, size=12)
plt.yticks(size = 12)
plt.xlabel('Region',fontsize = 18)
plt.ylabel('Total Cases', fontsize = 18)
plt.title ('Total cases meningitis per region',fontsize = 18)


# In[22]:


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


# In[23]:


df_death_hist = df_death_hist.groupby(["Year", 'Region'])['Cardiovascular Diseases'].mean().reset_index(name="Cardiovascular Diseases")

df_death_hist["Year"] = df_death_hist["Year"].astype(str)


# In[24]:


wp_c = wp.groupby(["Year", "Country"])['Cardiovascular Diseases'].mean().reset_index(name="Cardiovascular Diseases")

wp_c["Year"] = wp_c["Year"].astype(str)


# In[25]:

def histogram(zuidoost):
    fig2 = px.histogram(wp_c, x="Year", y="Cardiovascular Diseases", color="Country")
    
    return zuidoost


# In[26]:


wp_2 = wp.loc[(wp['Country'] == "China")].sort_values("Year")


# In[30]:

def scatter(china):
    fig3 = px.line(wp_2, x="Diabetes Mellitus", y="Cardiovascular Diseases", symbol = "Country", hover_name="Year")
    fig.update_xaxes(autorange = True)
    
    return china


# In[29]:


#fig = px.scatter_matrix(wp_2,
    #dimensions=["Cardiovascular Diseases", "Diabetes Mellitus", "Tuberculosis"], hover_name="Year")
#fig.show()


# In[ ]:


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

pages = st.sidebar.selectbox('Pagina',('Home','Wereld','Zuid-Oost Azië', 'China'))

if pages == 'Home':
    
    st.header("**Doodsoorzaken wereld**")
    st.markdown("Met dit dashboard worden de meest voorkomende doodsoorzaken laten zien over de hele wereld van 1990 tot 2019.")
    

elif pages == 'Wereld':
    
    st.subheader('Alle doodsoorzaken over alle landen van de wereld (1990 - 2019)')
    st.markdown("In de kaart zijn de energiebehoeftes van Schiphol tradepark en WFO per gebouw weergegeven. Hiermee gaan we een geschatte energievraag analyseren van op basis van voertuigregistraties. Op basis van publieke data en deelse CBS data. Wordt een inschatting gemaakt hoe de energiebehoefte/voorraad op bedrijventerreinen.")
    
    fig = go.Figure()
    for col in df_death_box:
        fig.add_trace(go.Box(y=df_death_box[col].values, name=df_death_box[col].name))
        fig.update_layout(title_text="Boxplots all diseases")
    st.plotly_chart(fig) 
    
    fig1 = px.scatter_geo(df_death, locations="Code", color="Region", hover_name="Country", size="Cardiovascular Diseases",
         animation_frame="Year", projection="natural earth")
    st.plotly_chart(fig1)
    
elif pages == 'Zuid-Oost Azië':
    
    st.subheader('Hart- en vaatziekten, suikerziekte (diabetes) & tuberculose over Zuid-Oost Azië (1990 - 2019)')
    st.markdown('In onderstaande velden voer een voertuig ID in om het energieverbruik over een dag van een vrachtwagen te visualiseren.')
    #number = st.number_input('Voeg een voertuig ID in', min_value=1, max_value=200, value=1, step=1)
    
    fig2 = px.histogram(wp_c, x="Year", y="Cardiovascular Diseases", color="Country")
    st.plotly_chart(fig2)
    
elif pages == 'China':
    
    st.subheader("Hart- en vaatziekten, suikerziekte (diabetes) & tuberculose in China (1990 - 2019)")
    st.markdown("Voor enkele transporteurs is hieronder een schatting van wat de energievraag zou zijn in het geval van dat de ritten door een elektrische vrachtwagen zou worden uitgevoerd. Hierbij gaan we de vraag beantwoorden hoeveel elektriciteit er nodig zou zijn om de rit uit te voeren, en of dit op een bedrijventerrein kan worden gedaan of langs de snelweg.")
    
    fig3 = px.line(wp_2, x="Diabetes Mellitus", y="Cardiovascular Diseases", symbol = "Country", hover_name="Year")
    fig.update_xaxes(autorange = True)
    st.plotly_chart(fig3)

