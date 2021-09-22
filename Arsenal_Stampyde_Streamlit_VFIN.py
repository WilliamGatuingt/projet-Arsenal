import pandas as pd #importation de pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import FactorRange
from bokeh.models import ColumnDataSource
output_notebook()
import warnings
from math import *
import streamlit as st
import plotly_express as px
from urllib.request import urlopen
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')


#_____________________________________________________________________________________________________________________________________
#@st.cache(allow_output_mutation=True)

@st.cache
def read_csv(name):
  
    df = pd.read_csv(name, sep = ',')
    return df


df = read_csv('df_full_premierleague.csv')

#df = pd.read_csv('df_full_premierleague.csv', sep = ',') #lecture du fichier csv



df = df.iloc[:,2:38]

#La premiere colonne de df est l'index. Nous allons donc la supprimer
#df.drop('Unnamed: 0', axis=1, inplace=True)

#La saison 20/21 n'est pas compléte. Nous allons donc la supprimer également
df.drop(df.loc[df['season']=='20/21'].index, inplace=True)

#Gestion de Nan 
def valeur_manquante(df):
    flag=0
    for col in df.columns:
            if df[col].isna().sum() > 0:
                flag=1
                print(f'"{col}": {df[col].isna().sum()} valeurs manquantes')
valeur_manquante(df)

#_____________________________________________________________________________________________________________________________________


df['Winner'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]>df['goal_away_ft'][i]):
        df['Winner'][i] = df['home_team'][i]
    elif (df['goal_home_ft'][i]<df['goal_away_ft'][i]):
        df['Winner'][i] = df['away_team'][i]
    elif(df['goal_home_ft'][i]==df['goal_away_ft'][i]):
        df['Winner'][i] = 'Draw'

df['Loser'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]<df['goal_away_ft'][i]):
        df['Loser'][i] = df['home_team'][i]
    elif (df['goal_home_ft'][i]>df['goal_away_ft'][i]):
        df['Loser'][i] = df['away_team'][i]
    elif(df['goal_home_ft'][i]==df['goal_away_ft'][i]):
        df['Loser'][i] = 'Draw'

df['Draw_Home'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]==df['goal_away_ft'][i]):
        df['Draw_Home'][i] = 1


df['Draw_Away'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]==df['goal_away_ft'][i]):
        df['Draw_Away'][i] = 1


df['Winner_Home'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]>df['goal_away_ft'][i]):
        df['Winner_Home'][i] = 1


df['Winner_Away'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]<df['goal_away_ft'][i]):
        df['Winner_Away'][i] = 1


df['Loser_Home'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]<df['goal_away_ft'][i]):
        df['Loser_Home'][i] = 1


df['Loser_Away'] = np.nan
for i in range(len(df['home_team'])):
    if (df['goal_home_ft'][i]>df['goal_away_ft'][i]):
        df['Loser_Away'][i] = 1

#Extraction des résultats dans 'results'
results=df[['season','home_team','away_team','Draw_Home','Draw_Away','Winner_Home','Winner_Away',
            'Loser_Home','Loser_Away','Winner','Loser']]

#Variable des victoires d'Arsenal à domicile
ars_wins_h=results.loc[results['home_team']=='Arsenal']
ars_wins_h=ars_wins_h.loc[ars_wins_h['Winner_Home']==1]
ars_wins_h=ars_wins_h.groupby(ars_wins_h['season']).sum()

#Variable des victoires d'Arsenal à l'exterieur
ars_wins_a=results.loc[results['away_team']=='Arsenal']
ars_wins_a=ars_wins_a.loc[ars_wins_a['Winner_Away']==1]
ars_wins_a=ars_wins_a.groupby(ars_wins_a['season']).sum()

#Variable des victoires totales d'Arsenal
ars_all_wins=ars_wins_h[['Winner_Home']]
ars_all_wins['away_wins']=ars_wins_a[['Winner_Away']]
ars_all_wins['total_wins']=ars_all_wins['Winner_Home']+ars_all_wins['away_wins']
ars_all_wins.rename(columns={'Winner_Home':'home_wins'},inplace=True)

#Création colonne du nombre de points gagnés en fonction des victoires
ars_all_wins['total_points']=ars_all_wins['total_wins']*3


#Variable des victoires du Big5 à domicile
big5_wins_h = results.loc[(results["home_team"]=="Manchester City")|(results["home_team"]=="Manchester United")|
                          (results["home_team"]=="Chelsea")|(results["home_team"]=="Liverpool")|
                          (results["home_team"]=="Tottenham Hotspur")]
big5_wins_h = big5_wins_h. loc[big5_wins_h ['Winner_Home']==1]
big5_wins_h =big5_wins_h.groupby(big5_wins_h['season']).sum()/5

#Variable des victoires du Big5 à l'exterieur
big5_wins_a = results.loc[(results["away_team"]=="Manchester City")|(results["away_team"]=="Manchester United")|(results["away_team"]=="Chelsea")|(results["away_team"]=="Liverpool")|(results["away_team"]=="Tottenham Hotspur")]
big5_wins_a = big5_wins_a. loc[big5_wins_a ['Winner_Away']==1]
big5_wins_a =big5_wins_a.groupby(big5_wins_a['season']).sum()/5

#Variable des victoires totales d'Arsenal
all_big5_wins=big5_wins_h[['Winner_Home']]
all_big5_wins['away_wins']=big5_wins_a[['Winner_Away']]
all_big5_wins['total_wins']=all_big5_wins['Winner_Home']+all_big5_wins['away_wins']
all_big5_wins.rename(columns={'Winner_Home':'home_wins'},inplace=True)

#Création colonne du nombre de points gagnés en fonction des victoires
all_big5_wins['total_points']=all_big5_wins['total_wins']*3

#Variable des victoires des 14 autres équipes à domicile
teams14_wins_h=results.loc[(results["home_team"]=="Blackpool")|(results["home_team"]=="Stoke City")|(results["home_team"]=="Fulham")|(results["home_team"]=="Blackburn Rovers")|(results["home_team"]=="Sunderland")|(results["home_team"]=="Bolton Wanderers")|(results["home_team"]=="Birmingham City")|(results["home_team"]=="West Bromwich Albion")|(results["home_team"]=="West Ham United")|(results["home_team"]=="AstonVilla")|(results["home_team"]=="Everton")|(results["home_team"]=="Newcastle United")|(results["home_team"]=="Wigan Athletic")|(results["home_team"]=="Wolverhampton Wanderers")|(results["home_team"]=="Swansea City")|(results["home_team"]=="Queens Park Rangers")|(results["home_team"]=="Norwich City")|(results["home_team"]=="Reading")|(results["home_team"]=="Southampton")|(results["home_team"]=="Crystal Palace")|(results["home_team"]=="Cardiff City")|(results["home_team"]=="Hull City")|(results["home_team"]=="Burnley")|(results["home_team"]=="Leicester City")|(results["home_team"]=="Watford")|(results["home_team"]=="AFC Bournemouth")|(results["home_team"]=="Middlesbrough")|(results["home_team"]=="Brighton and Hove Albion")|(results["home_team"]=="Huddersfield Town")|(results["home_team"]=="Sheffield United")]
teams14_wins_h = teams14_wins_h.loc[teams14_wins_h['Winner_Home']==1]
teams14_wins_h =teams14_wins_h.groupby(teams14_wins_h['season']).sum()/14

#Variable des victoires des 14 autres équipes à l'exterieur
teams14_wins_a=results.loc[(results["away_team"]=="Blackpool")|(results["away_team"]=="Stoke City")|(results["away_team"]=="Fulham")|(results["away_team"]=="Blackburn Rovers")|(results["away_team"]=="Sunderland")|(results["away_team"]=="Bolton Wanderers")|(results["away_team"]=="Birmingham City")|(results["away_team"]=="West Bromwich Albion")|(results["away_team"]=="West Ham United")|(results["away_team"]=="AstonVilla")|(results["away_team"]=="Everton")|(results["away_team"]=="Newcastle United")|(results["away_team"]=="Wigan Athletic")|(results["away_team"]=="Wolverhampton Wanderers")|(results["away_team"]=="Swansea City")|(results["away_team"]=="Queens Park Rangers")|(results["away_team"]=="Norwich City")|(results["away_team"]=="Reading")|(results["away_team"]=="Southampton")|(results["away_team"]=="Crystal Palace")|(results["away_team"]=="Cardiff City")|(results["away_team"]=="Hull City")|(results["away_team"]=="Burnley")|(results["away_team"]=="Leicester City")|(results["away_team"]=="Watford")|(results["away_team"]=="AFC Bournemouth")|(results["away_team"]=="Middlesbrough")|(results["away_team"]=="Brighton and Hove Albion")|(results["away_team"]=="Huddersfield Town")|(results["away_team"]=="Sheffield United")]
teams14_wins_a = teams14_wins_a.loc[teams14_wins_a['Winner_Away']==1]
teams14_wins_a =teams14_wins_a.groupby(teams14_wins_a['season']).sum()/14

#Variable des victoires totales des 14 équipes
all_teams14_wins=teams14_wins_h[['Winner_Home']]
all_teams14_wins['away_wins']=teams14_wins_a[['Winner_Away']]
all_teams14_wins['total_wins']=all_teams14_wins['Winner_Home']+all_teams14_wins['away_wins']
all_teams14_wins.rename(columns={'Winner_Home':'home_wins'},inplace=True)

#Création colonne du nombre de points gagnés en fonction des victoires
all_teams14_wins['total_points']=all_teams14_wins['total_wins']*3

#Variable des victoires des 20 équipes à domicile
teams20_wins_h=results.loc[(results["home_team"]=="Arsenal")|(results["home_team"]=="Blackpool")|(results["home_team"]=="Manchester City")|(results["home_team"]=="Manchester United")|
                          (results["home_team"]=="Chelsea")|(results["home_team"]=="Liverpool")|
                          (results["home_team"]=="Tottenham Hotspur")|
                           (results["home_team"]=="Stoke City")|(results["home_team"]=="Fulham")|(results["home_team"]=="Blackburn Rovers")|(results["home_team"]=="Sunderland")|(results["home_team"]=="Bolton Wanderers")|(results["home_team"]=="Birmingham City")|(results["home_team"]=="West Bromwich Albion")|(results["home_team"]=="West Ham United")|(results["home_team"]=="AstonVilla")|(results["home_team"]=="Everton")|(results["home_team"]=="Newcastle United")|(results["home_team"]=="Wigan Athletic")|(results["home_team"]=="Wolverhampton Wanderers")|(results["home_team"]=="Swansea City")|(results["home_team"]=="Queens Park Rangers")|(results["home_team"]=="Norwich City")|(results["home_team"]=="Reading")|(results["home_team"]=="Southampton")|(results["home_team"]=="Crystal Palace")|(results["home_team"]=="Cardiff City")|(results["home_team"]=="Hull City")|(results["home_team"]=="Burnley")|(results["home_team"]=="Leicester City")|(results["home_team"]=="Watford")|(results["home_team"]=="AFC Bournemouth")|(results["home_team"]=="Middlesbrough")|(results["home_team"]=="Brighton and Hove Albion")|(results["home_team"]=="Huddersfield Town")|(results["home_team"]=="Sheffield United")]
teams20_wins_h = teams20_wins_h.loc[teams20_wins_h['Winner_Home']==1]
teams20_wins_h =teams20_wins_h.groupby(teams20_wins_h['season']).sum()/20

#Variable des victoires des 20 équipes à l'exterieur
teams20_wins_a=results.loc[(results["away_team"]=="Manchester City")|(results["away_team"]=="Manchester United")|(results["away_team"]=="Chelsea")|(results["away_team"]=="Liverpool")|(results["away_team"]=="Tottenham Hotspur")|(results["away_team"]=="Blackpool")|(results["away_team"]=="Stoke City")|(results["away_team"]=="Fulham")|(results["away_team"]=="Blackburn Rovers")|(results["away_team"]=="Sunderland")|(results["away_team"]=="Bolton Wanderers")|(results["away_team"]=="Birmingham City")|(results["away_team"]=="West Bromwich Albion")|(results["away_team"]=="West Ham United")|(results["away_team"]=="AstonVilla")|(results["away_team"]=="Everton")|(results["away_team"]=="Newcastle United")|(results["away_team"]=="Wigan Athletic")|(results["away_team"]=="Wolverhampton Wanderers")|(results["away_team"]=="Swansea City")|(results["away_team"]=="Queens Park Rangers")|(results["away_team"]=="Norwich City")|(results["away_team"]=="Reading")|(results["away_team"]=="Southampton")|(results["away_team"]=="Crystal Palace")|(results["away_team"]=="Cardiff City")|(results["away_team"]=="Hull City")|(results["away_team"]=="Burnley")|(results["away_team"]=="Leicester City")|(results["away_team"]=="Watford")|(results["away_team"]=="AFC Bournemouth")|(results["away_team"]=="Middlesbrough")|(results["away_team"]=="Brighton and Hove Albion")|(results["away_team"]=="Huddersfield Town")|(results["away_team"]=="Sheffield United")]
teams20_wins_a = teams20_wins_a. loc[teams20_wins_a['Winner_Away']==1]
teams20_wins_a =teams20_wins_a.groupby(teams20_wins_a['season']).sum()/20

#Variable des victoires totales des 20 équipes
all_teams20_wins=teams20_wins_h[['Winner_Home']]
all_teams20_wins['away_wins']=teams20_wins_a[['Winner_Away']]
all_teams20_wins['total_wins']=all_teams20_wins['Winner_Home']+all_teams20_wins['away_wins']
all_teams20_wins.rename(columns={'Winner_Home':'home_wins'},inplace=True)

#Création colonne du nombre de points gagnés en fonction des victoires
all_teams20_wins['total_points']=all_teams20_wins['total_wins']*3

#Variable des défaites d'Arsenal à domicile
ars_losses_h=results.loc[results['home_team']=='Arsenal']
ars_losses_h=ars_losses_h.loc[ars_losses_h['Loser_Home']==1]
ars_losses_h=ars_losses_h.groupby(ars_losses_h['season']).sum()

#Variable des défaites d'Arsenal à l'éxterieur
ars_losses_a=results.loc[results['away_team']=='Arsenal']
ars_losses_a=ars_losses_a.loc[ars_losses_a['Loser_Away']==1]
ars_losses_a=ars_losses_a.groupby(ars_losses_a['season']).sum()

#Variable des défaites totales d'Arsenal 
ars_all_losses=ars_losses_h[['Loser_Home']]
ars_all_losses['away_losses']=ars_losses_a[['Loser_Away']]
ars_all_losses['total_losses']=ars_all_losses['Loser_Home']+ars_all_losses['away_losses']
ars_all_losses.rename(columns={'Loser_Home':'home_losses'},inplace=True)

#Variable des défaites du Big5 à domicile
big5_losses_h= results.loc[(results["home_team"]=="Manchester City")|(results["home_team"]=="Manchester United")|(results["home_team"]=="Chelsea")|(results["home_team"]=="Liverpool")|(results["home_team"]=="Tottenham Hotspur")]
big5_losses_h= big5_losses_h . loc[big5_losses_h['Loser_Home']==1]
big5_losses_h=big5_losses_h .groupby(big5_losses_h ['season']).sum()/5

#Variable des défaites du Big5 à l'éxterieur
big5_losses_a = results.loc[(results["away_team"]=="Manchester City")|(results["away_team"]=="Manchester United")|(results["away_team"]=="Chelsea")|(results["away_team"]=="Liverpool")|(results["away_team"]=="Tottenham Hotspur")]
big5_losses_a = big5_losses_a. loc[big5_losses_a ['Loser_Away']==1]
big5_losses_a =big5_losses_a.groupby(big5_losses_a['season']).sum()/5

#Variable des défaites totales du Big5
all_big5_losses=big5_losses_h [['Loser_Home']]
all_big5_losses['away_losses']=big5_losses_a[['Loser_Away']]
all_big5_losses['total_losses']=all_big5_losses['Loser_Home']+all_big5_losses['away_losses']
all_big5_losses.rename(columns={'Loser_Home':'home_losses'},inplace=True)

#Variable des défaites des 14 autres équipes à domicile
teams14_losses_h=results.loc[(results["home_team"]=="Blackpool")|(results["home_team"]=="Stoke City")|(results["home_team"]=="Fulham")|(results["home_team"]=="Blackburn Rovers")|(results["home_team"]=="Sunderland")|(results["home_team"]=="Bolton Wanderers")|(results["home_team"]=="Birmingham City")|(results["home_team"]=="West Bromwich Albion")|(results["home_team"]=="West Ham United")|(results["home_team"]=="AstonVilla")|(results["home_team"]=="Everton")|(results["home_team"]=="Newcastle United")|(results["home_team"]=="Wigan Athletic")|(results["home_team"]=="Wolverhampton Wanderers")|(results["home_team"]=="Swansea City")|(results["home_team"]=="Queens Park Rangers")|(results["home_team"]=="Norwich City")|(results["home_team"]=="Reading")|(results["home_team"]=="Southampton")|(results["home_team"]=="Crystal Palace")|(results["home_team"]=="Cardiff City")|(results["home_team"]=="Hull City")|(results["home_team"]=="Burnley")|(results["home_team"]=="Leicester City")|(results["home_team"]=="Watford")|(results["home_team"]=="AFC Bournemouth")|(results["home_team"]=="Middlesbrough")|(results["home_team"]=="Brighton and Hove Albion")|(results["home_team"]=="Huddersfield Town")|(results["home_team"]=="Sheffield United")]
teams14_losses_h =teams14_losses_h.loc[teams14_losses_h['Loser_Home']==1]
teams14_losses_h=teams14_losses_h.groupby(teams14_losses_h['season']).sum()/14

#Variable des défaites des 14 autres équipes à l'exterieur
teams14_losses_a=results.loc[(results["away_team"]=="Blackpool")|(results["away_team"]=="Stoke City")|(results["away_team"]=="Fulham")|(results["away_team"]=="Blackburn Rovers")|(results["away_team"]=="Sunderland")|(results["away_team"]=="Bolton Wanderers")|(results["away_team"]=="Birmingham City")|(results["away_team"]=="West Bromwich Albion")|(results["away_team"]=="West Ham United")|(results["away_team"]=="AstonVilla")|(results["away_team"]=="Everton")|(results["away_team"]=="Newcastle United")|(results["away_team"]=="Wigan Athletic")|(results["away_team"]=="Wolverhampton Wanderers")|(results["away_team"]=="Swansea City")|(results["away_team"]=="Queens Park Rangers")|(results["away_team"]=="Norwich City")|(results["away_team"]=="Reading")|(results["away_team"]=="Southampton")|(results["away_team"]=="Crystal Palace")|(results["away_team"]=="Cardiff City")|(results["away_team"]=="Hull City")|(results["away_team"]=="Burnley")|(results["away_team"]=="Leicester City")|(results["away_team"]=="Watford")|(results["away_team"]=="AFC Bournemouth")|(results["away_team"]=="Middlesbrough")|(results["away_team"]=="Brighton and Hove Albion")|(results["away_team"]=="Huddersfield Town")|(results["away_team"]=="Sheffield United")]
teams14_losses_a= teams14_losses_a.loc[teams14_losses_a['Loser_Away']==1]
teams14_losses_a=teams14_losses_a.groupby(teams14_losses_a['season']).sum()/14

#Variable des défaites totales des 14 équipes
all_teams14_losses=teams14_losses_h[['Loser_Home']]
all_teams14_losses['away_losses']=teams14_losses_a[['Loser_Away']]
all_teams14_losses['total_losses']=all_teams14_losses['Loser_Home']+all_teams14_losses['away_losses']
all_teams14_losses.rename(columns={'Loser_Home':'home_losses'},inplace=True)

#Variable des défaites des 20 équipes à domicile
teams20_losses_h=results.loc[(results["home_team"]=="Arsenal")|(results["home_team"]=="Blackpool")|(results["home_team"]=="Manchester City")|(results["home_team"]=="Manchester United")|
                          (results["home_team"]=="Chelsea")|(results["home_team"]=="Liverpool")|
                          (results["home_team"]=="Tottenham Hotspur")|
                           (results["home_team"]=="Stoke City")|(results["home_team"]=="Fulham")|(results["home_team"]=="Blackburn Rovers")|(results["home_team"]=="Sunderland")|(results["home_team"]=="Bolton Wanderers")|(results["home_team"]=="Birmingham City")|(results["home_team"]=="West Bromwich Albion")|(results["home_team"]=="West Ham United")|(results["home_team"]=="AstonVilla")|(results["home_team"]=="Everton")|(results["home_team"]=="Newcastle United")|(results["home_team"]=="Wigan Athletic")|(results["home_team"]=="Wolverhampton Wanderers")|(results["home_team"]=="Swansea City")|(results["home_team"]=="Queens Park Rangers")|(results["home_team"]=="Norwich City")|(results["home_team"]=="Reading")|(results["home_team"]=="Southampton")|(results["home_team"]=="Crystal Palace")|(results["home_team"]=="Cardiff City")|(results["home_team"]=="Hull City")|(results["home_team"]=="Burnley")|(results["home_team"]=="Leicester City")|(results["home_team"]=="Watford")|(results["home_team"]=="AFC Bournemouth")|(results["home_team"]=="Middlesbrough")|(results["home_team"]=="Brighton and Hove Albion")|(results["home_team"]=="Huddersfield Town")|(results["home_team"]=="Sheffield United")]
teams20_losses_h = teams20_losses_h.loc[teams20_losses_h['Loser_Home']==1]
teams20_losses_h =teams20_losses_h.groupby(teams20_losses_h['season']).sum()/20

#Variable des défaites des 20 équipes à l'exterieur
teams20_losses_a=results.loc[(results["away_team"]=="Manchester City")|(results["away_team"]=="Manchester United")|(results["away_team"]=="Chelsea")|(results["away_team"]=="Liverpool")|(results["away_team"]=="Tottenham Hotspur")|(results["away_team"]=="Blackpool")|(results["away_team"]=="Stoke City")|(results["away_team"]=="Fulham")|(results["away_team"]=="Blackburn Rovers")|(results["away_team"]=="Sunderland")|(results["away_team"]=="Bolton Wanderers")|(results["away_team"]=="Birmingham City")|(results["away_team"]=="West Bromwich Albion")|(results["away_team"]=="West Ham United")|(results["away_team"]=="AstonVilla")|(results["away_team"]=="Everton")|(results["away_team"]=="Newcastle United")|(results["away_team"]=="Wigan Athletic")|(results["away_team"]=="Wolverhampton Wanderers")|(results["away_team"]=="Swansea City")|(results["away_team"]=="Queens Park Rangers")|(results["away_team"]=="Norwich City")|(results["away_team"]=="Reading")|(results["away_team"]=="Southampton")|(results["away_team"]=="Crystal Palace")|(results["away_team"]=="Cardiff City")|(results["away_team"]=="Hull City")|(results["away_team"]=="Burnley")|(results["away_team"]=="Leicester City")|(results["away_team"]=="Watford")|(results["away_team"]=="AFC Bournemouth")|(results["away_team"]=="Middlesbrough")|(results["away_team"]=="Brighton and Hove Albion")|(results["away_team"]=="Huddersfield Town")|(results["away_team"]=="Sheffield United")]
teams20_losses_a = teams20_losses_a. loc[teams20_losses_a['Loser_Away']==1]
teams20_losses_a =teams20_losses_a.groupby(teams20_losses_a['season']).sum()/20

#Variable des défaites totals des 20 équipes
all_teams20_losses=teams20_losses_h[['Loser_Home']]
all_teams20_losses['away_losses']=teams20_losses_a[['Loser_Away']]
all_teams20_losses['total_losses']=all_teams20_losses['Loser_Home']+all_teams20_losses['away_losses']
all_teams20_losses.rename(columns={'Loser_Home':'home_losses'},inplace=True)

#******************************************************************************************************************************************
from urllib.request import urlopen
from bs4 import BeautifulSoup
             
#Webscraping classement saison 2010/2011
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2010-2011/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())

#Dataframe comportant le classement de la saison 2010/2011    
saison10_11= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison10_11["season"]=np.nan
saison10_11["season"].fillna("10/11",inplace=True)

#Webscraping classement saison 2011/2012

page=urlopen("https://www.footmercato.net/angleterre/premier-league/2011-2012/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
#Dataframe comportant le classement de la saison 2011/2012     
saison11_12= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison11_12["season"]=np.nan
saison11_12["season"].fillna("11/12",inplace=True)

#Webscraping classement saison 2012/2013 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2012-2013/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison12_13= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison12_13["season"]=np.nan
saison12_13["season"].fillna("12/13",inplace=True)

#Webscraping classement saison 2013/2014 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2013-2014/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison13_14= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison13_14["season"]=np.nan
saison13_14["season"].fillna("13/14",inplace=True)

#Webscraping classement saison 2014/2015 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2014-2015/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison14_15= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison14_15["season"]=np.nan
saison14_15["season"].fillna("14/15",inplace=True)

#Webscraping classement saison 2015/2016 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2015-2016/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison15_16= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison15_16["season"]=np.nan
saison15_16["season"].fillna("15/16",inplace=True)

#Webscraping classement saison 2016/2017 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2016-2017/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison16_17= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison16_17["season"]=np.nan
saison16_17["season"].fillna("16/17",inplace=True)

#Webscraping classement saison 2017/2018 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2017-2018/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison17_18= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison17_18["season"]=np.nan
saison17_18["season"].fillna("17/18",inplace=True)

#Webscraping classement saison 2018/2019 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2018-2019/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison18_19= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison18_19["season"]=np.nan
saison18_19["season"].fillna("18/19",inplace=True)
             
#Webscraping classement saison 2019/2020 et création du dataframe 
page=urlopen("https://www.footmercato.net/angleterre/premier-league/2019-2020/classement")
soup = BeautifulSoup(page, 'html.parser')
rank = [] 
for element in soup.findAll('td',attrs={'class':'classement__rank'}):
    rank.append(element.text.strip())

teams= [] 
for element in soup.findAll('td',attrs={'class':'classement__team'}):
    teams.append(element.text.strip())   

pts= [] 
for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
    pts.append(element.text.strip())
    
saison19_20= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
saison19_20["season"]=np.nan
saison19_20["season"].fillna("19/20",inplace=True)

#Variable comportant les classement des 10 saisons
classement_all_season=pd.concat([saison10_11,saison11_12,saison12_13,saison13_14,saison14_15,saison15_16,saison16_17,saison17_18,saison18_19,saison19_20], ignore_index=True)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Variable points gagnés arsenal
classement_arsenal=classement_all_season.loc[classement_all_season['Equipes']=='Arsenal']
classement_arsenal['Points']=classement_arsenal['Points'].astype(int)

#Variable des points gagnés par Big 5
classement_big5=classement_all_season.loc[(classement_all_season["Equipes"]=="Man City")|(classement_all_season["Equipes"]=="Man United")|(classement_all_season["Equipes"]=="Chelsea")|(classement_all_season["Equipes"]=="Liverpool")|(classement_all_season["Equipes"]=="Tottenham")]
classement_big5['Points']=classement_big5['Points'].astype(int)
classement_big5=classement_big5.groupby(classement_big5['season']).sum()/5
             
#Variable de la moyenne des points gagnés par les équipes
classement_20teams=classement_all_season
classement_20teams['Points']=classement_20teams['Points'].astype(int)
classement_20teams=classement_20teams.groupby(classement_20teams['season']).sum()/20

#Varibale des points gagnés par les champions par saison
points_champion=classement_all_season.loc[classement_all_season['Classement']=='1']
points_champion['Points']=points_champion['Points'].astype(int)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#******************************************************************************************************************************************

#Extraction des buts dans 'goal_stat'
goal_stat=df[['season','home_team','away_team','goal_home_ft','goal_away_ft']]

#Variable des buts à domicile d'Arsenal
goal_ars_h=goal_stat.loc[goal_stat['home_team']=='Arsenal']
goal_ars_h.rename(columns={'goal_away_ft': 'goals_conced_H','goal_home_ft':'goals_scored_H'},inplace=True)
goal_ars_h['Goals difference']=goal_ars_h['goals_scored_H']-goal_ars_h['goals_conced_H']
goal_ars_h=goal_ars_h.groupby(goal_ars_h['season']).sum()

#Variable des buts à l'éxterieur d'Arsenal
goal_ars_a=goal_stat.loc[goal_stat['away_team']=='Arsenal']
goal_ars_a.rename(columns={'goal_away_ft': 'goals_scored_A','goal_home_ft': 'goals_conced_A'},inplace=True)
goal_ars_a['Goals difference']=goal_ars_a['goals_scored_A']-goal_ars_a['goals_conced_A']
goal_ars_a=goal_ars_a.groupby(goal_ars_a['season']).sum()

#Variable de tous buts d'Arsenal
all_goals_arsn=goal_ars_h[['goals_scored_H','goals_conced_H']]
all_goals_arsn['goals_scored_A']=goal_ars_a['goals_scored_A']
all_goals_arsn['goals_conced_A']=goal_ars_a['goals_conced_A']
all_goals_arsn['Total Goals Scored']=all_goals_arsn['goals_scored_H']+all_goals_arsn['goals_scored_A']
all_goals_arsn['Total Goals conced']=all_goals_arsn['goals_conced_H']+all_goals_arsn['goals_conced_A']
all_goals_arsn['Goals difference']=all_goals_arsn['Total Goals Scored']-all_goals_arsn['Total Goals conced']

#Variable des buts à domicile du Big5
goal_big5_h=goal_stat.loc[(goal_stat["home_team"]=="Manchester City")|(goal_stat["home_team"]=="Manchester United")|(goal_stat["home_team"]=="Chelsea")|(goal_stat["home_team"]=="Liverpool")|(goal_stat["home_team"]=="Tottenham Hotspur")]
goal_big5_h.rename(columns={'goal_away_ft': 'goals_conced_H','goal_home_ft':'goals_scored_H'},inplace=True)
goal_big5_h['Goals difference']=goal_big5_h['goals_scored_H']-goal_big5_h['goals_conced_H']
goal_big5_h=goal_big5_h.groupby(goal_big5_h['season']).sum()/5

#Variable des buts à l'éxterieur du Big5
goal_big5_a=goal_stat.loc[(goal_stat["away_team"]=="Manchester City")|(goal_stat["away_team"]=="Manchester United")|(goal_stat["away_team"]=="Chelsea")|(goal_stat["away_team"]=="Liverpool")|(goal_stat["away_team"]=="Tottenham Hotspur")]
goal_big5_a.rename(columns={'goal_home_ft': 'goals_conced_A','goal_away_ft':'goals_scored_A'},inplace=True)
goal_big5_a['Goals difference']=goal_big5_a['goals_scored_A']-goal_big5_a['goals_conced_A']
goal_big5_a=goal_big5_a.groupby(goal_big5_a['season']).sum()/5

#Variable de tous buts du Big5
all_goals_big5=goal_big5_h[['goals_scored_H','goals_conced_H']]
all_goals_big5['goals_scored_A']=goal_big5_a['goals_scored_A']
all_goals_big5['goals_conced_A']=goal_big5_a['goals_conced_A']
all_goals_big5['Total Goals Scored']=all_goals_big5['goals_scored_H']+all_goals_big5['goals_scored_A']
all_goals_big5['Total Goals conced']=all_goals_big5['goals_conced_H']+all_goals_big5['goals_conced_A']
all_goals_big5['Goals difference']=all_goals_big5['Total Goals Scored']-all_goals_big5['Total Goals conced']

#Variable des buts à domicile des 14 autres équipes
goal_team14_h=goal_stat.loc[(goal_stat["home_team"]=="Blackpool")|(goal_stat["home_team"]=="Stoke City")|(goal_stat["home_team"]=="Fulham")|(goal_stat["home_team"]=="Blackburn Rovers")|(goal_stat["home_team"]=="Sunderland")|(goal_stat["home_team"]=="Bolton Wanderers")|(goal_stat["home_team"]=="Birmingham City")|(goal_stat["home_team"]=="West Bromwich Albion")|(goal_stat["home_team"]=="West Ham United")|(goal_stat["home_team"]=="AstonVilla")|(goal_stat["home_team"]=="Everton")|(goal_stat["home_team"]=="Newcastle United")|(goal_stat["home_team"]=="Wigan Athletic")|(goal_stat["home_team"]=="Wolverhampton Wanderers")|(goal_stat["home_team"]=="Swansea City")|(goal_stat["home_team"]=="Queens Park Rangers")|(goal_stat["home_team"]=="Norwich City")|(goal_stat["home_team"]=="Reading")|(goal_stat["home_team"]=="Southampton")|(goal_stat["home_team"]=="Crystal Palace")|(goal_stat["home_team"]=="Cardiff City")|(goal_stat["home_team"]=="Hull City")|(goal_stat["home_team"]=="Burnley")|(goal_stat["home_team"]=="Leicester City")|(goal_stat["home_team"]=="Watford")|(goal_stat["home_team"]=="AFC Bournemouth")|(goal_stat["home_team"]=="Middlesbrough")|(goal_stat["home_team"]=="Brighton and Hove Albion")|(goal_stat["home_team"]=="Huddersfield Town")|(goal_stat["home_team"]=="Sheffield United")]
goal_team14_h.rename(columns={'goal_away_ft': 'goals_conced_H','goal_home_ft':'goals_scored_H'},inplace=True)
goal_team14_h['Goals difference']=goal_team14_h['goals_scored_H']-goal_team14_h['goals_conced_H']
goal_team14_h=goal_team14_h.groupby(goal_team14_h['season']).sum()/14

#Variable des buts à l'éxterieur des 14 autres équipes
goal_team14_a=goal_stat.loc[(goal_stat["away_team"]=="Blackpool")|(goal_stat["away_team"]=="Stoke City")|(goal_stat["away_team"]=="Fulham")|(goal_stat["away_team"]=="Blackburn Rovers")|(goal_stat["away_team"]=="Sunderland")|(goal_stat["away_team"]=="Bolton Wanderers")|(goal_stat["away_team"]=="Birmingham City")|(goal_stat["away_team"]=="West Bromwich Albion")|(goal_stat["away_team"]=="West Ham United")|(goal_stat["away_team"]=="AstonVilla")|(goal_stat["away_team"]=="Everton")|(goal_stat["away_team"]=="Newcastle United")|(goal_stat["away_team"]=="Wigan Athletic")|(goal_stat["away_team"]=="Wolverhampton Wanderers")|(goal_stat["away_team"]=="Swansea City")|(goal_stat["away_team"]=="Queens Park Rangers")|(goal_stat["away_team"]=="Norwich City")|(goal_stat["away_team"]=="Reading")|(goal_stat["away_team"]=="Southampton")|(goal_stat["away_team"]=="Crystal Palace")|(goal_stat["away_team"]=="Cardiff City")|(goal_stat["away_team"]=="Hull City")|(goal_stat["away_team"]=="Burnley")|(goal_stat["away_team"]=="Leicester City")|(goal_stat["away_team"]=="Watford")|(goal_stat["away_team"]=="AFC Bournemouth")|(goal_stat["away_team"]=="Middlesbrough")|(goal_stat["away_team"]=="Brighton and Hove Albion")|(goal_stat["away_team"]=="Huddersfield Town")|(goal_stat["away_team"]=="Sheffield United")]
goal_team14_a.rename(columns={'goal_home_ft': 'goals_conced_A','goal_away_ft':'goals_scored_A'},inplace=True)
goal_team14_a['Goals difference']=goal_team14_a['goals_scored_A']-goal_team14_a['goals_conced_A']
goal_team14_a=goal_team14_a.groupby(goal_team14_a['season']).sum()/14

#Variable de tous les buts des 14 autres équipes
all_goals_teams14=goal_team14_h[['goals_scored_H','goals_conced_H']]
all_goals_teams14['goals_scored_A']=goal_team14_a['goals_scored_A']
all_goals_teams14['goals_conced_A']=goal_team14_a['goals_conced_A']
all_goals_teams14['Total Goals Scored']=all_goals_teams14['goals_scored_H']+all_goals_teams14['goals_scored_A']
all_goals_teams14['Total Goals conced']=all_goals_teams14['goals_conced_H']+all_goals_teams14['goals_conced_A']
all_goals_teams14['Goals difference']=all_goals_teams14['Total Goals Scored']-all_goals_teams14['Total Goals conced']

#Variable des buts à domicile des 20 équipes
goal_team20_h=goal_stat.loc[(goal_stat["home_team"]=="Arsenal")|(goal_stat["home_team"]=="Manchester City")|(goal_stat["home_team"]=="Manchester United")|(goal_stat["home_team"]=="Chelsea")|(goal_stat["home_team"]=="Liverpool")|(goal_stat["home_team"]=="Tottenham Hotspur")|(goal_stat["home_team"]=="Blackpool")|(goal_stat["home_team"]=="Stoke City")|(goal_stat["home_team"]=="Fulham")|(goal_stat["home_team"]=="Blackburn Rovers")|(goal_stat["home_team"]=="Sunderland")|(goal_stat["home_team"]=="Bolton Wanderers")|(goal_stat["home_team"]=="Birmingham City")|(goal_stat["home_team"]=="West Bromwich Albion")|(goal_stat["home_team"]=="West Ham United")|(goal_stat["home_team"]=="AstonVilla")|(goal_stat["home_team"]=="Everton")|(goal_stat["home_team"]=="Newcastle United")|(goal_stat["home_team"]=="Wigan Athletic")|(goal_stat["home_team"]=="Wolverhampton Wanderers")|(goal_stat["home_team"]=="Swansea City")|(goal_stat["home_team"]=="Queens Park Rangers")|(goal_stat["home_team"]=="Norwich City")|(goal_stat["home_team"]=="Reading")|(goal_stat["home_team"]=="Southampton")|(goal_stat["home_team"]=="Crystal Palace")|(goal_stat["home_team"]=="Cardiff City")|(goal_stat["home_team"]=="Hull City")|(goal_stat["home_team"]=="Burnley")|(goal_stat["home_team"]=="Leicester City")|(goal_stat["home_team"]=="Watford")|(goal_stat["home_team"]=="AFC Bournemouth")|(goal_stat["home_team"]=="Middlesbrough")|(goal_stat["home_team"]=="Brighton and Hove Albion")|(goal_stat["home_team"]=="Huddersfield Town")|(goal_stat["home_team"]=="Sheffield United")]
goal_team20_h.rename(columns={'goal_away_ft': 'goals_conced_H','goal_home_ft':'goals_scored_H'},inplace=True)
goal_team20_h['Goals difference']=goal_team20_h['goals_scored_H']-goal_team20_h['goals_conced_H']
goal_team20_h=goal_team20_h.groupby(goal_team20_h['season']).sum()/20

#Variable des buts à l'éxterieur des 20 équipes
goal_team20_a=goal_stat.loc[(goal_stat["away_team"]=="Arsenal")|(goal_stat["away_team"]=="Manchester City")|(goal_stat["away_team"]=="Manchester United")|(goal_stat["away_team"]=="Chelsea")|(goal_stat["away_team"]=="Liverpool")|(goal_stat["away_team"]=="Tottenham Hotspur")|(goal_stat["away_team"]=="Blackpool")|(goal_stat["away_team"]=="Stoke City")|(goal_stat["away_team"]=="Fulham")|(goal_stat["away_team"]=="Blackburn Rovers")|(goal_stat["away_team"]=="Sunderland")|(goal_stat["away_team"]=="Bolton Wanderers")|(goal_stat["away_team"]=="Birmingham City")|(goal_stat["away_team"]=="West Bromwich Albion")|(goal_stat["away_team"]=="West Ham United")|(goal_stat["away_team"]=="AstonVilla")|(goal_stat["away_team"]=="Everton")|(goal_stat["away_team"]=="Newcastle United")|(goal_stat["away_team"]=="Wigan Athletic")|(goal_stat["away_team"]=="Wolverhampton Wanderers")|(goal_stat["away_team"]=="Swansea City")|(goal_stat["away_team"]=="Queens Park Rangers")|(goal_stat["away_team"]=="Norwich City")|(goal_stat["away_team"]=="Reading")|(goal_stat["away_team"]=="Southampton")|(goal_stat["away_team"]=="Crystal Palace")|(goal_stat["away_team"]=="Cardiff City")|(goal_stat["away_team"]=="Hull City")|(goal_stat["away_team"]=="Burnley")|(goal_stat["away_team"]=="Leicester City")|(goal_stat["away_team"]=="Watford")|(goal_stat["away_team"]=="AFC Bournemouth")|(goal_stat["away_team"]=="Middlesbrough")|(goal_stat["away_team"]=="Brighton and Hove Albion")|(goal_stat["away_team"]=="Huddersfield Town")|(goal_stat["away_team"]=="Sheffield United")]
goal_team20_a.rename(columns={'goal_home_ft': 'goals_conced_A','goal_away_ft':'goals_scored_A'},inplace=True)
goal_team20_a['Goals difference']=goal_team20_a['goals_scored_A']-goal_team20_a['goals_conced_A']
goal_team20_a=goal_team20_a.groupby(goal_team20_a['season']).sum()/20

#Variable de tous les buts des 20 équipes
all_goals_teams20=goal_team20_h[['goals_scored_H','goals_conced_H']]
all_goals_teams20['goals_scored_A']=goal_team20_a['goals_scored_A']
all_goals_teams20['goals_conced_A']=goal_team20_a['goals_conced_A']
all_goals_teams20['Total Goals Scored']=all_goals_teams20['goals_scored_H']+all_goals_teams20['goals_scored_A']
all_goals_teams20['Total Goals conced']=all_goals_teams20['goals_conced_H']+all_goals_teams20['goals_conced_A']
all_goals_teams20['Goals difference']=all_goals_teams20['Total Goals Scored']-all_goals_teams20['Total Goals conced']

#******************************************************************************************************************************************

# Analyse approfondie - Réalisme aggressif - Nombre de tirs par match / nombre de tirs cadré par match / nombre de goals par match

#Extraction des tirs, tirs cadrés et goals dans 'attack_realism'
attack_realism=df[['season','home_team','away_team','home_shots','home_shots_on_target','goal_home_ft','away_shots','away_shots_on_target','goal_away_ft']]

#Variable des tirs à domicile d'Arsenal
shots_ars_h=attack_realism.loc[attack_realism['home_team']=='Arsenal']
shots_ars_h.rename(columns={'away_shots': 'shots_conced_h','home_shots':'shots_h'},inplace=True)
shots_ars_h['Shots difference home games']=shots_ars_h['shots_h']-shots_ars_h['shots_conced_h']
shots_ars_h=shots_ars_h.groupby(shots_ars_h['season']).sum()

#Variable des tirs à l'exterieur d'Arsenal
shots_ars_a=attack_realism.loc[attack_realism['away_team']=='Arsenal']
shots_ars_a.rename(columns={'away_shots': 'shots_a','home_shots': 'shots_conced_a'},inplace=True)
shots_ars_a['Shot difference away games']=shots_ars_a['shots_a']-shots_ars_a['shots_conced_a']
shots_ars_a=shots_ars_a.groupby(shots_ars_a['season']).sum()

#Variable de tous les tirs d'Arsenal
all_shots_arsn=shots_ars_h[['shots_h','shots_conced_h']]
all_shots_arsn['shots_a']=shots_ars_a['shots_a']
all_shots_arsn['shots_conced_a']=shots_ars_a['shots_conced_a']
all_shots_arsn['Total Shots']=all_shots_arsn['shots_h']+all_shots_arsn['shots_a']
all_shots_arsn['Total Shots Conced']=all_shots_arsn['shots_conced_h']+all_shots_arsn['shots_conced_a']
all_shots_arsn['Shots difference']=all_shots_arsn['Total Shots']-all_shots_arsn['Total Shots Conced']

#Variable des tirs cadrés à domicile d'Arsenal
target_shots_ars_h=attack_realism.loc[attack_realism['home_team']=='Arsenal']
target_shots_ars_h.rename(columns={'away_shots_on_target': 'target_shots_conced_h','home_shots_on_target':'target_shots_h'},inplace=True)
target_shots_ars_h['Target Shots difference home games']=target_shots_ars_h['target_shots_h']-target_shots_ars_h['target_shots_conced_h']
target_shots_ars_h=target_shots_ars_h.groupby(target_shots_ars_h['season']).sum()

#Variable des tirs cadrés à l'exterieur d'Arsenal
target_shots_ars_a=attack_realism.loc[attack_realism['away_team']=='Arsenal']
target_shots_ars_a.rename(columns={'away_shots_on_target': 'target_shots_a','home_shots_on_target': 'target_shots_conced_a'},inplace=True)
target_shots_ars_a['Target Shot difference away games']=target_shots_ars_a['target_shots_a']-target_shots_ars_a['target_shots_conced_a']
target_shots_ars_a=target_shots_ars_a.groupby(target_shots_ars_a['season']).sum()

#Variable de tous les tirs cadrés d'Arsenal
all_target_shots_arsn=target_shots_ars_h[['target_shots_h','target_shots_conced_h']]
all_target_shots_arsn['target_shots_a']=target_shots_ars_a['target_shots_a']
all_target_shots_arsn['target_shots_conced_a']=target_shots_ars_a['target_shots_conced_a']
all_target_shots_arsn['Total Target Shots']=all_target_shots_arsn['target_shots_h']+all_target_shots_arsn['target_shots_a']
all_target_shots_arsn['Total Target Shots Conced']=all_target_shots_arsn['target_shots_conced_h']+all_target_shots_arsn['target_shots_conced_a']
all_target_shots_arsn['Target Shots difference']=all_target_shots_arsn['Total Target Shots']-all_target_shots_arsn['Total Target Shots Conced']

attack_realism_data = all_shots_arsn.merge(right = all_target_shots_arsn, on = 'season', how = 'left')

attack_realism_data =attack_realism_data.merge(right = all_goals_arsn, on = 'season', how = 'left')

#Calcul ratio HOME - Ratio tirs vs tirs cadrés
attack_realism_data['HOME - Ratio tirs vs tirs cadrés'] = attack_realism_data['target_shots_h']/attack_realism_data['shots_h']

#Calcul ratio HOME - Ratio tirs vs goals - new as per meeting with Gabriel
attack_realism_data['HOME - Ratio tirs - goals'] = attack_realism_data['goals_scored_H']/attack_realism_data['shots_h']

#Calcul ratio HOME - Ratio tirs cadrés vs goals
attack_realism_data['HOME - Ratio tirs cadrés - goals'] = attack_realism_data['goals_scored_H']/attack_realism_data['target_shots_h']

#Calcul ratio AWAY - Ratio tirs vs tirs cadrés
attack_realism_data['AWAY - Ratio tirs vs tirs cadrés'] = attack_realism_data['target_shots_a']/attack_realism_data['shots_a']

#Calcul ratio - AWAY - Ratio tirs vs goals - new as per meeting with Gabriel
attack_realism_data['AWAY - Ratio tirs - goals'] = attack_realism_data['goals_scored_A']/attack_realism_data['shots_a']

#Calcul ratio - AWAY - Ratio goals - tirs cadrés
attack_realism_data['AWAY - Ratio tirs cadrés - goals'] = attack_realism_data['goals_scored_A']/attack_realism_data['target_shots_a']


#Calcul ratio TOTAL - Ratio tirs vs tirs cadrés
attack_realism_data['TOTAL - Ratio tirs vs tirs cadrés'] = attack_realism_data['Total Target Shots']/attack_realism_data['Total Shots']

#Calcul ratio - TOTAL - Ratio goals - tirs cadrés - new as per meeting with Gabriel
attack_realism_data['TOTAL - Ratio tirs - goals'] = attack_realism_data['Total Goals Scored']/attack_realism_data['Total Shots']

#Calcul ratio - TOTAL - Ratio goals - tirs cadrés
attack_realism_data['TOTAL - Ratio tirs cadrés - goals'] = attack_realism_data['Total Goals Scored']/attack_realism_data['Total Target Shots']

#Dataframe Arsenal attack
ars_attack_realism_ratio = pd.DataFrame(attack_realism_data, columns = 
                             ['shots_h','target_shots_h','HOME - Ratio tirs vs tirs cadrés','goals_scored_H',
                              'HOME - Ratio tirs - goals','HOME - Ratio tirs cadrés - goals',
                               'shots_a','target_shots_a','AWAY - Ratio tirs vs tirs cadrés','goals_scored_A',
                               'AWAY - Ratio tirs - goals','AWAY - Ratio tirs cadrés - goals',
                               'Total Shots','Total Target Shots','TOTAL - Ratio tirs vs tirs cadrés',
                               'TOTAL - Ratio tirs - goals','Total Goals Scored','TOTAL - Ratio tirs cadrés - goals'])
#******************************************************************************************************************************************

# Analyse approfondie aspect défensif (même ratios)

#Calcul ratio HOME - Ratio tirs vs tirs cadrés
attack_realism_data['HOME_conced - Ratio tirs vs tirs cadrés'] = attack_realism_data['target_shots_conced_h']/attack_realism_data['shots_conced_h']

#Calcul ratio HOME - Ratio tirs vs goals - new as per meeting with Gabriel
attack_realism_data['HOME_conded - Ratio tirs - goals'] = attack_realism_data['goals_conced_H']/attack_realism_data['shots_conced_h']

#Calcul ratio HOME - Ratio tirs cadrés vs goals
attack_realism_data['HOME_conced - Ratio tirs cadrés - goals'] = attack_realism_data['goals_conced_H']/attack_realism_data['target_shots_conced_h']

#Calcul ratio AWAY - Ratio tirs vs tirs cadrés
attack_realism_data['AWAY_conced - Ratio tirs vs tirs cadrés'] = attack_realism_data['target_shots_conced_a']/attack_realism_data['shots_conced_a']

#Calcul ratio - AWAY - Ratio tirs vs goals - new as per meeting with Gabriel
attack_realism_data['AWAY_conced - Ratio tirs - goals'] = attack_realism_data['goals_conced_A']/attack_realism_data['shots_conced_a']

#Calcul ratio - AWAY - Ratio goals - tirs cadrés
attack_realism_data['AWAY_conced - Ratio tirs cadrés - goals'] = attack_realism_data['goals_conced_A']/attack_realism_data['target_shots_conced_a']


#Calcul ratio TOTAL - Ratio tirs vs tirs cadrés
attack_realism_data['TOTAL_conced - Ratio tirs vs tirs cadrés'] = attack_realism_data['Total Target Shots Conced']/attack_realism_data['Total Shots Conced']

#Calcul ratio - TOTAL - Ratio goals - tirs cadrés - new as per meeting with Gabriel
attack_realism_data['TOTAL_conced - Ratio tirs - goals'] = attack_realism_data['Total Goals conced']/attack_realism_data['Total Shots Conced']

#Calcul ratio - TOTAL - Ratio goals - tirs cadrés
attack_realism_data['TOTAL_conced - Ratio tirs cadrés - goals'] = attack_realism_data['Total Goals conced']/attack_realism_data['Total Target Shots Conced']

#Création df défense
ars_defense_realism_ratio = pd.DataFrame(attack_realism_data, columns = 
                        ['shots_conced_h','target_shots_conced_h','HOME_conced - Ratio tirs vs tirs cadrés','goals_conced_H',
                         'HOME_conded - Ratio tirs - goals','HOME_conced - Ratio tirs cadrés - goals',
                         'shots_conced_a','target_shots_conced_a','AWAY_conced - Ratio tirs vs tirs cadrés','goals_conced_A',
                         'AWAY_conced - Ratio tirs - goals','AWAY_conced - Ratio tirs cadrés - goals',
                         'Total Shots Conced','Total Target Shots Conced','TOTAL_conced - Ratio tirs vs tirs cadrés',
                         'TOTAL_conced - Ratio tirs - goals','Total Goals conced','TOTAL_conced - Ratio tirs cadrés - goals'])             



#Variable stat d'Arsenal à domicile
ars_stat_h=df.loc[df['home_team']=='Arsenal']
ars_stat_h=ars_stat_h.groupby(ars_stat_h['season']).sum()

#Variable stat d'Arsenal à l'exterieur
ars_stat_a=df.loc[df['away_team']=='Arsenal']
ars_stat_a=ars_stat_a.groupby(ars_stat_a['season']).sum()

#Ajout des passes, des fautes concédés, des hors-jeux et de la possession dans la variable ars_attack_realism_ratio
ars_attack_realism_ratio['passes_h']=ars_stat_h['home_passes']
ars_attack_realism_ratio['passes_a']=ars_stat_a['away_passes']
ars_attack_realism_ratio['Total_passes']=ars_attack_realism_ratio['passes_a']+ars_attack_realism_ratio['passes_h']
ars_attack_realism_ratio['average_possession_h']=ars_stat_h['home_possession']/19
ars_attack_realism_ratio['average_possession_a']=ars_stat_a['away_possession']/19
ars_attack_realism_ratio['average_possession']=(ars_attack_realism_ratio['average_possession_a']+ars_attack_realism_ratio['average_possession_h'])/2
ars_attack_realism_ratio['offsides_h']=ars_stat_h['home_offsides']
ars_attack_realism_ratio['offsides_a']=ars_stat_a['away_offsides']
ars_attack_realism_ratio['Total_offsides']=ars_attack_realism_ratio['offsides_h']+ars_attack_realism_ratio['offsides_a']
ars_attack_realism_ratio['home_fouls_conceded']=ars_stat_h['home_fouls_conceded']
ars_attack_realism_ratio['away_fouls_conceded']=ars_stat_a['away_fouls_conceded']
ars_attack_realism_ratio['Total_fouls_conceded']=ars_attack_realism_ratio['home_fouls_conceded']+ars_attack_realism_ratio['away_fouls_conceded']

#Ajout des tacles, des cartons, passes concédées et possession adverse dans la variable ars_defense_realism_ratio
ars_defense_realism_ratio['passes_conced_h']=ars_stat_h['away_passes']
ars_defense_realism_ratio['passes_conced_a']=ars_stat_a['home_passes']
ars_defense_realism_ratio['Total_passes_conced']=ars_defense_realism_ratio['passes_conced_h']+ars_defense_realism_ratio['passes_conced_a']
ars_defense_realism_ratio['home_red_cards']=ars_stat_h['home_red_cards']
ars_defense_realism_ratio['away_red_cards']=ars_stat_a['away_red_cards']
ars_defense_realism_ratio['Total_red_cards']=ars_defense_realism_ratio['home_red_cards']+ars_defense_realism_ratio['away_red_cards']
ars_defense_realism_ratio['home_yellow_cards']=ars_stat_h['home_yellow_cards']
ars_defense_realism_ratio['away_yellow_cards']=ars_stat_a['away_yellow_cards']
ars_defense_realism_ratio['Total_yellow_cards']=ars_defense_realism_ratio['home_yellow_cards']+ars_defense_realism_ratio['away_yellow_cards']
ars_defense_realism_ratio['home_tackles']=ars_stat_h['home_tackles']
ars_defense_realism_ratio['away_tackles']=ars_stat_a['away_tackles']
ars_defense_realism_ratio['Total_tackles']=ars_defense_realism_ratio['home_tackles']+ars_defense_realism_ratio['away_tackles']
ars_defense_realism_ratio['average_possession_conced_h']=ars_stat_h['away_possession']/19
ars_defense_realism_ratio['average_possession_conced_a']=ars_stat_a['home_possession']/19
ars_defense_realism_ratio['average_possession_conced']=(ars_defense_realism_ratio['average_possession_conced_a']+ars_defense_realism_ratio['average_possession_conced_h'])/2
ars_defense_realism_ratio['home_fouls']=ars_stat_h['away_fouls_conceded']
ars_defense_realism_ratio['away_fouls']=ars_stat_a['home_fouls_conceded']
ars_defense_realism_ratio['Total_fouls']=ars_defense_realism_ratio['home_fouls']+ars_defense_realism_ratio['away_fouls'] 

#******************************************************************************************************************************************

#Analyse des "Clean Sheet"

#Création des variables

df_cs=df

df_cs['clean_sheet_home'] = np.nan
for i in range(len(df['home_team'])):
    if df_cs['goal_away_ft'][i]==0:
        df_cs['clean_sheet_home'][i] = 1 
    
        
df_cs['clean_sheet_away'] = np.nan
for i in range(len(df['home_team'])):
    if df_cs['goal_home_ft'][i]==0:
        df_cs['clean_sheet_away'][i] = 1 
        
#****************************************************************************************************************************************** 

#Analyse du résultat à la mi-temps et à la fin du match

HT_FT=df

#Création des variables à la mi-temps

HT_FT['Draw_Home_ht'] = np.nan
for i in range(len(df['home_team'])):
    if (HT_FT['goal_home_ht'][i]==HT_FT['goal_away_ht'][i]):
        HT_FT['Draw_Home_ht'][i] = 1

HT_FT['Draw_Away_ht'] = np.nan
for i in range(len(df['home_team'])):
    if (HT_FT['goal_home_ht'][i]==HT_FT['goal_away_ht'][i]):
        HT_FT['Draw_Away_ht'][i] = 1


HT_FT['Winner_Home_ht'] = np.nan
for i in range(len(df['home_team'])):
    if (HT_FT['goal_home_ht'][i]>HT_FT['goal_away_ht'][i]):
        HT_FT['Winner_Home_ht'][i] = 1
        
HT_FT['Winner_Away_ht'] = np.nan
for i in range(len(df['home_team'])):
    if (HT_FT['goal_home_ht'][i]<HT_FT['goal_away_ht'][i]):
        HT_FT['Winner_Away_ht'][i] = 1
        
HT_FT['Loser_Home_ht'] = np.nan
for i in range(len(df['home_team'])):
    if (HT_FT['goal_home_ht'][i]<HT_FT['goal_away_ht'][i]):
        HT_FT['Loser_Home_ht'][i] = 1

HT_FT['Loser_Away_ht'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['goal_home_ht'][i]>HT_FT['goal_away_ht'][i]):
        HT_FT['Loser_Away_ht'][i] = 1

#Création des variables des combinaisons Victoire/Nul/Défaite vs Victoire/Nul/Défaite

## Home___________________________________________

HT_FT['win_win_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Winner_Home_ht'][i] == 1) & (HT_FT['Winner_Home'][i] == 1): 
        HT_FT['win_win_home'][i] = 1 
          
HT_FT['win_lose_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Winner_Home_ht'][i] == 1) & (HT_FT['Loser_Home'][i] == 1):
        HT_FT['win_lose_home'][i] = 1 
        
HT_FT['win_draw_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Winner_Home_ht'][i] == 1) & (HT_FT['Draw_Home'][i] == 1):
        HT_FT['win_draw_home'][i] = 1 

##

HT_FT['lose_win_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Loser_Home_ht'][i] == 1) & (HT_FT['Winner_Home'][i] == 1):
        HT_FT['lose_win_home'][i] = 1 
        
HT_FT['lose_lose_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Loser_Home_ht'][i] == 1) & (HT_FT['Loser_Home'][i] == 1):
        HT_FT['lose_lose_home'][i] = 1 
        
HT_FT['lose_draw_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Loser_Home_ht'][i] == 1) & (HT_FT['Draw_Home'][i] == 1):
        HT_FT['lose_draw_home'][i] = 1 
##

HT_FT['draw_win_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Draw_Home_ht'][i] == 1) & (HT_FT['Winner_Home'][i] == 1):
        HT_FT['draw_win_home'][i] = 1 
        
HT_FT['draw_lose_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Draw_Home_ht'][i] == 1) & (HT_FT['Loser_Home'][i] == 1):
        HT_FT['draw_lose_home'][i] = 1 
        
HT_FT['draw_draw_home'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Draw_Home_ht'][i] == 1) & (HT_FT['Draw_Home'][i] == 1):
        HT_FT['draw_draw_home'][i] = 1 

## Away__________________________________________

HT_FT['win_win_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Winner_Away_ht'][i] == 1) & (HT_FT['Winner_Away'][i] == 1):
        HT_FT['win_win_away'][i] = 1 


HT_FT['win_lose_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Winner_Away_ht'][i] == 1) & (HT_FT['Loser_Away'][i] == 1):
        HT_FT['win_lose_away'][i] = 1 
        
HT_FT['win_draw_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Winner_Away_ht'][i] == 1) & (HT_FT['Draw_Away'][i] == 1):
        HT_FT['win_draw_away'][i] = 1 

##

HT_FT['lose_win_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Loser_Away_ht'][i] == 1) & (HT_FT['Winner_Away'][i] == 1):
        HT_FT['lose_win_away'][i] = 1 
        
HT_FT['lose_lose_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Loser_Away_ht'][i] == 1) & (HT_FT['Loser_Away'][i] == 1):
        HT_FT['lose_lose_away'][i] = 1 
        
HT_FT['lose_draw_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Loser_Away_ht'][i] == 1) & (HT_FT['Draw_Away'][i] == 1):
        HT_FT['lose_draw_away'][i] = 1 

##

HT_FT['draw_win_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Draw_Away_ht'][i] == 1) & (HT_FT['Winner_Away'][i] == 1):
        HT_FT['draw_win_away'][i] = 1 
        
HT_FT['draw_lose_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Draw_Away_ht'][i] == 1) & (HT_FT['Loser_Away'][i] == 1):
        HT_FT['draw_lose_away'][i] = 1 
        
HT_FT['draw_draw_away'] = np.nan
for i in range(len(HT_FT['home_team'])):
    if (HT_FT['Draw_Away_ht'][i] == 1) & (HT_FT['Draw_Away'][i] == 1):
        HT_FT['draw_draw_away'][i] = 1 

HT_FT = HT_FT.fillna(0)

#Regroupement des résultats par saison

HT_FT = HT_FT.set_index('season')

#Filtre sur Arsenal
HT_FT_Ars = HT_FT.loc[HT_FT['home_team'] == 'Arsenal']

HT_FT_season_grouped_Ars = HT_FT_Ars[['win_win_home', 'draw_win_home', 'lose_win_home',
                              'win_draw_home', 'draw_draw_home', 'lose_draw_home',
                              'win_lose_home', 'draw_lose_home', 'lose_lose_home',
                              'win_win_away', 'draw_win_away', 'lose_win_away',
                              'win_draw_away', 'draw_draw_away', 'lose_draw_away',
                              'win_lose_away', 'draw_lose_away', 'lose_lose_away']].groupby('season').sum()

#******************************************************************************************************************************************

df_WP = df

df_WP['point_home'] = np.nan
for i in range(len(df_WP['home_team'])):
    if df_WP['sg_match_ft'][i]>0:
        df_WP['point_home'][i] = 3
        
    elif df_WP['sg_match_ft'][i]==0:
          df['point_home'][i] = 1
            
    elif df_WP['sg_match_ft'][i]<0:      
          df_WP['point_home'][i] = 0
        
df_WP['point_away'] = np.nan
for i in range(len(df_WP['home_team'])):
    if df_WP['sg_match_ft'][i]>0:
        df_WP['point_away'][i] = 0
        
    elif df_WP['sg_match_ft'][i]==0:
          df['point_away'][i] = 1
            
    elif df_WP['sg_match_ft'][i]<0:      
          df_WP['point_away'][i] = 3
            
df_WP['month'] = (pd.to_datetime(df.date)).dt.month
df_WP['Month_chrono'] = df_WP['month']
df_WP['Month_chrono'] = df_WP['Month_chrono'].replace({7:1, 8:2,9:3,10:4,11:5,12:6,1:7,2:8,3:9,4:10,5:11,6:12})
#df_WP['month2'] = df_WP['month2'].replace({1:'Juillet', 2:'Aout', 3:'Septembre', 4:'Octobre', 5:'Novembre', 6:'Décembre', 7:'Janvier', 8:'Février', 9:'Mars', 10:'Avril', 11:'Mai', 12:'Juin' })

df_WP_grouped_home =df_WP.groupby(['home_team', 'season', 'Month_chrono']).sum()['point_home'].reset_index()
df_WP_grouped_away =df_WP.groupby(['away_team', 'season', 'Month_chrono']).sum()['point_away'].reset_index()

df_WP_grouped_home_team = df_WP_grouped_home.loc[df_WP_grouped_home['home_team']=='Arsenal']
df_WP_grouped_away_team = df_WP_grouped_away.loc[df_WP_grouped_away['away_team']=='Arsenal']

#******************************************************************************************************************************************

#LA SUITE SE FERA AVEVC LES MEMES NOMS DES VARIABLES POUR GAGNER DU TEMPS 
#LA MODIFCATION SE FAIT DANS LA DERNIERE LIGNES DE LA CELLULE 
#LA CELLULE DE ARSENAL SERA REPETER UNE NOUVELLE FOIS A LA FIN 

games_ars_big5=df.loc[(df["home_team"]=="Arsenal")&(df["away_team"]=="Manchester City")
                        |(df["home_team"]=="Manchester City")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Manchester United")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Chelsea")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Arsenal")]
games_ars_big5=games_ars_big5[['season','home_team','away_team','result_full','Loser_Away','Loser_Home','Winner_Away','Winner_Home','Draw_Home','Draw_Away']]

games_ars_big5=games_ars_big5.fillna(0)

#Victoire à domicile d'Arsenal contre le Big 5 
ars_wins_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Arsenal']
ars_wins_home_big5=ars_wins_home_big5.loc[ars_wins_home_big5['Winner_Home']==1]
ars_wins_home_big5=ars_wins_home_big5.groupby(ars_wins_home_big5['season']).sum()

#Victoire à l'exterieur d'Arsenal contre le Big 5
ars_wins_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Arsenal']
ars_wins_away_big5=ars_wins_away_big5.loc[ars_wins_away_big5['Winner_Away']==1]
ars_wins_away_big5=ars_wins_away_big5.groupby(ars_wins_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5=ars_wins_home_big5[['Winner_Home']]
recap_ars_big5['Winner_Away']=ars_wins_away_big5['Winner_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total wins']=recap_ars_big5['Winner_Home']+recap_ars_big5['Winner_Away']

#Défaite à domicile d'Arsenal contre le Big 5 
ars_losses_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Arsenal']
ars_losses_home_big5=ars_losses_home_big5.loc[ars_losses_home_big5['Loser_Home']==1]
ars_losses_home_big5=ars_losses_home_big5.groupby(ars_losses_home_big5['season']).sum()

#Défaite à l'exterieur d'Arsenal contre le Big 5
ars_losses_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Arsenal']
ars_losses_away_big5=ars_losses_away_big5.loc[ars_losses_away_big5['Loser_Away']==1]
ars_losses_away_big5=ars_losses_away_big5.groupby(ars_losses_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Loser_Home']=ars_losses_home_big5['Loser_Home']
recap_ars_big5['Loser_Away']=ars_losses_away_big5['Loser_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total losses']=recap_ars_big5['Loser_Home']+recap_ars_big5['Loser_Away']

#Matchs nuls à domicile d'Arsenal contre le Big 5 
ars_draws_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Arsenal']
ars_draws_home_big5=ars_draws_home_big5.loc[ars_draws_home_big5['Draw_Home']==1]
ars_draws_home_big5=ars_draws_home_big5.groupby(ars_draws_home_big5['season']).sum()

#Matchs nuls à l'exterieur d'Arsenal contre le Big 5
ars_draws_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Arsenal']
ars_draws_away_big5=ars_draws_away_big5.loc[ars_draws_away_big5['Draw_Away']==1]
ars_draws_away_big5=ars_draws_away_big5.groupby(ars_draws_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Draw_Home']=ars_draws_home_big5['Draw_Home']
recap_ars_big5['Draw_Away']=ars_draws_away_big5['Draw_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total draws']=recap_ars_big5['Draw_Home']+recap_ars_big5['Draw_Away']
recap_ars_big5['Points gagnés']=(recap_ars_big5['total wins']*3)+recap_ars_big5['total draws']
recap_ars_big5['Classement']=classement_arsenal['Classement']

#Récap des confrontations big 5 - Chelsea
classement_chelsea=classement_all_season.loc[classement_all_season['Equipes']=='Chelsea']
classement_chelsea['Points']=classement_chelsea['Points'].astype(int)
classement_chelsea.set_index('season', inplace = True)

games_ars_big5=df.loc[(df["home_team"]=="Chelsea")&(df["away_team"]=="Manchester City")
                        |(df["home_team"]=="Manchester City")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Chelsea")&(df["away_team"]=="Manchester United")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Chelsea")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Chelsea")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Chelsea")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Chelsea")]
games_ars_big5=games_ars_big5[['season','home_team','away_team','result_full','Loser_Away','Loser_Home','Winner_Away','Winner_Home','Draw_Home','Draw_Away']]

games_ars_big5=games_ars_big5.fillna(0)

#Victoire à domicile d'Arsenal contre le Big 5 
ars_wins_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Chelsea']
ars_wins_home_big5=ars_wins_home_big5.loc[ars_wins_home_big5['Winner_Home']==1]
ars_wins_home_big5=ars_wins_home_big5.groupby(ars_wins_home_big5['season']).sum()

#Victoire à l'exterieur d'Arsenal contre le Big 5
ars_wins_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Chelsea']
ars_wins_away_big5=ars_wins_away_big5.loc[ars_wins_away_big5['Winner_Away']==1]
ars_wins_away_big5=ars_wins_away_big5.groupby(ars_wins_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5=ars_wins_home_big5[['Winner_Home']]
recap_ars_big5['Winner_Away']=ars_wins_away_big5['Winner_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total wins']=recap_ars_big5['Winner_Home']+recap_ars_big5['Winner_Away']


#Défaite à domicile d'Arsenal contre le Big 5 
ars_losses_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Chelsea']
ars_losses_home_big5=ars_losses_home_big5.loc[ars_losses_home_big5['Loser_Home']==1]
ars_losses_home_big5=ars_losses_home_big5.groupby(ars_losses_home_big5['season']).sum()

#Défaite à l'exterieur d'Arsenal contre le Big 5
ars_losses_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Chelsea']
ars_losses_away_big5=ars_losses_away_big5.loc[ars_losses_away_big5['Loser_Away']==1]
ars_losses_away_big5=ars_losses_away_big5.groupby(ars_losses_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Loser_Home']=ars_losses_home_big5['Loser_Home']
recap_ars_big5['Loser_Away']=ars_losses_away_big5['Loser_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total losses']=recap_ars_big5['Loser_Home']+recap_ars_big5['Loser_Away']

#Matchs nuls à domicile d'Arsenal contre le Big 5 
ars_draws_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Chelsea']
ars_draws_home_big5=ars_draws_home_big5.loc[ars_draws_home_big5['Draw_Home']==1]
ars_draws_home_big5=ars_draws_home_big5.groupby(ars_draws_home_big5['season']).sum()

#Matchs nuls à l'exterieur d'Arsenal contre le Big 5
ars_draws_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Chelsea']
ars_draws_away_big5=ars_draws_away_big5.loc[ars_draws_away_big5['Draw_Away']==1]
ars_draws_away_big5=ars_draws_away_big5.groupby(ars_draws_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Draw_Home']=ars_draws_home_big5['Draw_Home']
recap_ars_big5['Draw_Away']=ars_draws_away_big5['Draw_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total draws']=recap_ars_big5['Draw_Home']+recap_ars_big5['Draw_Away']
recap_ars_big5['Points gagnés']=(recap_ars_big5['total wins']*3)+recap_ars_big5['total draws']
recap_chelsea_big5 =pd.concat([recap_ars_big5])
recap_chelsea_big5['Classement']=classement_chelsea['Classement']
recap_chelsea_big5["Equipe"]=np.nan
recap_chelsea_big5["Equipe"].fillna("Chelsea",inplace=True)

#Récap des confrontations big5- Manchester City
classement_mancity=classement_all_season.loc[classement_all_season['Equipes']=='Man City']
classement_mancity['Points']=classement_mancity['Points'].astype(int)
classement_mancity.set_index('season', inplace = True)

games_ars_big5=df.loc[(df["home_team"]=="Chelsea")&(df["away_team"]=="Manchester City")
                        |(df["home_team"]=="Manchester City")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Manchester City")&(df["away_team"]=="Manchester United")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Manchester City")
                     |(df["home_team"]=="Manchester City")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Manchester City")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Manchester City")
                     |(df["home_team"]=="Manchester City")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Manchester City")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Manchester City")]
games_ars_big5=games_ars_big5[['season','home_team','away_team','result_full','Loser_Away','Loser_Home','Winner_Away','Winner_Home','Draw_Home','Draw_Away']]

games_ars_big5=games_ars_big5.fillna(0)

#Victoire à domicile d'Arsenal contre le Big 5 
ars_wins_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Manchester City']
ars_wins_home_big5=ars_wins_home_big5.loc[ars_wins_home_big5['Winner_Home']==1]
ars_wins_home_big5=ars_wins_home_big5.groupby(ars_wins_home_big5['season']).sum()

#Victoire à l'exterieur d'Arsenal contre le Big 5
ars_wins_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Manchester City']
ars_wins_away_big5=ars_wins_away_big5.loc[ars_wins_away_big5['Winner_Away']==1]
ars_wins_away_big5=ars_wins_away_big5.groupby(ars_wins_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5=ars_wins_home_big5[['Winner_Home']]
recap_ars_big5['Winner_Away']=ars_wins_away_big5['Winner_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total wins']=recap_ars_big5['Winner_Home']+recap_ars_big5['Winner_Away']


#Défaite à domicile d'Arsenal contre le Big 5 
ars_losses_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Manchester City']
ars_losses_home_big5=ars_losses_home_big5.loc[ars_losses_home_big5['Loser_Home']==1]
ars_losses_home_big5=ars_losses_home_big5.groupby(ars_losses_home_big5['season']).sum()

#Défaite à l'exterieur d'Arsenal contre le Big 5
ars_losses_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Manchester City']
ars_losses_away_big5=ars_losses_away_big5.loc[ars_losses_away_big5['Loser_Away']==1]
ars_losses_away_big5=ars_losses_away_big5.groupby(ars_losses_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Loser_Home']=ars_losses_home_big5['Loser_Home']
recap_ars_big5['Loser_Away']=ars_losses_away_big5['Loser_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total losses']=recap_ars_big5['Loser_Home']+recap_ars_big5['Loser_Away']

#Matchs nuls à domicile d'Arsenal contre le Big 5 
ars_draws_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Manchester City']
ars_draws_home_big5=ars_draws_home_big5.loc[ars_draws_home_big5['Draw_Home']==1]
ars_draws_home_big5=ars_draws_home_big5.groupby(ars_draws_home_big5['season']).sum()

#Matchs nuls à l'exterieur d'Arsenal contre le Big 5
ars_draws_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Manchester City']
ars_draws_away_big5=ars_draws_away_big5.loc[ars_draws_away_big5['Draw_Away']==1]
ars_draws_away_big5=ars_draws_away_big5.groupby(ars_draws_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Draw_Home']=ars_draws_home_big5['Draw_Home']
recap_ars_big5['Draw_Away']=ars_draws_away_big5['Draw_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total draws']=recap_ars_big5['Draw_Home']+recap_ars_big5['Draw_Away']
recap_ars_big5['Points gagnés']=(recap_ars_big5['total wins']*3)+recap_ars_big5['total draws']
recap_mancity_big5 =pd.concat([recap_ars_big5])
recap_mancity_big5['Classement']=classement_mancity['Classement']
recap_mancity_big5["Equipe"]=np.nan
recap_mancity_big5["Equipe"].fillna("Man City",inplace=True)

#Récap des confrontations big5- Manchester United
classement_manu=classement_all_season.loc[classement_all_season['Equipes']=='Man United']
classement_manu['Points']=classement_manu['Points'].astype(int)
classement_manu.set_index('season', inplace = True)

games_ars_big5=df.loc[(df["home_team"]=="Chelsea")&(df["away_team"]=="Manchester United")
                        |(df["home_team"]=="Manchester United")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Manchester City")&(df["away_team"]=="Manchester United")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Manchester City")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Manchester United")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Manchester United")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Manchester United")]
games_ars_big5=games_ars_big5[['season','home_team','away_team','result_full','Loser_Away','Loser_Home','Winner_Away','Winner_Home','Draw_Home','Draw_Away']]

games_ars_big5=games_ars_big5.fillna(0)

#Victoire à domicile d'Arsenal contre le Big 5 
ars_wins_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Manchester United']
ars_wins_home_big5=ars_wins_home_big5.loc[ars_wins_home_big5['Winner_Home']==1]
ars_wins_home_big5=ars_wins_home_big5.groupby(ars_wins_home_big5['season']).sum()

#Victoire à l'exterieur d'Arsenal contre le Big 5
ars_wins_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Manchester United']
ars_wins_away_big5=ars_wins_away_big5.loc[ars_wins_away_big5['Winner_Away']==1]
ars_wins_away_big5=ars_wins_away_big5.groupby(ars_wins_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5=ars_wins_home_big5[['Winner_Home']]
recap_ars_big5['Winner_Away']=ars_wins_away_big5['Winner_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total wins']=recap_ars_big5['Winner_Home']+recap_ars_big5['Winner_Away']


#Défaite à domicile d'Arsenal contre le Big 5 
ars_losses_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Manchester United']
ars_losses_home_big5=ars_losses_home_big5.loc[ars_losses_home_big5['Loser_Home']==1]
ars_losses_home_big5=ars_losses_home_big5.groupby(ars_losses_home_big5['season']).sum()

#Défaite à l'exterieur d'Arsenal contre le Big 5
ars_losses_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Manchester United']
ars_losses_away_big5=ars_losses_away_big5.loc[ars_losses_away_big5['Loser_Away']==1]
ars_losses_away_big5=ars_losses_away_big5.groupby(ars_losses_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Loser_Home']=ars_losses_home_big5['Loser_Home']
recap_ars_big5['Loser_Away']=ars_losses_away_big5['Loser_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total losses']=recap_ars_big5['Loser_Home']+recap_ars_big5['Loser_Away']

#Matchs nuls à domicile d'Arsenal contre le Big 5 
ars_draws_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Manchester United']
ars_draws_home_big5=ars_draws_home_big5.loc[ars_draws_home_big5['Draw_Home']==1]
ars_draws_home_big5=ars_draws_home_big5.groupby(ars_draws_home_big5['season']).sum()

#Matchs nuls à l'exterieur d'Arsenal contre le Big 5
ars_draws_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Manchester United']
ars_draws_away_big5=ars_draws_away_big5.loc[ars_draws_away_big5['Draw_Away']==1]
ars_draws_away_big5=ars_draws_away_big5.groupby(ars_draws_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Draw_Home']=ars_draws_home_big5['Draw_Home']
recap_ars_big5['Draw_Away']=ars_draws_away_big5['Draw_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total draws']=recap_ars_big5['Draw_Home']+recap_ars_big5['Draw_Away']
recap_ars_big5['Points gagnés']=(recap_ars_big5['total wins']*3)+recap_ars_big5['total draws']
recap_manu_big5 =pd.concat([recap_ars_big5])
recap_manu_big5['Classement']=classement_manu['Classement']
recap_manu_big5["Equipe"]=np.nan
recap_manu_big5["Equipe"].fillna("Man United",inplace=True)

#Récap des confrontations big5- Liverpool
classement_liv=classement_all_season.loc[classement_all_season['Equipes']=='Liverpool']
classement_liv['Points']=classement_liv['Points'].astype(int)
classement_liv.set_index('season', inplace = True)

games_ars_big5=df.loc[(df["home_team"]=="Chelsea")&(df["away_team"]=="Liverpool")
                        |(df["home_team"]=="Liverpool")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Manchester City")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Manchester City")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Manchester United")]
games_ars_big5=games_ars_big5[['season','home_team','away_team','result_full','Loser_Away','Loser_Home','Winner_Away','Winner_Home','Draw_Home','Draw_Away']]

games_ars_big5=games_ars_big5.fillna(0)

#Victoire à domicile d'Arsenal contre le Big 5 
ars_wins_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Liverpool']
ars_wins_home_big5=ars_wins_home_big5.loc[ars_wins_home_big5['Winner_Home']==1]
ars_wins_home_big5=ars_wins_home_big5.groupby(ars_wins_home_big5['season']).sum()

#Victoire à l'exterieur d'Arsenal contre le Big 5
ars_wins_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Liverpool']
ars_wins_away_big5=ars_wins_away_big5.loc[ars_wins_away_big5['Winner_Away']==1]
ars_wins_away_big5=ars_wins_away_big5.groupby(ars_wins_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5=ars_wins_home_big5[['Winner_Home']]
recap_ars_big5['Winner_Away']=ars_wins_away_big5['Winner_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total wins']=recap_ars_big5['Winner_Home']+recap_ars_big5['Winner_Away']


#Défaite à domicile d'Arsenal contre le Big 5 
ars_losses_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Liverpool']
ars_losses_home_big5=ars_losses_home_big5.loc[ars_losses_home_big5['Loser_Home']==1]
ars_losses_home_big5=ars_losses_home_big5.groupby(ars_losses_home_big5['season']).sum()

#Défaite à l'exterieur d'Arsenal contre le Big 5
ars_losses_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Liverpool']
ars_losses_away_big5=ars_losses_away_big5.loc[ars_losses_away_big5['Loser_Away']==1]
ars_losses_away_big5=ars_losses_away_big5.groupby(ars_losses_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Loser_Home']=ars_losses_home_big5['Loser_Home']
recap_ars_big5['Loser_Away']=ars_losses_away_big5['Loser_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total losses']=recap_ars_big5['Loser_Home']+recap_ars_big5['Loser_Away']

#Matchs nuls à domicile d'Arsenal contre le Big 5 
ars_draws_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Liverpool']
ars_draws_home_big5=ars_draws_home_big5.loc[ars_draws_home_big5['Draw_Home']==1]
ars_draws_home_big5=ars_draws_home_big5.groupby(ars_draws_home_big5['season']).sum()

#Matchs nuls à l'exterieur d'Arsenal contre le Big 5
ars_draws_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Liverpool']
ars_draws_away_big5=ars_draws_away_big5.loc[ars_draws_away_big5['Draw_Away']==1]
ars_draws_away_big5=ars_draws_away_big5.groupby(ars_draws_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Draw_Home']=ars_draws_home_big5['Draw_Home']
recap_ars_big5['Draw_Away']=ars_draws_away_big5['Draw_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total draws']=recap_ars_big5['Draw_Home']+recap_ars_big5['Draw_Away']
recap_ars_big5['Points gagnés']=(recap_ars_big5['total wins']*3)+recap_ars_big5['total draws']
recap_liv_big5 =pd.concat([recap_ars_big5])
recap_liv_big5['Classement']=classement_liv['Classement']
recap_liv_big5["Equipe"]=np.nan
recap_liv_big5["Equipe"].fillna("Liverpool",inplace=True)

#Récap des confrontations big5- Tottenham
classement_tot=classement_all_season.loc[classement_all_season['Equipes']=='Tottenham']
classement_tot['Points']=classement_tot['Points'].astype(int)
classement_tot.set_index('season', inplace = True)

games_ars_big5=df.loc[(df["home_team"]=="Chelsea")&(df["away_team"]=="Tottenham Hotspur")
                        |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Manchester City")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Manchester City")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Manchester United")]
games_ars_big5=games_ars_big5[['season','home_team','away_team','result_full','Loser_Away','Loser_Home','Winner_Away','Winner_Home','Draw_Home','Draw_Away']]

games_ars_big5=games_ars_big5.fillna(0)

#Victoire à domicile d'Arsenal contre le Big 5 
ars_wins_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Tottenham Hotspur']
ars_wins_home_big5=ars_wins_home_big5.loc[ars_wins_home_big5['Winner_Home']==1]
ars_wins_home_big5=ars_wins_home_big5.groupby(ars_wins_home_big5['season']).sum()

#Victoire à l'exterieur d'Arsenal contre le Big 5
ars_wins_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Tottenham Hotspur']
ars_wins_away_big5=ars_wins_away_big5.loc[ars_wins_away_big5['Winner_Away']==1]
ars_wins_away_big5=ars_wins_away_big5.groupby(ars_wins_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5=ars_wins_home_big5[['Winner_Home']]
recap_ars_big5['Winner_Away']=ars_wins_away_big5['Winner_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total wins']=recap_ars_big5['Winner_Home']+recap_ars_big5['Winner_Away']


#Défaite à domicile d'Arsenal contre le Big 5 
ars_losses_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Tottenham Hotspur']
ars_losses_home_big5=ars_losses_home_big5.loc[ars_losses_home_big5['Loser_Home']==1]
ars_losses_home_big5=ars_losses_home_big5.groupby(ars_losses_home_big5['season']).sum()

#Défaite à l'exterieur d'Arsenal contre le Big 5
ars_losses_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Tottenham Hotspur']
ars_losses_away_big5=ars_losses_away_big5.loc[ars_losses_away_big5['Loser_Away']==1]
ars_losses_away_big5=ars_losses_away_big5.groupby(ars_losses_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Loser_Home']=ars_losses_home_big5['Loser_Home']
recap_ars_big5['Loser_Away']=ars_losses_away_big5['Loser_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total losses']=recap_ars_big5['Loser_Home']+recap_ars_big5['Loser_Away']

#Matchs nuls à domicile d'Arsenal contre le Big 5 
ars_draws_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Tottenham Hotspur']
ars_draws_home_big5=ars_draws_home_big5.loc[ars_draws_home_big5['Draw_Home']==1]
ars_draws_home_big5=ars_draws_home_big5.groupby(ars_draws_home_big5['season']).sum()

#Matchs nuls à l'exterieur d'Arsenal contre le Big 5
ars_draws_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Tottenham Hotspur']
ars_draws_away_big5=ars_draws_away_big5.loc[ars_draws_away_big5['Draw_Away']==1]
ars_draws_away_big5=ars_draws_away_big5.groupby(ars_draws_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Draw_Home']=ars_draws_home_big5['Draw_Home']
recap_ars_big5['Draw_Away']=ars_draws_away_big5['Draw_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total draws']=recap_ars_big5['Draw_Home']+recap_ars_big5['Draw_Away']
recap_ars_big5['Points gagnés']=(recap_ars_big5['total wins']*3)+recap_ars_big5['total draws']
recap_tot_big5 =pd.concat([recap_ars_big5])
recap_tot_big5['Classement']=classement_tot['Classement']
recap_tot_big5["Equipe"]=np.nan
recap_tot_big5["Equipe"].fillna("Tottenham",inplace=True)

#Récap des confrontations big 5 - Arsanle
#REMETTRE EN A JOUR LES VARIABLES D'ARSENAL

classement_arsenal=classement_all_season.loc[classement_all_season['Equipes']=='Arsenal']
classement_arsenal['Points']=classement_arsenal['Points'].astype(int)
classement_arsenal.set_index('season', inplace = True)

games_ars_big5=df.loc[(df["home_team"]=="Arsenal")&(df["away_team"]=="Manchester City")
                        |(df["home_team"]=="Manchester City")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Manchester United")
                     |(df["home_team"]=="Manchester United")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Tottenham Hotspur")
                     |(df["home_team"]=="Tottenham Hotspur")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Chelsea")
                     |(df["home_team"]=="Chelsea")&(df["away_team"]=="Arsenal")
                     |(df["home_team"]=="Arsenal")&(df["away_team"]=="Liverpool")
                     |(df["home_team"]=="Liverpool")&(df["away_team"]=="Arsenal")]
games_ars_big5=games_ars_big5[['season','home_team','away_team','result_full','Loser_Away','Loser_Home','Winner_Away','Winner_Home','Draw_Home','Draw_Away']]

games_ars_big5=games_ars_big5.fillna(0)

#Victoire à domicile d'Arsenal contre le Big 5 
ars_wins_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Arsenal']
ars_wins_home_big5=ars_wins_home_big5.loc[ars_wins_home_big5['Winner_Home']==1]
ars_wins_home_big5=ars_wins_home_big5.groupby(ars_wins_home_big5['season']).sum()

#Victoire à l'exterieur d'Arsenal contre le Big 5
ars_wins_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Arsenal']
ars_wins_away_big5=ars_wins_away_big5.loc[ars_wins_away_big5['Winner_Away']==1]
ars_wins_away_big5=ars_wins_away_big5.groupby(ars_wins_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5=ars_wins_home_big5[['Winner_Home']]
recap_ars_big5['Winner_Away']=ars_wins_away_big5['Winner_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total wins']=recap_ars_big5['Winner_Home']+recap_ars_big5['Winner_Away']

#Défaite à domicile d'Arsenal contre le Big 5 
ars_losses_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Arsenal']
ars_losses_home_big5=ars_losses_home_big5.loc[ars_losses_home_big5['Loser_Home']==1]
ars_losses_home_big5=ars_losses_home_big5.groupby(ars_losses_home_big5['season']).sum()

#Défaite à l'exterieur d'Arsenal contre le Big 5
ars_losses_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Arsenal']
ars_losses_away_big5=ars_losses_away_big5.loc[ars_losses_away_big5['Loser_Away']==1]
ars_losses_away_big5=ars_losses_away_big5.groupby(ars_losses_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Loser_Home']=ars_losses_home_big5['Loser_Home']
recap_ars_big5['Loser_Away']=ars_losses_away_big5['Loser_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total losses']=recap_ars_big5['Loser_Home']+recap_ars_big5['Loser_Away']

#Matchs nuls à domicile d'Arsenal contre le Big 5 
ars_draws_home_big5=games_ars_big5.loc[games_ars_big5['home_team']=='Arsenal']
ars_draws_home_big5=ars_draws_home_big5.loc[ars_draws_home_big5['Draw_Home']==1]
ars_draws_home_big5=ars_draws_home_big5.groupby(ars_draws_home_big5['season']).sum()

#Matchs nuls à l'exterieur d'Arsenal contre le Big 5
ars_draws_away_big5=games_ars_big5.loc[games_ars_big5['away_team']=='Arsenal']
ars_draws_away_big5=ars_draws_away_big5.loc[ars_draws_away_big5['Draw_Away']==1]
ars_draws_away_big5=ars_draws_away_big5.groupby(ars_draws_away_big5['season']).sum()

#Variable qui récapitule les résultats
recap_ars_big5['Draw_Home']=ars_draws_home_big5['Draw_Home']
recap_ars_big5['Draw_Away']=ars_draws_away_big5['Draw_Away']
recap_ars_big5=recap_ars_big5.fillna(0)
recap_ars_big5['total draws']=recap_ars_big5['Draw_Home']+recap_ars_big5['Draw_Away']
recap_ars_big5['Points gagnés']=(recap_ars_big5['total wins']*3)+recap_ars_big5['total draws']
recap_ars_big5['Classement']=classement_arsenal['Classement']
recap_ars_big5["Equipe"]=np.nan
recap_ars_big5["Equipe"].fillna("Arsenal",inplace=True)

recap_big_5=pd.concat([recap_ars_big5,recap_tot_big5,recap_chelsea_big5,recap_manu_big5,recap_mancity_big5,recap_liv_big5])

#******************************************************************************************************************************************


#Création Datafram des statistiques des joueurs d'Arsenal de la saison 19997/1998 à 2003/2004
stat_joueur_ars97_04= pd.DataFrame({'Meilleur buteur': [16,17,17,17,24,24,30],'Meilleur passeur' : [11,13,9,9,15,20,8],'Gardien' : [33,17,43,38,36,42,28]},index=['97/98','98/99','99/00','00/01','01/02','02/03','03/04'])

#Création Dataframe des statistiques des joueurs d'Arsenal et des meilleurs joueurs de la saison 2010/2011 à 2019/2020
stat_joueur_10_20=pd.DataFrame({'Meilleur buteur Arsenal': [18,30,14,16,16,16,24,14,22,22],
                                'Meilleur buteur de la saison': [21,30,26,31,26,25,29,32,22,23],
                                'Meilleur passeur Arsenal' : [14,11,12,9,12,19,10,8,8,5],
                                'Meilleur passeur de la saison' : [17,17,13,14,19,19,18,16,15,20],
                                'Gardien Arsenal' : [43,49,37,41,36,36,44,51,51,48],
                               'Meilleur gardien de la saison' : [33,29,34,27,32,35,26,27,22,33]},
                               index=['10/11','11/12','12/13','13/14','14/15','15/16','16/17','17/18','18/19','19/20'])


#******************************************************************************************************************************************


#Classement des propriétaires
classification_owners = (1,2,3,4,5,6,7,7,9,10,11,12,13,14,15,16,17,18,19,20)

clubs_owned = ('Manchester City', 'Chelsea','Arsenal','Fulham','Aston Villa','Wolves','Tottenham','Manchester United','Crystal Palace',
               'Southampton','Leicester City','Newcastle','Liverpool','West Bromwich Albion','Everton','West Ham',
               'Brighton','Leeds United','Sheffield United','Burnley')

clubs_owners =('Sheikh Mansour - Man City','Roman Abramovich - Chelsea','Stan Kroenke - Arsenal',
               'Shahid Khan','Nassef Sawiris','Guo Guangchang','Joe Lewis - Tottenham',
               'The Glazers - Man Utd','Joshua Harris','Gao Jisheng',
               'Aiyawatt Srivaddhanaprabha','Mike Ashley','John Henry - Liverpool','Lai Guochuan',
               'Farhad Moshiri','David Sullivan and David Gold','Tony Bloom','Andrea Radrizzani'
               'Prince Abdullah bin Musa ed','Mike Garlick')

owners_worth = (23300,8500,6800,5800,5500,5000,3600,3600,3100,3000,2660,2350,2200,1900,1560,1300,450,198,62)


df_owners_summary = pd.DataFrame(list(zip(classification_owners, clubs_owned,clubs_owners, owners_worth)),
               columns =['Ranking', 'Clubs','Owners', 'Worth'])

#Total des investissements direct des propriétaires dans leur clubs

#https://www.themag.co.uk/2021/05/premier-league-clubs-how-much-each-owner-has-financed-their-club-2010-2020-newcastle-united/

finance_owners=pd.DataFrame({'Clubs': ['Manchester City','Chelsea','Aston Villa','Everton','Brighton','Leicester City','Manchester United','Wolves','Liverpool','Leeds United','West Ham','Crystal Palace','Southampton','Sheffield United','West Bromwich Albion','Arsenal','Tottenham','Norwich City','Burnley','Newcaslte'],
                               'Owner Financing 2010 to 2020(in £ millions)' : [1133,570,459,348,325,312,297,131,131,85,80,60,58,44,22,15,15,2,2,0]})

#Merge des deux dataframe
ownership_analysis = pd.merge(df_owners_summary, finance_owners, on="Clubs")

ownership_analysis['Direct Financing / Total Worth ratio'] = ownership_analysis['Owner Financing 2010 to 2020(in £ millions)']/ownership_analysis['Worth']


#ownership_analysis of Big 5

big5_ownership = ownership_analysis.loc[(ownership_analysis['Clubs'] == 'Manchester City')|(ownership_analysis['Clubs'] == 'Manchester United')| (ownership_analysis['Clubs'] == 'Chelsea')|(ownership_analysis['Clubs'] == 'Liverpool') |(ownership_analysis['Clubs'] == 'Tottenham') |(ownership_analysis['Clubs'] == 'Arsenal')]

#préparation données pour Lollipop

big5_ownership.sort_values('Worth', inplace=True)
big5_ownership.reset_index(inplace=True)
big5_ownership = big5_ownership.drop(['index', 'Ranking'], axis=1)

#ownership_analysis of Big 5

big5_ownership_wet = ownership_analysis.loc[(ownership_analysis['Clubs'] == 'Manchester City')|(ownership_analysis['Clubs'] == 'Manchester United')| (ownership_analysis['Clubs'] == 'Chelsea')|(ownership_analysis['Clubs'] == 'Liverpool')|(ownership_analysis['Clubs'] == 'Tottenham') |(ownership_analysis['Clubs'] == 'Arsenal')]

#préparation données pour Lollipop

big5_ownership_wet.sort_values('Direct Financing / Total Worth ratio', inplace=True)
big5_ownership_wet.reset_index(inplace=True)

big5_ownership_wet = big5_ownership_wet.drop(['index', 'Ranking'], axis=1)


#******************************************************************************************************************************************



#_____________________________________________________________________________________________________________________________________
#111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

st.sidebar.title("Le storytelling d'Arsenal")

sommaire = st.sidebar.radio('Menu',['La « Premier League »','Exploration du jeu de données','Zoom sur Arsenal', "Que s'est-il passé à Arsenal ?"])  


if sommaire == 'La « Premier League »':
        
    st.image("Premier_League_Logo.svg.png", width=350)
    
    st.title('La « Premier League »')
          
    st.write("La « Premier League » est le championnat de football anglais. Elle est composée de 20 équipes et la saison se déroule d'août à mai sur 38 journées.")
    
    st.write("Ce championnat est l’un des plus prestigieux au monde et le plus populaire en termes de téléspectateurs. Il est réputé pour être l’un des plus exigeants physiquement et chaque match met en évidence l’engagement physique traditionnel du football anglais.")
    
    
    st.image("big6.jpg", width=350)
    
    st.title('Le « Big Six»')
    
    st.write("La renommée de ce championnat vient également de la présence du **Big Six** qui sont les équipes les plus performantes de la division: Manchester United, Liverpool, Arsenal, Chelsea, Manchester City et Tottenham.")

#_______________________________________________________________________________________________________________________________________
#22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222

if sommaire == 'Exploration du jeu de données':

    st.title("Exploration du jeu de données")
    st.write("Le jeu de données est composé de toutes les rencontres de Premier League de la saison 2010/2011 à 2019/2020. Nous retrouvons les statistiques du match et le résultat de la rencontre. ")

    st.write("Pour chaque équipe de Premier League, que vous souhaitez, nous vous proposons de regarder les résultats obtenus et le classement par saison, mais également, l’évolution des différentes statistiques au fil des années. ")

    from urllib.request import urlopen
    from bs4 import BeautifulSoup

    dp = pd.read_csv('df_full_premierleague.csv', sep = ',')
    dp=dp.iloc[:, 2:38]

    dp['home_wins'] = np.nan
    for i in range(len(dp['home_team'])):
        if (dp['goal_home_ft'][i]>dp['goal_away_ft'][i]):
            dp['home_wins'][i] = 1
    dp['away_wins'] = np.nan
    for i in range(len(dp['home_team'])):
        if (dp['goal_home_ft'][i]<dp['goal_away_ft'][i]):
            dp['away_wins'][i] = 1
    dp['home_losses'] = np.nan
    for i in range(len(dp['home_team'])):
        if (dp['goal_home_ft'][i]<dp['goal_away_ft'][i]):
            dp['away_wins'][i] = 1       
    dp['away_losses'] = np.nan
    for i in range(len(dp['home_team'])):
        if (dp['goal_home_ft'][i]>dp['goal_away_ft'][i]):
            dp['away_losses'][i] = 1
    dp['home_draws'] = np.nan
    for i in range(len(dp['home_team'])):
        if (dp['goal_home_ft'][i]==dp['goal_away_ft'][i]):
            dp['home_draws'][i] = 1
    dp['away_draws'] = np.nan
    for i in range(len(dp['home_team'])):
        if (dp['goal_home_ft'][i]==dp['goal_away_ft'][i]):
            dp['away_draws'][i] = 1




    team=st.selectbox(label="Selectionner l'équipe:",options=['AFC Bournemouth', 'Arsenal', 'Aston Villa', 'Birmingham City', 'Blackburn Rovers', 'Blackpool', 'Bolton Wanderers', 'Brighton and Hove Albion', 'Burnley', 'Cardiff City', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield Town', 'Hull City', 'Leeds United', 'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United', 'Middlesbrough', 'Newcastle United', 'Norwich City', 'Queens Park Rangers', 'Reading', 'Sheffield United', 'Southampton', 'Stoke City', 'Sunderland', 'Swansea City', 'Tottenham Hotspur', 'Watford', 'West Bromwich Albion', 'West Ham United', 'Wigan Athletic', 'Wolverhampton Wanderers'])

    def team_choisi(team):
        if team ==team :
            return team
    #----------------------------------------------------------------------------------------------
    #saisons 2010/2011
    dp_10_11=dp.loc[dp['season']=='10/11']
    dp_10_11h=dp_10_11.loc[(dp_10_11['home_team']==team_choisi(team))]
    dp_10_11h=dp_10_11h.groupby(dp_10_11h['season']).sum()
    dp_10_11a=dp_10_11.loc[(dp_10_11['away_team']==team_choisi(team))]
    dp_10_11a=dp_10_11a.groupby(dp_10_11a['season']).sum()
    result_10_11=dp_10_11h[['home_wins','home_draws']]
    result_10_11['home_losses']=(19-(result_10_11['home_wins']+result_10_11['home_draws']))
    result_10_11[['away_wins','away_draws']]=dp_10_11a[['away_wins','away_draws']]
    result_10_11['away_losses']=(19-(result_10_11['away_wins']+result_10_11['away_draws']))
    result_10_11['total_wins']=result_10_11['home_wins']+result_10_11['away_wins']
    result_10_11['total_draws']=result_10_11['home_draws']+result_10_11['away_draws']
    result_10_11['total_losses']=result_10_11['home_losses']+result_10_11['away_losses']

    stat_10_11=dp_10_11h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_10_11.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_10_11[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_10_11a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_10_11.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_10_11['total_goals_scored']=stat_10_11['home_goals_scored']+stat_10_11['away_goals_scored']
    stat_10_11['total_goals_conced']=stat_10_11['home_goals_conced']+stat_10_11['away_goals_conced']
    stat_10_11['total_target_shots']=stat_10_11['home_shots_on_target']+stat_10_11['away_shots_on_target']
    stat_10_11['total_shots']=stat_10_11['home_shots']+stat_10_11['away_shots']
    stat_10_11['total_passes']=stat_10_11['home_passes']+stat_10_11['away_passes']
    stat_10_11['average_possession_h']=stat_10_11['home_possession']/19
    stat_10_11['average_possession_a']=stat_10_11['away_possession']/19
    stat_10_11['average_possession']=(stat_10_11['average_possession_a']+stat_10_11['average_possession_h'])/2
    stat_10_11['total_offsides']=stat_10_11['home_offsides']+stat_10_11['away_offsides']
    stat_10_11['total_fouls_conceded']=stat_10_11['home_fouls_conceded']+stat_10_11['away_fouls_conceded']
    stat_10_11['total_red_cards']=stat_10_11['home_red_cards']+stat_10_11['away_red_cards']
    stat_10_11['total_yellow_cards']=stat_10_11['home_yellow_cards']+stat_10_11['away_yellow_cards']
    stat_10_11['total_tackles']=stat_10_11['home_tackles']+stat_10_11['away_tackles']
    stat_10_11['total_clearances']=stat_10_11['home_clearances']+stat_10_11['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2011/2012
    dp_11_12=dp.loc[dp['season']=='11/12']
    dp_11_12h=dp_11_12.loc[(dp_11_12['home_team']==team_choisi(team))]
    dp_11_12h=dp_11_12h.groupby(dp_11_12h['season']).sum()
    dp_11_12a=dp_11_12.loc[(dp_11_12['away_team']==team_choisi(team))]
    dp_11_12a=dp_11_12a.groupby(dp_11_12a['season']).sum()
    result_11_12=dp_11_12h[['home_wins','home_draws']]
    result_11_12['home_losses']=(19-(result_11_12['home_wins']+result_11_12['home_draws']))
    result_11_12[['away_wins','away_draws']]=dp_11_12a[['away_wins','away_draws']]
    result_11_12['away_losses']=(19-(result_11_12['away_wins']+result_11_12['away_draws']))
    result_11_12['total_wins']=result_11_12['home_wins']+result_11_12['away_wins']
    result_11_12['total_draws']=result_11_12['home_draws']+result_11_12['away_draws']
    result_11_12['total_losses']=result_11_12['home_losses']+result_11_12['away_losses']


    stat_11_12=dp_11_12h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_11_12.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_11_12[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_11_12a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_11_12.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_11_12['total_goals_scored']=stat_11_12['home_goals_scored']+stat_11_12['away_goals_scored']
    stat_11_12['total_goals_conced']=stat_11_12['home_goals_conced']+stat_11_12['away_goals_conced']
    stat_11_12['total_target_shots']=stat_11_12['home_shots_on_target']+stat_11_12['away_shots_on_target']
    stat_11_12['total_shots']=stat_11_12['home_shots']+stat_11_12['away_shots']
    stat_11_12['total_passes']=stat_11_12['home_passes']+stat_11_12['away_passes']
    stat_11_12['average_possession_h']=stat_11_12['home_possession']/19
    stat_11_12['average_possession_a']=stat_11_12['away_possession']/19
    stat_11_12['average_possession']=(stat_11_12['average_possession_a']+stat_11_12['average_possession_h'])/2
    stat_11_12['total_offsides']=stat_11_12['home_offsides']+stat_11_12['away_offsides']
    stat_11_12['total_fouls_conceded']=stat_11_12['home_fouls_conceded']+stat_11_12['away_fouls_conceded']
    stat_11_12['total_red_cards']=stat_11_12['home_red_cards']+stat_11_12['away_red_cards']
    stat_11_12['total_yellow_cards']=stat_11_12['home_yellow_cards']+stat_11_12['away_yellow_cards']
    stat_11_12['total_tackles']=stat_11_12['home_tackles']+stat_11_12['away_tackles']
    stat_11_12['total_clearances']=stat_11_12['home_clearances']+stat_11_12['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2012/2013
    dp_12_13=dp.loc[dp['season']=='12/13']
    dp_12_13h=dp_12_13.loc[(dp_12_13['home_team']==team_choisi(team))]
    dp_12_13h=dp_12_13h.groupby(dp_12_13h['season']).sum()
    dp_12_13a=dp_12_13.loc[(dp_12_13['away_team']==team_choisi(team))]
    dp_12_13a=dp_12_13a.groupby(dp_12_13a['season']).sum()
    result_12_13=dp_12_13h[['home_wins','home_draws']]
    result_12_13['home_losses']=(19-(result_12_13['home_wins']+result_12_13['home_draws']))
    result_12_13[['away_wins','away_draws']]=dp_12_13a[['away_wins','away_draws']]
    result_12_13['away_losses']=(19-(result_12_13['away_wins']+result_12_13['away_draws']))
    result_12_13['total_wins']=result_12_13['home_wins']+result_12_13['away_wins']
    result_12_13['total_draws']=result_12_13['home_draws']+result_12_13['away_draws']
    result_12_13['total_losses']=result_12_13['home_losses']+result_12_13['away_losses']


    stat_12_13=dp_12_13h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_12_13.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_12_13[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_12_13a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_12_13.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_12_13['total_goals_scored']=stat_12_13['home_goals_scored']+stat_12_13['away_goals_scored']
    stat_12_13['total_goals_conced']=stat_12_13['home_goals_conced']+stat_12_13['away_goals_conced']
    stat_12_13['total_target_shots']=stat_12_13['home_shots_on_target']+stat_12_13['away_shots_on_target']
    stat_12_13['total_shots']=stat_12_13['home_shots']+stat_12_13['away_shots']
    stat_12_13['total_passes']=stat_12_13['home_passes']+stat_12_13['away_passes']
    stat_12_13['average_possession_h']=stat_12_13['home_possession']/19
    stat_12_13['average_possession_a']=stat_12_13['away_possession']/19
    stat_12_13['average_possession']=(stat_12_13['average_possession_a']+stat_12_13['average_possession_h'])/2
    stat_12_13['total_offsides']=stat_12_13['home_offsides']+stat_12_13['away_offsides']
    stat_12_13['total_fouls_conceded']=stat_12_13['home_fouls_conceded']+stat_12_13['away_fouls_conceded']
    stat_12_13['total_red_cards']=stat_12_13['home_red_cards']+stat_12_13['away_red_cards']
    stat_12_13['total_yellow_cards']=stat_12_13['home_yellow_cards']+stat_12_13['away_yellow_cards']
    stat_12_13['total_tackles']=stat_12_13['home_tackles']+stat_12_13['away_tackles']
    stat_12_13['total_clearances']=stat_12_13['home_clearances']+stat_12_13['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2013/2014
    dp_13_14=dp.loc[dp['season']=='13/14']
    dp_13_14h=dp_13_14.loc[(dp_13_14['home_team']==team_choisi(team))]
    dp_13_14h=dp_13_14h.groupby(dp_13_14h['season']).sum()
    dp_13_14a=dp_13_14.loc[(dp_13_14['away_team']==team_choisi(team))]
    dp_13_14a=dp_13_14a.groupby(dp_13_14a['season']).sum()
    result_13_14=dp_13_14h[['home_wins','home_draws']]
    result_13_14['home_losses']=(19-(result_13_14['home_wins']+result_13_14['home_draws']))
    result_13_14[['away_wins','away_draws']]=dp_13_14a[['away_wins','away_draws']]
    result_13_14['away_losses']=(19-(result_13_14['away_wins']+result_13_14['away_draws']))
    result_13_14['total_wins']=result_13_14['home_wins']+result_13_14['away_wins']
    result_13_14['total_draws']=result_13_14['home_draws']+result_13_14['away_draws']
    result_13_14['total_losses']=result_13_14['home_losses']+result_13_14['away_losses']


    stat_13_14=dp_13_14h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_13_14.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_13_14[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_13_14a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_13_14.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_13_14['total_goals_scored']=stat_13_14['home_goals_scored']+stat_13_14['away_goals_scored']
    stat_13_14['total_goals_conced']=stat_13_14['home_goals_conced']+stat_13_14['away_goals_conced']
    stat_13_14['total_target_shots']=stat_13_14['home_shots_on_target']+stat_13_14['away_shots_on_target']
    stat_13_14['total_shots']=stat_13_14['home_shots']+stat_13_14['away_shots']
    stat_13_14['total_passes']=stat_13_14['home_passes']+stat_13_14['away_passes']
    stat_13_14['average_possession_h']=stat_13_14['home_possession']/19
    stat_13_14['average_possession_a']=stat_13_14['away_possession']/19
    stat_13_14['average_possession']=(stat_13_14['average_possession_a']+stat_13_14['average_possession_h'])/2
    stat_13_14['total_offsides']=stat_13_14['home_offsides']+stat_13_14['away_offsides']
    stat_13_14['total_fouls_conceded']=stat_13_14['home_fouls_conceded']+stat_13_14['away_fouls_conceded']
    stat_13_14['total_red_cards']=stat_13_14['home_red_cards']+stat_13_14['away_red_cards']
    stat_13_14['total_yellow_cards']=stat_13_14['home_yellow_cards']+stat_13_14['away_yellow_cards']
    stat_13_14['total_tackles']=stat_13_14['home_tackles']+stat_13_14['away_tackles']
    stat_13_14['total_clearances']=stat_13_14['home_clearances']+stat_13_14['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2014/2015
    dp_14_15=dp.loc[dp['season']=='14/15']
    dp_14_15h=dp_14_15.loc[(dp_14_15['home_team']==team_choisi(team))]
    dp_14_15h=dp_14_15h.groupby(dp_14_15h['season']).sum()
    dp_14_15a=dp_14_15.loc[(dp_14_15['away_team']==team_choisi(team))]
    dp_14_15a=dp_14_15a.groupby(dp_14_15a['season']).sum()
    result_14_15=dp_14_15h[['home_wins','home_draws']]
    result_14_15['home_losses']=(19-(result_14_15['home_wins']+result_14_15['home_draws']))
    result_14_15[['away_wins','away_draws']]=dp_14_15a[['away_wins','away_draws']]
    result_14_15['away_losses']=(19-(result_14_15['away_wins']+result_14_15['away_draws']))
    result_14_15['total_wins']=result_14_15['home_wins']+result_14_15['away_wins']
    result_14_15['total_draws']=result_14_15['home_draws']+result_14_15['away_draws']
    result_14_15['total_losses']=result_14_15['home_losses']+result_14_15['away_losses']


    stat_14_15=dp_14_15h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_14_15.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_14_15[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_14_15a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_14_15.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_14_15['total_goals_scored']=stat_14_15['home_goals_scored']+stat_14_15['away_goals_scored']
    stat_14_15['total_goals_conced']=stat_14_15['home_goals_conced']+stat_14_15['away_goals_conced']
    stat_14_15['total_target_shots']=stat_14_15['home_shots_on_target']+stat_14_15['away_shots_on_target']
    stat_14_15['total_shots']=stat_14_15['home_shots']+stat_14_15['away_shots']
    stat_14_15['total_passes']=stat_14_15['home_passes']+stat_14_15['away_passes']
    stat_14_15['average_possession_h']=stat_14_15['home_possession']/19
    stat_14_15['average_possession_a']=stat_14_15['away_possession']/19
    stat_14_15['average_possession']=(stat_14_15['average_possession_a']+stat_14_15['average_possession_h'])/2
    stat_14_15['total_offsides']=stat_14_15['home_offsides']+stat_14_15['away_offsides']
    stat_14_15['total_fouls_conceded']=stat_14_15['home_fouls_conceded']+stat_14_15['away_fouls_conceded']
    stat_14_15['total_red_cards']=stat_14_15['home_red_cards']+stat_14_15['away_red_cards']
    stat_14_15['total_yellow_cards']=stat_14_15['home_yellow_cards']+stat_14_15['away_yellow_cards']
    stat_14_15['total_tackles']=stat_14_15['home_tackles']+stat_14_15['away_tackles']
    stat_14_15['total_clearances']=stat_14_15['home_clearances']+stat_14_15['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2015/2016
    dp_15_16=dp.loc[dp['season']=='15/16']
    dp_15_16h=dp_15_16.loc[(dp_15_16['home_team']==team_choisi(team))]
    dp_15_16h=dp_15_16h.groupby(dp_15_16h['season']).sum()
    dp_15_16a=dp_15_16.loc[(dp_15_16['away_team']==team_choisi(team))]
    dp_15_16a=dp_15_16a.groupby(dp_15_16a['season']).sum()
    result_15_16=dp_15_16h[['home_wins','home_draws']]
    result_15_16['home_losses']=(19-(result_15_16['home_wins']+result_15_16['home_draws']))
    result_15_16[['away_wins','away_draws']]=dp_15_16a[['away_wins','away_draws']]
    result_15_16['away_losses']=(19-(result_15_16['away_wins']+result_15_16['away_draws']))
    result_15_16['total_wins']=result_15_16['home_wins']+result_15_16['away_wins']
    result_15_16['total_draws']=result_15_16['home_draws']+result_15_16['away_draws']
    result_15_16['total_losses']=result_15_16['home_losses']+result_15_16['away_losses']


    stat_15_16=dp_15_16h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_15_16.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_15_16[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_15_16a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_15_16.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_15_16['total_goals_scored']=stat_15_16['home_goals_scored']+stat_15_16['away_goals_scored']
    stat_15_16['total_goals_conced']=stat_15_16['home_goals_conced']+stat_15_16['away_goals_conced']
    stat_15_16['total_target_shots']=stat_15_16['home_shots_on_target']+stat_15_16['away_shots_on_target']
    stat_15_16['total_shots']=stat_15_16['home_shots']+stat_15_16['away_shots']
    stat_15_16['total_passes']=stat_15_16['home_passes']+stat_15_16['away_passes']
    stat_15_16['average_possession_h']=stat_15_16['home_possession']/19
    stat_15_16['average_possession_a']=stat_15_16['away_possession']/19
    stat_15_16['average_possession']=(stat_15_16['average_possession_a']+stat_15_16['average_possession_h'])/2
    stat_15_16['total_offsides']=stat_15_16['home_offsides']+stat_15_16['away_offsides']
    stat_15_16['total_fouls_conceded']=stat_15_16['home_fouls_conceded']+stat_15_16['away_fouls_conceded']
    stat_15_16['total_red_cards']=stat_15_16['home_red_cards']+stat_15_16['away_red_cards']
    stat_15_16['total_yellow_cards']=stat_15_16['home_yellow_cards']+stat_15_16['away_yellow_cards']
    stat_15_16['total_tackles']=stat_15_16['home_tackles']+stat_15_16['away_tackles']
    stat_15_16['total_clearances']=stat_15_16['home_clearances']+stat_15_16['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2016/2017
    dp_16_17=dp.loc[dp['season']=='16/17']
    dp_16_17h=dp_16_17.loc[(dp_16_17['home_team']==team_choisi(team))]
    dp_16_17h=dp_16_17h.groupby(dp_16_17h['season']).sum()
    dp_16_17a=dp_16_17.loc[(dp_16_17['away_team']==team_choisi(team))]
    dp_16_17a=dp_16_17a.groupby(dp_16_17a['season']).sum()
    result_16_17=dp_16_17h[['home_wins','home_draws']]
    result_16_17['home_losses']=(19-(result_16_17['home_wins']+result_16_17['home_draws']))
    result_16_17[['away_wins','away_draws']]=dp_16_17a[['away_wins','away_draws']]
    result_16_17['away_losses']=(19-(result_16_17['away_wins']+result_16_17['away_draws']))
    result_16_17['total_wins']=result_16_17['home_wins']+result_16_17['away_wins']
    result_16_17['total_draws']=result_16_17['home_draws']+result_16_17['away_draws']
    result_16_17['total_losses']=result_16_17['home_losses']+result_16_17['away_losses']

    stat_16_17=dp_16_17h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_16_17.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_16_17[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_16_17a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_16_17.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_16_17['total_goals_scored']=stat_16_17['home_goals_scored']+stat_16_17['away_goals_scored']
    stat_16_17['total_goals_conced']=stat_16_17['home_goals_conced']+stat_16_17['away_goals_conced']
    stat_16_17['total_target_shots']=stat_16_17['home_shots_on_target']+stat_16_17['away_shots_on_target']
    stat_16_17['total_shots']=stat_16_17['home_shots']+stat_16_17['away_shots']
    stat_16_17['total_passes']=stat_16_17['home_passes']+stat_16_17['away_passes']
    stat_16_17['average_possession_h']=stat_16_17['home_possession']/19
    stat_16_17['average_possession_a']=stat_16_17['away_possession']/19
    stat_16_17['average_possession']=(stat_16_17['average_possession_a']+stat_16_17['average_possession_h'])/2
    stat_16_17['total_offsides']=stat_16_17['home_offsides']+stat_16_17['away_offsides']
    stat_16_17['total_fouls_conceded']=stat_16_17['home_fouls_conceded']+stat_16_17['away_fouls_conceded']
    stat_16_17['total_red_cards']=stat_16_17['home_red_cards']+stat_16_17['away_red_cards']
    stat_16_17['total_yellow_cards']=stat_16_17['home_yellow_cards']+stat_16_17['away_yellow_cards']
    stat_16_17['total_tackles']=stat_16_17['home_tackles']+stat_16_17['away_tackles']
    stat_16_17['total_clearances']=stat_16_17['home_clearances']+stat_16_17['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2017/2018
    dp_17_18=dp.loc[dp['season']=='17/18']
    dp_17_18h=dp_17_18.loc[(dp_17_18['home_team']==team_choisi(team))]
    dp_17_18h=dp_17_18h.groupby(dp_17_18h['season']).sum()
    dp_17_18a=dp_17_18.loc[(dp_17_18['away_team']==team_choisi(team))]
    dp_17_18a=dp_17_18a.groupby(dp_17_18a['season']).sum()
    result_17_18=dp_17_18h[['home_wins','home_draws']]
    result_17_18['home_losses']=(19-(result_17_18['home_wins']+result_17_18['home_draws']))
    result_17_18[['away_wins','away_draws']]=dp_17_18a[['away_wins','away_draws']]
    result_17_18['away_losses']=(19-(result_17_18['away_wins']+result_17_18['away_draws']))
    result_17_18['total_wins']=result_17_18['home_wins']+result_17_18['away_wins']
    result_17_18['total_draws']=result_17_18['home_draws']+result_17_18['away_draws']
    result_17_18['total_losses']=result_17_18['home_losses']+result_17_18['away_losses']


    stat_17_18=dp_17_18h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_17_18.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_17_18[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_17_18a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_17_18.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_17_18['total_goals_scored']=stat_17_18['home_goals_scored']+stat_17_18['away_goals_scored']
    stat_17_18['total_goals_conced']=stat_17_18['home_goals_conced']+stat_17_18['away_goals_conced']
    stat_17_18['total_target_shots']=stat_17_18['home_shots_on_target']+stat_17_18['away_shots_on_target']
    stat_17_18['total_shots']=stat_17_18['home_shots']+stat_17_18['away_shots']
    stat_17_18['total_passes']=stat_17_18['home_passes']+stat_17_18['away_passes']
    stat_17_18['average_possession_h']=stat_17_18['home_possession']/19
    stat_17_18['average_possession_a']=stat_17_18['away_possession']/19
    stat_17_18['average_possession']=(stat_17_18['average_possession_a']+stat_17_18['average_possession_h'])/2
    stat_17_18['total_offsides']=stat_17_18['home_offsides']+stat_17_18['away_offsides']
    stat_17_18['total_fouls_conceded']=stat_17_18['home_fouls_conceded']+stat_17_18['away_fouls_conceded']
    stat_17_18['total_red_cards']=stat_17_18['home_red_cards']+stat_17_18['away_red_cards']
    stat_17_18['total_yellow_cards']=stat_17_18['home_yellow_cards']+stat_17_18['away_yellow_cards']
    stat_17_18['total_tackles']=stat_17_18['home_tackles']+stat_17_18['away_tackles']
    stat_17_18['total_clearances']=stat_17_18['home_clearances']+stat_17_18['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2018/2019
    dp_18_19=dp.loc[dp['season']=='18/19']
    dp_18_19h=dp_18_19.loc[(dp_18_19['home_team']==team_choisi(team))]
    dp_18_19h=dp_18_19h.groupby(dp_18_19h['season']).sum()
    dp_18_19a=dp_18_19.loc[(dp_18_19['away_team']==team_choisi(team))]
    dp_18_19a=dp_18_19a.groupby(dp_18_19a['season']).sum()
    result_18_19=dp_18_19h[['home_wins','home_draws']]
    result_18_19['home_losses']=(19-(result_18_19['home_wins']+result_18_19['home_draws']))
    result_18_19[['away_wins','away_draws']]=dp_18_19a[['away_wins','away_draws']]
    result_18_19['away_losses']=(19-(result_18_19['away_wins']+result_18_19['away_draws']))
    result_18_19['total_wins']=result_18_19['home_wins']+result_18_19['away_wins']
    result_18_19['total_draws']=result_18_19['home_draws']+result_18_19['away_draws']
    result_18_19['total_losses']=result_18_19['home_losses']+result_18_19['away_losses']


    stat_18_19=dp_18_19h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_18_19.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_18_19[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_18_19a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_18_19.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_18_19['total_goals_scored']=stat_18_19['home_goals_scored']+stat_18_19['away_goals_scored']
    stat_18_19['total_goals_conced']=stat_18_19['home_goals_conced']+stat_18_19['away_goals_conced']
    stat_18_19['total_target_shots']=stat_18_19['home_shots_on_target']+stat_18_19['away_shots_on_target']
    stat_18_19['total_shots']=stat_18_19['home_shots']+stat_18_19['away_shots']
    stat_18_19['total_passes']=stat_18_19['home_passes']+stat_18_19['away_passes']
    stat_18_19['average_possession_h']=stat_18_19['home_possession']/19
    stat_18_19['average_possession_a']=stat_18_19['away_possession']/19
    stat_18_19['average_possession']=(stat_18_19['average_possession_a']+stat_18_19['average_possession_h'])/2
    stat_18_19['total_offsides']=stat_18_19['home_offsides']+stat_18_19['away_offsides']
    stat_18_19['total_fouls_conceded']=stat_18_19['home_fouls_conceded']+stat_18_19['away_fouls_conceded']
    stat_18_19['total_red_cards']=stat_18_19['home_red_cards']+stat_18_19['away_red_cards']
    stat_18_19['total_yellow_cards']=stat_18_19['home_yellow_cards']+stat_18_19['away_yellow_cards']
    stat_18_19['total_tackles']=stat_18_19['home_tackles']+stat_18_19['away_tackles']
    stat_18_19['total_clearances']=stat_18_19['home_clearances']+stat_18_19['away_clearances']

    #----------------------------------------------------------------------------------------------
    #saisons 2019/2020
    dp_19_20=dp.loc[dp['season']=='19/20']
    dp_19_20h=dp_19_20.loc[(dp_19_20['home_team']==team_choisi(team))]
    dp_19_20h=dp_19_20h.groupby(dp_19_20h['season']).sum()
    dp_19_20a=dp_19_20.loc[(dp_19_20['away_team']==team_choisi(team))]
    dp_19_20a=dp_19_20a.groupby(dp_19_20a['season']).sum()
    result_19_20=dp_19_20h[['home_wins','home_draws']]
    result_19_20['home_losses']=(19-(result_19_20['home_wins']+result_19_20['home_draws']))
    result_19_20[['away_wins','away_draws']]=dp_19_20a[['away_wins','away_draws']]
    result_19_20['away_losses']=(19-(result_19_20['away_wins']+result_19_20['away_draws']))
    result_19_20['total_wins']=result_19_20['home_wins']+result_19_20['away_wins']
    result_19_20['total_draws']=result_19_20['home_draws']+result_19_20['away_draws']
    result_19_20['total_losses']=result_19_20['home_losses']+result_19_20['away_losses']


    stat_19_20=dp_19_20h[["home_clearances","home_corners","home_fouls_conceded","home_offsides","home_passes","home_possession",
    "home_red_cards","home_shots","home_shots_on_target","home_tackles","home_touches","home_yellow_cards","goal_home_ft","goal_away_ft"]]
    stat_19_20.rename(columns={'goal_away_ft': 'home_goals_conced','goal_home_ft':'home_goals_scored'},inplace=True)
    stat_19_20[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]=dp_19_20a[["away_clearances","away_corners","away_fouls_conceded","away_offsides","away_passes","away_possession",
    "away_red_cards","away_shots","away_shots_on_target","away_tackles","away_touches","away_yellow_cards","goal_away_ft","goal_home_ft"]]
    stat_19_20.rename(columns={'goal_away_ft': 'away_goals_scored','goal_home_ft':'away_goals_conced'},inplace=True)
    stat_19_20['total_goals_scored']=stat_19_20['home_goals_scored']+stat_19_20['away_goals_scored']
    stat_19_20['total_goals_conced']=stat_19_20['home_goals_conced']+stat_19_20['away_goals_conced']
    stat_19_20['total_target_shots']=stat_19_20['home_shots_on_target']+stat_19_20['away_shots_on_target']
    stat_19_20['total_shots']=stat_19_20['home_shots']+stat_19_20['away_shots']
    stat_19_20['total_passes']=stat_19_20['home_passes']+stat_19_20['away_passes']
    stat_19_20['average_possession_h']=stat_19_20['home_possession']/19
    stat_19_20['average_possession_a']=stat_19_20['away_possession']/19
    stat_19_20['average_possession']=(stat_19_20['average_possession_a']+stat_19_20['average_possession_h'])/2
    stat_19_20['total_offsides']=stat_19_20['home_offsides']+stat_19_20['away_offsides']
    stat_19_20['total_fouls_conceded']=stat_19_20['home_fouls_conceded']+stat_19_20['away_fouls_conceded']
    stat_19_20['total_red_cards']=stat_19_20['home_red_cards']+stat_19_20['away_red_cards']
    stat_19_20['total_yellow_cards']=stat_19_20['home_yellow_cards']+stat_19_20['away_yellow_cards']
    stat_19_20['total_tackles']=stat_19_20['home_tackles']+stat_19_20['away_tackles']
    stat_19_20['total_clearances']=stat_19_20['home_clearances']+stat_19_20['away_clearances']
    #-----------------------------------------------------------------------------------------
    #Webscraping classement saison 2010/2011

    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2010-2011/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())
    #Dataframe comportant le classement de la saison 2010/2011    
    saison10_11= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison10_11.set_index('Classement',inplace=True)
    saison10_11['Points']=saison10_11['Points'].astype(int)


    #Webscraping classement saison 2011/2012

    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2011-2012/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())
    #Dataframe comportant le classement de la saison 2011/2012     
    saison11_12= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison11_12.set_index('Classement',inplace=True)
    saison11_12['Points']=saison11_12['Points'].astype(int)
    # In[38]:


    #Webscraping classement saison 2012/2013 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2012-2013/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison12_13= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison12_13.set_index('Classement',inplace=True)
    saison12_13['Points']=saison12_13['Points'].astype(int)

    # In[39]:


    #Webscraping classement saison 2013/2014 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2013-2014/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison13_14= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison13_14.set_index('Classement',inplace=True)
    saison13_14['Points']=saison13_14['Points'].astype(int)

    # In[40]:


    #Webscraping classement saison 2014/2015 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2014-2015/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison14_15= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison14_15.set_index('Classement',inplace=True)
    saison14_15['Points']=saison14_15['Points'].astype(int)

    # In[41]:


    #Webscraping classement saison 2015/2016 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2015-2016/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison15_16= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison15_16.set_index('Classement',inplace=True)
    saison15_16['Points']=saison15_16['Points'].astype(int)

    # In[42]:


    #Webscraping classement saison 2016/2017 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2016-2017/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison16_17= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison16_17.set_index('Classement',inplace=True)
    saison16_17['Points']=saison16_17['Points'].astype(int)

    # In[43]:


    #Webscraping classement saison 2017/2018 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2017-2018/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison17_18= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison17_18.set_index('Classement',inplace=True)
    saison17_18['Points']=saison17_18['Points'].astype(int)

    # In[44]:


    #Webscraping classement saison 2018/2019 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2018-2019/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison18_19= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison18_19.set_index('Classement',inplace=True)
    saison18_19['Points']=saison18_19['Points'].astype(int)

    # In[45]:


    #Webscraping classement saison 2019/2020 et création du dataframe 
    page=urlopen("https://www.footmercato.net/angleterre/premier-league/2019-2020/classement")
    soup = BeautifulSoup(page, 'html.parser')
    rank = [] 
    for element in soup.findAll('td',attrs={'class':'classement__rank'}):
        rank.append(element.text.strip())

    teams= [] 
    for element in soup.findAll('td',attrs={'class':'classement__team'}):
        teams.append(element.text.strip())   

    pts= [] 
    for element in soup.findAll('td',attrs={'class':'classement__highlight'}):
        pts.append(element.text.strip())

    saison19_20= pd.DataFrame(list(zip(rank,teams,pts)), columns=["Classement","Equipes","Points"])
    saison19_20.set_index('Classement',inplace=True)
    saison19_20['Points']=saison19_20['Points'].astype(int)

    #-------------------------------------------------------------------------------------      
    fig1 = px.bar(saison10_11, x='Equipes', y='Points', height=400)
    fig2 = px.bar(saison11_12, x='Equipes', y='Points', height=400)
    fig3 = px.bar(saison12_13, x='Equipes', y='Points', height=400)
    fig4 = px.bar(saison13_14, x='Equipes', y='Points', height=400)
    fig5 = px.bar(saison14_15, x='Equipes', y='Points', height=400)
    fig6 = px.bar(saison15_16, x='Equipes', y='Points', height=400)
    fig7 = px.bar(saison16_17, x='Equipes', y='Points', height=400)
    fig8 = px.bar(saison17_18, x='Equipes', y='Points', height=400)
    fig9 = px.bar(saison18_19, x='Equipes', y='Points', height=400)
    fig10 = px.bar(saison19_20, x='Equipes', y='Points', height=400)

    if st.checkbox("Saison 2010/2011"):
        st.dataframe(result_10_11)
        barplot_chart = st.write(fig1)

    if st.checkbox("Saison 2011/2012"):
        st.dataframe(result_11_12)
        barplot_chart = st.write(fig2)
    if st.checkbox("Saison 2012/2013"):
        st.dataframe(result_12_13)
        barplot_chart = st.write(fig3)

    if st.checkbox("Saison 2013/2014"):
        st.dataframe(result_13_14) 
        barplot_chart = st.write(fig4)

    if st.checkbox("Saison 2014/2015"):
        st.dataframe(result_14_15)
        barplot_chart = st.write(fig5)

    if st.checkbox("Saison 2015/2016"):
        st.dataframe(result_15_16)
        barplot_chart = st.write(fig6)

    if st.checkbox("Saison 2016/2017"):
        st.dataframe(result_16_17)
        barplot_chart = st.write(fig7)

    if st.checkbox("Saison 2017/2018"):
        st.dataframe(result_17_18)
        barplot_chart = st.write(fig8)

    if st.checkbox("Saison 2018/2019"):
        st.dataframe(result_18_19)
        barplot_chart = st.write(fig9)

    if st.checkbox("Saison 2019/2020"):
        st.dataframe(result_19_20)
        barplot_chart = st.write(fig10)

    st.title("Evolution de la statistique choisie au fil des saisons ") 
    col=st.selectbox(label="Selectionner la catégorie:",options=['home_clearances', 'home_corners', 'home_fouls_conceded',
           'home_offsides', 'home_passes', 'home_possession', 'home_red_cards',
           'home_shots', 'home_shots_on_target', 'home_tackles', 'home_touches',
           'home_yellow_cards', 'home_goals_scored', 'home_goals_conced',
           'away_clearances', 'away_corners', 'away_fouls_conceded',
           'away_offsides', 'away_passes', 'away_possession', 'away_red_cards',
           'away_shots', 'away_shots_on_target', 'away_tackles', 'away_touches',
           'away_yellow_cards', 'away_goals_scored', 'away_goals_conced',
           'total_goals_scored', 'total_goals_conced', 'total_target_shots',
           'total_shots', 'total_passes', 'average_possession_h',
           'average_possession_a', 'average_possession', 'total_offsides',
           'total_fouls_conceded', 'total_red_cards', 'total_yellow_cards',
           'total_tackles', 'total_clearances']) 
    recap=pd.concat([stat_10_11,stat_11_12,stat_12_13,stat_13_14,stat_14_15,stat_15_16,stat_16_17,stat_17_18,stat_18_19,stat_19_20])
    recap=recap.fillna(0)

    def cat_choisi(col):
        if col==col:
            return col

    figcat=px.line(recap, y=cat_choisi(col)) 
    st.write(figcat)
    st.write("")


#_______________________________________________________________________________________________________________________________________
#333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

if sommaire == 'Zoom sur Arsenal':

    st.title('Zoom sur Arsenal')  
    
    st.image("Arsenal_FC_logo.png", width=150)
              
    st.write("Arsenal Football Club a été fondé le 1er décembre 1886 et participe au championnat d'Angleterre depuis 1919 dont il a remporté treize éditions (3e club le plus titré) et quatorze coupes (1er). Ce statut de club phare du championnat leur a été attribué entre les années 1990 et 2005 avec un Arsenal au sommet.")     
    st.write("Quelques années plus tard, Arsenal est dans une situation beaucoup plus délicate. Nous sommes dans une période où les titres sont plus rares, le jeu produit est moins attrayant et on commence à remettre en question leur présence dans le Big Six.")
    
    st.write("Notre objectif est de répondre par l'analyse des données à la problématique suivante: **_Comment le niveau du club de football Arsenal FC s’est dégradé pendant la dernière décennie_** ?")
    
    
    st.write("Nous cherchons dans un premier temps à quel moment a eu lieu le déclin d’Arsenal puis, nous analysons les explications de cette dégradation en approfondissant l’analyse du dataset.") 
    
    st.title("Analyse des données")

    st.write("La performance d’une équipe de football est toujours jugée au travers de ses résultats. Le graphique suivant met en évidence le nombres de points gagnés par saison par le champion, la moyenne du Big Five (Big Six sans Arsenal), des 2O équipes et les points d’Arsenal.")
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
    #Variable points gagnés arsenal
    classement_arsenal=classement_all_season.loc[classement_all_season['Equipes']=='Arsenal']
    classement_arsenal['Points']=classement_arsenal['Points'].astype(int)
    
    #Variable des points gagnés par Big 5
    classement_big5=classement_all_season.loc[(classement_all_season["Equipes"]=="Man City")|(classement_all_season["Equipes"]=="Man United")|(classement_all_season["Equipes"]=="Chelsea")|(classement_all_season["Equipes"]=="Liverpool")|(classement_all_season["Equipes"]=="Tottenham")]
    classement_big5['Points']=classement_big5['Points'].astype(int)
    classement_big5=classement_big5.groupby(classement_big5['season']).sum()/5
    
    #Variable de la moyenne des points gagnés par les équipes
    classement_20teams=classement_all_season
    classement_20teams['Points']=classement_20teams['Points'].astype(int)
    classement_20teams=classement_20teams.groupby(classement_20teams['season']).sum()/20
    
    #Varibale des points gagnés par les champions par saison
    points_champion=classement_all_season.loc[classement_all_season['Classement']=='1']
    points_champion['Points']=points_champion['Points'].astype(int)
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    fig = plt.figure(figsize=(14,7))
    
    
    plt.plot(classement_arsenal['season'],classement_arsenal['Points'],'#95190C',linewidth=3, label='Arsenal')
    plt.plot(classement_arsenal['season'],classement_big5['Points'],'#E3B505', linewidth=3,label='Moyenne Big5')
    plt.plot(classement_arsenal['season'],classement_20teams['Points'],'#107E7D', linewidth=3, label="Moyenne des 20 équipes")
    plt.plot(classement_arsenal['season'],points_champion['Points'],'#044B7F', linewidth=3, label="Champion par saison")
    plt.ylim([50,110]);
    plt.xlabel('Saison')
    plt.ylabel('Points gagnés')
    plt.title("Evolution du nombre de points gagnés en fonction des saisons")
    plt.legend();

    st.pyplot(fig)

    st.write("On observe un déclin d'Arsenal à partir de 16/17 et l'écart qui se creuse entre Arsenal et le reste des équipes du Big 6")

    st.write(" Les graphiques ci-après mettent en avant les victoires et les défaites d’Arsenal, du Big 5, des 14 autres équipes au fil des saisons. Les courbes représentent les performances des équipes du Big Five des 14 autres équipes du championnat et la moyenne des 20 équipes de la saison respective.")
    

#Graphique des victoires 
    fig = plt.figure(figsize=(15,15))

    plt.subplot(321)
    plt.bar(ars_wins_h.index,ars_wins_h['Winner_Home'],label='Arsenal')
    plt.plot(range(len(big5_wins_h.index)),big5_wins_h['Winner_Home'],'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_wins_h.index)),teams14_wins_h['Winner_Home'],'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_wins_h.index)),teams20_wins_h['Winner_Home'],'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('Victoires à domicile par saison')
    plt.ylim([0,16])
    plt.legend(loc='lower left')

    plt.subplot(323)
    plt.bar(ars_wins_a.index,ars_wins_a['Winner_Away'],label='Arsenal')
    plt.plot(range(len(big5_wins_a.index)),big5_wins_a['Winner_Away'],'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_wins_a.index)),teams14_wins_a['Winner_Away'],'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_wins_a.index)),teams20_wins_a['Winner_Away'],'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('''Victoires à l'éxterieur par saison''')
    plt.ylim([0,16])
    plt.legend(loc='upper left')

    plt.subplot(325)
    plt.bar(ars_all_wins.index,ars_all_wins['total_wins'],label='Arsenal')
    plt.plot(range(len(all_big5_wins.index)),all_big5_wins['total_wins'],'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(all_teams14_wins.index)),all_teams14_wins['total_wins'],'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(all_teams20_wins.index)),all_teams20_wins['total_wins'],'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('Total des victoires par saison')
    plt.ylim([0,26])
    plt.legend(loc='lower left');  

#Graphique des défaites 
    plt.subplot(322)
    plt.bar(ars_losses_h.index,ars_losses_h['Loser_Home'],label='Arsenal',color='orange')
    plt.plot(range(len(big5_losses_h.index)),big5_losses_h['Loser_Home'],'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_losses_h.index)),teams14_losses_h['Loser_Home'],'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_losses_h.index)),teams20_losses_h['Loser_Home'],'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('Défaites à domicile par saison')
    plt.ylim([0,16])
    plt.legend(loc='upper left')


    plt.subplot(324)
    plt.bar(ars_losses_a.index,ars_losses_a['Loser_Away'],label='Arsenal',color='orange')
    plt.plot(range(len(big5_losses_a.index)),big5_losses_a['Loser_Away'],'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_losses_a.index)),teams14_losses_a['Loser_Away'],'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_losses_a.index)),teams20_losses_a['Loser_Away'],'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('''Défaites à l'exterieur par saison''')
    plt.ylim([0,16])
    plt.legend(loc='upper left')

    plt.subplot(326)
    plt.bar(ars_all_losses.index,ars_all_losses['total_losses'],label='Arsenal',color='orange')
    plt.plot(range(len(all_big5_losses.index)),all_big5_losses['total_losses'],'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(all_teams14_losses.index)),all_teams14_losses['total_losses'],'r-*',label = 'Moyenne de 14 autres équipes',color='green')
    plt.plot(range(len(all_teams20_losses.index)),all_teams20_losses['total_losses'],'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('Total des défaites par saison')
    plt.ylim([0,26])
    plt.legend(loc='upper left');

    st.pyplot(fig)
  
    st.write("La saison 13/14 est la saison la plus aboutie d’Arsenal avec le meilleur total de victoires et le total le plus faible de défaites. Ces graphiques mettent également en lumière le déclin d'Arsenal à partir de la saison 16/17 qui s'expliquent principalement par les mauvais résultats à l'extérieur. Bien qu'Arsenal ne suit plus le rythme du Big Five, il reste cependant au-dessus de la moyenne du championnat.")

    st.write("La différence de but d’une équipe de football est la différence du nombre de buts marqués avec le nombre de buts encaissés. Ainsi, sur ce graphique, nous pouvons observer l’évolution d’Arsenal au fil des saisons.")

#Graphiques Différence de buts
    y1 = goal_ars_h['Goals difference']
    y2 = goal_ars_a['Goals difference']
    y3=all_goals_arsn['Goals difference']
    Categories = ["Arsenal","Big5","team14","team20"]
    nb_categories = len(Categories)
    largeur_barre = floor(1*10/nb_categories)/10

    x1 = range(len(y1))
    x2 =[i + largeur_barre for i in x1]
    x3=[i + 2*largeur_barre for i in x1]
    x4=[i + 3*largeur_barre for i in x1]


    fig = plt.figure(figsize=(14,7))

    plt.bar(goal_ars_h.index,y1,color='#FB6376',label='Domicile',width = largeur_barre)
    plt.bar(x2,y2,color='#FCB1A6',label = 'Extérieur',width = largeur_barre)
    plt.bar(x3,y3,label = 'Total',color='#5D2A42',width = largeur_barre)
    plt.title('''Différences de buts d'Arsenal''')
    plt.xlabel('Saison')
    plt.ylabel('Différence de but')
    plt.legend(loc='upper left');

    st.pyplot(fig)

    st.write("Concernant les difficultés d’Arsenal à l’extérieur, on observe que l'équipe encaisse plus de but qu’elle n’en marque. Par ailleurs, à partir de 17/18, on voit la difficulté d’Arsenal de maintenir le rythme qu’il a toujours eu les saisons précédentes.")

#Graphique des buts marqués

    fig = plt.figure(figsize=(15,15))

    y1 = goal_ars_h['goals_scored_H']
    y2 = goal_big5_h['goals_scored_H']
    y3 = goal_team14_h['goals_scored_H']
    y4 = goal_team20_h['goals_scored_H']
    y5 = goal_ars_a['goals_scored_A']
    y6 = goal_big5_a['goals_scored_A']
    y7 = goal_team14_a['goals_scored_A']
    y8 = goal_team20_a['goals_scored_A']
    y9 = all_goals_arsn['Total Goals Scored']
    y10 = all_goals_big5['Total Goals Scored']
    y11 = all_goals_teams14['Total Goals Scored']
    y12 = all_goals_teams20['Total Goals Scored']

    plt.subplot(321)
    plt.bar(goal_ars_h.index,y1,label='Arsenal')
    plt.plot(range(len(goal_big5_h.index)),y2,'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_wins_h.index)),y3,'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_wins_h.index)),y4,'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('Buts marqués à domicile par saison')
    plt.ylim([0,55])
    plt.legend(loc='lower left')

    plt.subplot(323)
    plt.bar(ars_wins_a.index,y5,label='Arsenal')
    plt.plot(range(len(big5_wins_a.index)),y6,'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_wins_a.index)),y7,'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_wins_a.index)),y8,'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('''Buts marqués à l'éxterieur par saison''')
    plt.ylim([0,55])
    plt.legend(loc='lower left')

    plt.subplot(325)
    plt.bar(ars_all_wins.index,y9,label='Arsenal')
    plt.plot(range(len(all_big5_wins.index)),y10,'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(all_teams14_wins.index)),y12,'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(all_teams20_wins.index)),y11,'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('Total des buts marqués par saison')
    plt.ylim([0,83])
    plt.legend(loc='lower left');

#Graphique des buts encaissés

    y1 = goal_ars_h['goals_conced_H']
    y2 = goal_big5_h['goals_conced_H']
    y3=goal_team14_h['goals_conced_H']
    y4=goal_team20_h['goals_conced_H']
    y5 = goal_ars_a['goals_conced_A']
    y6 = goal_big5_a['goals_conced_A']
    y7=goal_team14_a['goals_conced_A']
    y8=goal_team20_a['goals_conced_A']
    y9=all_goals_arsn['Total Goals conced']
    y10=all_goals_big5['Total Goals conced']
    y11=all_goals_teams14['Total Goals conced']
    y12=all_goals_teams20['Total Goals conced']

    plt.subplot(322)
    plt.bar(goal_ars_h.index,y1,label='Arsenal',color='orange')
    plt.plot(range(len(goal_big5_h.index)),y2,'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_wins_h.index)),y3,'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_wins_h.index)),y4,'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('Buts encaissés à domicile par saison')
    plt.ylim([0,55])
    plt.legend(loc='upper left')

    plt.subplot(324)
    plt.bar(ars_wins_a.index,y5,label='Arsenal',color='orange')
    plt.plot(range(len(big5_wins_a.index)),y6,'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(teams14_wins_a.index)),y7,'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(teams20_wins_a.index)),y8,'r-*',label = 'Moyenne des 20 équipes',color='black')
    plt.title('''Buts encaissés à l'éxterieur par saison''')
    plt.ylim([0,55])
    plt.legend(loc='lower left')

    plt.subplot(326)
    plt.bar(ars_all_wins.index,y9,label='Arsenal',color='orange')
    plt.plot(range(len(all_big5_wins.index)),y10,'r-*',label = 'Moyenne Big5')
    plt.plot(range(len(all_teams14_wins.index)),y12,'r-*',label = 'Moyenne des 14 autres équipes',color='green')
    plt.plot(range(len(all_teams20_wins.index)),y11,'r-*',label = 'Moyenne des 20 équipes',color='black')
    
    plt.title('Total des buts encaissés par saison')
    plt.ylim([0,83])
    plt.legend(loc='lower left');

    st.pyplot(fig)

    st.write("Sur les 4 dernières saisons, la tendance du déclin d’Arsenal peut s’expliquer par des difficultés à marquer et surtout par de nombreux buts encaissés. Une nouvelle fois, les statistiques à l’extérieur mettent en avant les difficultés à l’extérieur d’Arsenal depuis la saison 16/17. Difficultés à marquer mais surtout à ne pas prendre de but.")

    st.title("Conclusion")

    st.write("L’analyse exploratrice a pour but de mettre en avant les premiers points marquants d’Arsenal et d’en conclure nos premières problématiques.")

    st.write("Ce déclin s’explique par un nombre total de points gagnés en baisse, moins de victoires et plus de défaites, cette tendance s'accélérant pour les dernières saisons, à partir de 16/17. Cette tendance est particulièrement forte pour les matchs à l'extérieur")

    st.write("Les premières explications sont découpées en deux aspects :")     
    st.write("   - l’aspect offensif : Arsenal ne marque plus autant de buts que par le passé.")  
    st.write("   - l’aspect défensif : Arsenal encaisse plus de buts que dans le passé.")  

    st.write("Cette première exploration suscite plusieurs problématiques que nous souhaitons approfondir à savoir:")
    st.write("   - Pourquoi Arsenal marque moins de but ?")
    st.write("   - Pourquoi Arsenal encaisse plus de buts ?")
    st.write("   - A quel moment du match Arsenal est-il plus enclin à perdre un match ?")
    st.write("   - Comment se comporte Arsenal face aux rencontres des équipes du Big Six ?")
    st.write("   - Hors du terrain, comment Arsenal est-il dirigé pour un en arriver là ?")
        




#_______________________________________________________________________________________________________________________________________
#444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444    
if sommaire == "Que s'est-il passé à Arsenal ?":


    st.title("Analyse de l’aspect offensif")
    
    st.write("Nous voulons ici analyser les statistiques d’Arsenal à domicile afin de déterminer comment l’équipe se comporte en attaque et tout particulièrement devant le but.")   
    
#Visualitation Ratio et Stat à domicile

    fig = plt.figure(figsize=(14,5))
        
    plt.plot(ars_attack_realism_ratio.index,ars_attack_realism_ratio['shots_h'],'r-*',color = '#AEDCC0',label = 'shots_h')
    plt.plot(range(len(ars_attack_realism_ratio.index)),ars_attack_realism_ratio['target_shots_h'],'r-*',color = '#7BD389',label = 'target_shots_h')
    plt.plot(range(len(ars_attack_realism_ratio.index)),ars_attack_realism_ratio['goals_scored_H'],'r-*',color = 'red',label = 'goals_scored_h')
    plt.title('Stats à domicile - Attaque')
    plt.legend(loc='upper right');
    st.pyplot(fig)
    
#Graphiques Buts encaissés

    y1 = ars_attack_realism_ratio['HOME - Ratio tirs vs tirs cadrés']
    y2 = ars_attack_realism_ratio['HOME - Ratio tirs - goals']
    y3 = ars_attack_realism_ratio['HOME - Ratio tirs cadrés - goals']

    Categories = ["Arsenal","Big5","team14","team20"]
    nb_categories = len(Categories)
    largeur_barre = floor(1*10/nb_categories)/10

    x1 = range(len(y1))
    x2 =[i + largeur_barre for i in x1]
    x3 =[i + 2*largeur_barre for i in x1]

    fig = plt.figure(figsize=(14,5))
    
    plt.bar(ars_attack_realism_ratio.index,y1,color='#73C2BE',label='HOME - Ratio tirs vs tirs cadrés',width = largeur_barre)
    plt.bar(x2,y2,color='#776885',label = 'HOME - Ratio tirs - goals',width = largeur_barre)
    plt.bar(x3,y3,color='#5F1A37',label = 'HOME - Ratio goals - tirs cadrés',width = largeur_barre)

    plt.title('Ratios à domicile - Attaque')
    plt.legend(loc='upper left')

    st.pyplot(fig)

    st.write("Nous notons une inconstance au niveau du réalisme face au goal puisque les ratios 'tirs cadrés vs goal's et 'tirs vs goals' augmentent de 10/11 jusqu’à la saison 12/13 puis décline à nouveau jusqu’à la saison 15/16. Nous notons que cette saison, dont les ratios sont au plus bas est pourtant la meilleure saison d’Arsenal sur ces 10 années.")
    st.write("Le réalisme face au goal s’est amélioré lors des 3 dernières saisons observées mais Arsenal n’a pas dépassé la 5ème place du championnat. Le nombre d’occasion créées est en déclin à compter de 18/19 avec une 8ème place de championnat lors de la saison 19/20.")


#Visualitation Ratio et Stat à l'extérieur

    fig = plt.figure(figsize=(14,5))

    #Stat à l'extérieur
    plt.plot(ars_attack_realism_ratio.index,ars_attack_realism_ratio['shots_a'],'r-*',color = '#AEDCC0',label = 'shots_a')
    plt.plot(range(len(ars_attack_realism_ratio.index)),ars_attack_realism_ratio['target_shots_a'],'r-*',color = '#7BD389',label = 'target_shots_a')
    plt.plot(range(len(ars_attack_realism_ratio.index)),ars_attack_realism_ratio['goals_scored_A'],'r-*',color = 'red',label = 'goals_scored_a')
    plt.title('Stats AWAY - Attaque')
    plt.legend(loc='upper right');
    st.pyplot(fig)

#Graphiques Buts encaissés
    y1 = ars_attack_realism_ratio['AWAY - Ratio tirs vs tirs cadrés']
    y2 = ars_attack_realism_ratio['AWAY - Ratio tirs - goals']
    y3 = ars_attack_realism_ratio['AWAY - Ratio tirs cadrés - goals']

    Categories = ["Arsenal","Big5","team14","team20"]
    nb_categories = len(Categories)
    largeur_barre = floor(1*10/nb_categories)/10

    x1 = range(len(y1))
    x2 =[i + largeur_barre for i in x1]
    x3 =[i + 2*largeur_barre for i in x1]

    fig = plt.figure(figsize=(14,14))

    fig = plt.figure(figsize=(14,5))
    plt.bar(ars_attack_realism_ratio.index,y1,color='#73C2BE',label='AWAY - Ratio tirs vs tirs cadrés',width = largeur_barre)
    plt.bar(x2,y2,color='#776885',label = 'AWAY - Ratio tirs - goals',width = largeur_barre)
    plt.bar(x3,y3,color='#5F1A37',label = 'AWAY - Ratio goals - tirs cadrés',width = largeur_barre)

    plt.title('Ratios AWAY - Attaque')
    plt.legend(loc='upper left')
    st.pyplot(fig)
    
    
    st.write("A l'extérieur, la tendance relative aux tirs indique un déclin encore plus marqué puisque les tirs passent de 300 au nombre de 200, ce qui représente une baisse d’environ 33%. Le déclin du nombre de goals à l’extérieur est quant à lui plus marqué qu’à domicile. La saison 16/17 dispose du meilleur ratio 'tirs-goals' mais le niveau de la Premier League était particulièrement élevé cette année-là. Ces dernières années, Arsenal s’est moins créé d’occasions durant les matchs à l’extérieur et le 'ratio goals vs tirs' a également diminué. Ceci confirme qu'Arsenal a été moins impactant lors de ses déplacements.")
    
    
    st.title("Analyse de l’aspect défensif")
    
    st.write("Nous voulons analyser ici les statistiques d’Arsenal à domicile afin de déterminer comment l’équipe se comporte en défense.")
    
    #Visualitation Ratio et Stat à domicile

    fig = plt.figure(figsize=(14,5))

    #Stat à domicile
    plt.plot(ars_defense_realism_ratio.index,ars_defense_realism_ratio['shots_conced_h'],'r-*',color = '#AEDCC0',label = 'shots_conced_h')
    plt.plot(range(len(ars_defense_realism_ratio.index)),ars_defense_realism_ratio['target_shots_conced_h'],'r-*',color = '#7BD389',label = 'target_shots_conced_h')
    plt.plot(range(len(ars_defense_realism_ratio.index)),ars_defense_realism_ratio['goals_conced_H'],'r-*',color = 'red',label = 'goals_conced_h')
    plt.title('Stats à domicile - Défense')
    plt.legend(loc='upper left');
    st.pyplot(fig)

    #Graphiques Buts encaissés
    y1 = ars_defense_realism_ratio['HOME_conced - Ratio tirs vs tirs cadrés']
    y2 = ars_defense_realism_ratio['HOME_conded - Ratio tirs - goals']
    y3 = ars_defense_realism_ratio['HOME_conced - Ratio tirs cadrés - goals']

    Categories = ["Arsenal","Big5","team14","team20"]
    nb_categories = len(Categories)
    largeur_barre = floor(1*10/nb_categories)/10

    x1 = range(len(y1))
    x2 =[i + largeur_barre for i in x1]
    x3 =[i + 2*largeur_barre for i in x1]

    fig = plt.figure(figsize=(14,5))

    plt.bar(ars_defense_realism_ratio.index,y1,color='#73C2BE',label='HOME_conced - Ratio tirs vs tirs cadrés',width = largeur_barre)
    plt.bar(x2,y2,color='#776885',label = 'HOME_conced - Ratio tirs - goals',width = largeur_barre)
    plt.bar(x3,y3,color='#5F1A37',label = 'HOME_conced - Ratio goals - tirs cadrés',width = largeur_barre)

    plt.title('Ratios à domicile - Défense')
    plt.legend(loc='upper left')
    st.pyplot(fig)
    
    
    st.write("Le nombre de tirs concédés est passé de 150 durant la saison 10/11 à environ 300 lors de la saison 19/20 ce qui correspond à une augmentation de près de 50%. Arsenal concède beaucoup plus d’occasions à domicile et particulièrement sur les 3 dernières saisons observées. Le nombre de tirs cadrés à lui aussi augmenté et le nombre de goals encaissés est resté relativement inconstant avec une augmentation lors de la dernière saison observée. La saison 15/16 dispose des ratios les plus faibles pour Arsenal lorsque l’on regarde les ratios 'tirs-goals' et 'ratio goals – tirs cadrés'. Cela indique très probablement des bonnes performances du gardien.")
    
    st.write("Les graphiques suivants analysent les résultats à l'extérieur")
    
    #Visualitation Ratio et Stat à l'extérieur

    fig = plt.figure(figsize=(14,5))

    #Stat à l'extérieur
    plt.plot(ars_defense_realism_ratio.index,ars_defense_realism_ratio['shots_conced_a'],'r-*',color = '#AEDCC0',label = 'shots_conced_a')
    plt.plot(range(len(ars_defense_realism_ratio.index)),ars_defense_realism_ratio['target_shots_conced_a'],'r-*',color = '#7BD389',label = 'target_shots_conced_a')
    plt.plot(range(len(ars_defense_realism_ratio.index)),ars_defense_realism_ratio['goals_conced_A'],'r-*',color = 'red',label = 'goals_conced_a')
    plt.title('Stats AWAY - Défense')
    plt.legend(loc='upper left');
    st.pyplot(fig)

    #Graphiques Buts encaissés
    y1 = ars_defense_realism_ratio['AWAY_conced - Ratio tirs vs tirs cadrés']
    y2 = ars_defense_realism_ratio['AWAY_conced - Ratio tirs - goals']
    y3 = ars_defense_realism_ratio['AWAY_conced - Ratio tirs cadrés - goals']

    Categories = ["Arsenal","Big5","team14","team20"]
    nb_categories = len(Categories)
    largeur_barre = floor(1*10/nb_categories)/10

    x1 = range(len(y1))
    x2 =[i + largeur_barre for i in x1]
    x3 =[i + 2*largeur_barre for i in x1]

    fig = plt.figure(figsize=(14,5))

    plt.bar(ars_defense_realism_ratio.index,y1,color='#73C2BE',label='AWAY_conced - Ratio tirs vs tirs cadrés',width = largeur_barre)
    plt.bar(x2,y2,color='#776885',label = 'AWAY_conced - Ratio tirs - goals',width = largeur_barre)
    plt.bar(x3,y3,color='#5F1A37',label = 'AWAY_conced - Ratio goals - tirs cadrés',width = largeur_barre)

    plt.title('Ratios AWAY - Défense')
    plt.legend(loc='upper left')
    st.pyplot(fig)
    
    st.write("Le nombre d’occasions concédées (tirs) indique une tendance haussière sur les 10 dernières saisons avec un pic lors de saison 18/19. Le nombre de but encaissés à l’extérieur a augmenté lors des saisons 14/15 jusqu’à la saison 18/19 avant de diminuer en 19/20. Nous observons d’ailleurs que le ratio 19/20 du nombre de buts encaissé a également diminué en 19/20.")
    st.write("**Conclusion sur l’analyse et la défense d’Arsenal :**")
    st.write("Sur une note générale, nous pouvons dire qu’Arsenal a été beaucoup moins impactant et a surtout beaucoup plus subi lors de ses matchs à domicile au vu de la tendance baissière du nombre d’occasions créées ainsi que la tendance haussière du nombre d’occasions concédées. Lors de ses déplacements, Arsenal peine à être percutant mais se défend mieux qu’à domicile.")
    
    
    st.title("Analyse des 'Clean Sheets'")
    
    st.write("Une équipe réalise une « Clean Sheet » lorsqu’elle n’encaisse aucun but dans le match.")
    st.write("L’analyse des « Clean Sheets » pour Arsenal est effectuée en comparaison avec les autres équipes du Big Six sur les 10 saisons.")

    st.write("Le graphique suivant analyse les matchs à domicile:")
    
#Graphique des CS à domicile

    fig = plt.figure(figsize = (14, 5))
    x = np.arange(10) 
    y1 = df_cs[df_cs['home_team']=='Arsenal'].groupby('season').sum()['clean_sheet_home']
    y2 = df_cs[df_cs['home_team']=='Tottenham Hotspur'].groupby('season').sum()['clean_sheet_home']
    y3 = df_cs[df_cs['home_team']=='Liverpool'].groupby('season').sum()['clean_sheet_home']
    y4 = df_cs[df_cs['home_team']=='Manchester City'].groupby('season').sum()['clean_sheet_home']
    y5 = df_cs[df_cs['home_team']=='Chelsea'].groupby('season').sum()['clean_sheet_home']
    y6 = df_cs[df_cs['home_team']=='Manchester United'].groupby('season').sum()['clean_sheet_home']
    width = 0.1

    plt.bar(x-0.25, y1, width, color='red',edgecolor='black') 
    plt.bar(x-0.15, y2, width, color='blue',edgecolor='black') 
    plt.bar(x-0.05, y3, width, color='green',edgecolor='black')
    plt.bar(x+0.05, y4, width, color='brown',edgecolor='black')
    plt.bar(x+0.15, y5, width, color='cyan',edgecolor='black')
    plt.bar(x+0.25, y6, width, color='yellow',edgecolor='black')

    plt.xticks(x, ['10/11', '11/12', '12/13', '13/14', '14/15', '15/16', '16/17', '17/18', '18/19', '19/20']) 
    plt.title('Nombre de "Clean Sheet" par saison - Big6 - Domicile')
    plt.ylim([0,13])
    plt.xlabel("Saisons")
    plt.ylabel("Clean Sheet") 
    plt.legend(['Arsenal', 'Tottenham', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United']) 
    st.pyplot(fig)    

    st.write("On observe ainsi qu’il n’y a pas de performance ou contre-performance systématique d’un club du Big Six. Le maxium est de 12 'CS' tous clubs confondus sur l’ensemble des saisons à domicile avec une rotation des clubs réalisant le plus haut et plus bas nombre.")
    st.write("Concernant Arsenal, il y a une stagnation entre 10/11 et 17/18 avec une contre-performance en 12/13 et une bonne performance en 13/14. On observe ensuite une baisse à partir de 17/18.")
    
    st.write("Matchs à l'extérieur:")
    #Graphique des CS à l'extérieur

    fig = plt.figure(figsize = (14, 5))
    x = np.arange(10) 
    y1 = df_cs[df_cs['away_team']=='Arsenal'].groupby('season').sum()['clean_sheet_home']
    y2 = df_cs[df_cs['away_team']=='Tottenham Hotspur'].groupby('season').sum()['clean_sheet_home']
    y3 = df_cs[df_cs['away_team']=='Liverpool'].groupby('season').sum()['clean_sheet_home']
    y4 = df_cs[df_cs['away_team']=='Manchester City'].groupby('season').sum()['clean_sheet_home']
    y5 = df_cs[df_cs['away_team']=='Chelsea'].groupby('season').sum()['clean_sheet_home']
    y6 = df_cs[df_cs['away_team']=='Manchester United'].groupby('season').sum()['clean_sheet_home']
    width = 0.1

    plt.bar(x-0.25, y1, width, color='red',edgecolor='black') 
    plt.bar(x-0.15, y2, width, color='blue',edgecolor='black') 
    plt.bar(x-0.05, y3, width, color='green',edgecolor='black')
    plt.bar(x+0.05, y4, width, color='brown',edgecolor='black')
    plt.bar(x+0.15, y5, width, color='cyan',edgecolor='black')
    plt.bar(x+0.25, y6, width, color='yellow',edgecolor='black')

    plt.xticks(x, ['10/11', '11/12', '12/13', '13/14', '14/15', '15/16', '16/17', '17/18', '18/19', '19/20']) 
    plt.title('Nombre de "Clean Sheet" par saison - Big6 - Extérieur')
    plt.ylim([0,13])
    plt.xlabel("Saison")
    plt.ylabel("Clean Sheet") 
    plt.legend(['Arsenal', 'Tottenham', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United']) 
    st.pyplot(fig)
    
    st.write("Les 'Clean-Sheets' réalisées à l’extérieur sont inférieures d'environ 50% à celles réalisées à domicile pour l’ensemble des équipes. Pour Arsenal, il ressort une progression globaleau fil des saisons et même une meilleure performance en comparaison des autres équipes sur les dernières saisons.")
    
    
    
    st.title("Analyse de la possession de balle")
    
    st.write("La visualisation suivante se concentre sur l’analyse de la possession de balle d’Arsenal.")
    
    #Visualitation Possession de balle

    fig = plt.figure(figsize=(14,5))

    #Stat à domicile
    plt.plot(ars_attack_realism_ratio.index,ars_attack_realism_ratio['average_possession_h'],'r-*',color = '#08B2E3',label = 'average_possession_h')
    plt.plot(range(len(ars_attack_realism_ratio.index)),ars_attack_realism_ratio['average_possession_a'],'r-*',color = '#EFE9F4',label = 'average_possession_a')
    plt.plot(range(len(ars_attack_realism_ratio.index)),ars_attack_realism_ratio['average_possession'],'r-*',color = '#484D6D',label = 'average_possession')
    
    st.title('Possession de balle')
    st.pyplot(fig)
    
    st.write("Premièrement, nous constatons qu’Arsenal reste majoritairement en possession de la balle, tant pour les matchs à domicile que pour les matchs à l’extérieurs. D’après nos recherches, cette philosophie de possession de balle diffère avec l’âge d’or d’Arsenal durant lequel l’équipe possédait moins la balle. Les statistiques plus détaillées à ce sujet n’ont pu être trouvée.")
    st.write("La possession de balle a particulièrement chuté à partir de la saison 17/18 et nous notons qu’Arsenal dispose d’une meilleure possession de balle à l’extérieur qu’à domicile lors de l’ultime saison observée ce qui rejoint le point précédemment constaté à savoir qu’Arsenal subissait plus à domicile qu’à l’extérieur lors de sa dernière saison.")
    st.write("Nous notons également que la saison 15/16 qui représente le meilleur classement d’Arsenal ne dispose cependant pas pour autant de la meilleure possession de balle observée puisqu’elle se situe environ à 58% contre un maximum de 62% et un minimum de 54%.")
    
    st.title('Comparaison du score à la mi-temps et en fin de match')
    
    st.write("L’objectif de cette partie est de comparer le score à la mi-temps et en fin de période réglementaire des matchs d’Arsenal.")
    st.write("Cette comparaison s’appuie sur la répartition en 9 catégories des combinaisons victoire/nul/défaite de la première manche et du match sur les 10 saisons étudiées de la décennie 2010-20. Les matchs à domicile et à l’extérieur sont séparés en 2 graphiques.")
    
    st.write("Le graphique suivant concerne les matchs à domicile :")
        
    #Graphique à domicile

    fields = ['win_win_home', 'draw_win_home', 'lose_win_home',
             'win_draw_home', 'draw_draw_home', 'lose_draw_home',
             'win_lose_home', 'draw_lose_home', 'lose_lose_home']

    colors = ['#03045e', '#023e8a', '#0077b6',
             '#6c757d', '#adb5bd', '#ced4da',
             '#ffea00', '#ffaa00', '#ff7b00']

    labels = ['Victoire / Victoire', 'Nul / Victoire', 'Défaite / Victoire',
             'Victoire / Nul', 'Nul / Nul', 'Défaite / Nul',
             'Victoire / Défaite', 'Nul / Défaite', 'Défaite / Défaite']

    # figure and axis
    fig, ax = plt.subplots(1, figsize=(20, 12))

    # plot bars
    left = len(HT_FT_season_grouped_Ars) * [0]
    for idx, name in enumerate(fields):
        plt.barh(HT_FT_season_grouped_Ars.index, HT_FT_season_grouped_Ars[name], left = left, color=colors[idx])
        left = left + HT_FT_season_grouped_Ars[name]

    # title, legend, labels
    plt.title('Comparaison du résultat de la première match et du match - (1ere manche / match) - Domicile\n\n', loc='left')
    plt.legend(labels, bbox_to_anchor=([0.05, 1, 0, 0]), ncol=9, frameon=False)
    plt.xlabel('Somme des résultats par catégorie')
    plt.ylabel('Saison')

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # adjust limits and draw grid lines
    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    plt.xlim(0,19,5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    st.pyplot(fig)
    
    st.write("On observe globalement une baisse des performances d’Arsenal à domicile. Cela s’illustre depuis 2017 notamment par sa difficulté à s’imposer « d’entrée de jeu », représentée par la baisse de la catégorie « Victoire / Victoire » (5 « V/V » en 19/20, 11 en 17/18).")
    st.write("Par ailleurs, la dernière saison met en lumière la difficulté d’Arsenal à maintenir les bons résultats de la première mi-temps avec une augmentation singulière des « Victoire / Nul » et « Victoire / Défaite » (4 « V/N » et 1 « V/D »).")
    
    st.write("Graphique des matchs à l'extérieur:")
    
    #Graphique à l'extérieure

    fields = ['win_win_away', 'draw_win_away', 'lose_win_away',
             'win_draw_away', 'draw_draw_away', 'lose_draw_away',
             'win_lose_away', 'draw_lose_away', 'lose_lose_away']

    colors = ['#03045e', '#023e8a', '#0077b6',
             '#6c757d', '#adb5bd', '#ced4da',
             '#ffea00', '#ffaa00', '#ff7b00']

    labels = ['Victoire / Victoire', 'Nul / Victoire', 'Défaite / Victoire',
             'Victoire / Nul', 'Nul / Nul', 'Défaite / Nul',
             'Victoire / Défaite', 'Nul / Défaite', 'Défaite / Défaite']

    # figure and axis
    fig, ax = plt.subplots(1, figsize=(20, 12))

    # plot bars
    left = len(HT_FT_season_grouped_Ars) * [0]
    for idx, name in enumerate(fields):
        plt.barh(HT_FT_season_grouped_Ars.index, HT_FT_season_grouped_Ars[name], left = left, color=colors[idx])
        left = left + HT_FT_season_grouped_Ars[name]

    # title, legend, labels
    plt.title('Comparaison du résultat de la première match et du match - (1ere manche / match) - Extérieure\n\n', loc='left')
    plt.legend(labels, bbox_to_anchor=([0.05, 1, 0, 0]), ncol=9, frameon=False)
    plt.xlabel('Somme des résultats par catégorie')
    plt.ylabel('Saison')

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # adjust limits and draw grid lines

    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    plt.xlim(0,19)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    st.pyplot(fig)

    st.write("On observe à nouveau la difficulté permanente d’Arsenal lors de ces saisons de s’imposer à l’extérieur.")
    st.write("Les victoires et nuls sont sensiblement similaires au cours des 10 saisons, avec cependant une augmentation sur les 2 dernières années (7 nuls dont 4 « Défaite / Nul » en 19/20).")
    st.write("Cette augmentation des performances d’Arsenal en dernière année concerne également les défaites avec une diminution des catégories « Nul / Défaite » et « Défaite / Défaite » sur les 2 dernières saisons. Cette hausse des performances à l’extérieur est ainsi en contradiction avec la diminution des performances à domicile lors de cette saison.")
    st.write("Par ailleurs, la saison 16/17 est singulière dans l’analyse par la forte transformation des nuls en victoire à domicile (9 « N/V ») mais également des nuls en défaite à l’extérieur (9 « N/D »).")
    
    st.title("Points gagnés par mois au cours de l'avancement du championnat")
    
    
    st.write("L'objectif est de visualiser les points acquis à domicile et à l'extérieur au cours de l'avancement du championnat afin de visualiser une éventuelle bonne ou mauvaise performance à une période spécifique de la saison.")
    st.write("L'échelle de 1 à 12 représente l'avancement du championnant au cours de la saison. Par exemple, pour la saison 10/11,  Juillet/10 est le mois 1 et Juin/11, le mois 12.")
    st.write("Finalement, 'la longueur du rayon' des 2 graphiques est différente afin de mettre en lumière les disparités entre les saisons et non la différence de points acquis à domicile et à l'extérieur.")        
    
    st.write("**Points à domicile:**")
    
    fig = px.bar_polar(df_WP_grouped_home_team, 
                   r=df_WP_grouped_home_team['point_home'], 
                   theta=df_WP_grouped_home_team['season'], 
                   color=df_WP_grouped_home_team['Month_chrono'],
                      range_r=[0,45])

    st.plotly_chart(fig)
    
    st.write("Arsenal a acquis régulièrement des points à domicile entre 11/12 et 15/16. Le club a fait de bonnes fin de saison de 16/17 à 18/19 et un bon début de saison en 19/20.")

    st.write("**Points à l'extérieur:**")
    
    fig = px.bar_polar(df_WP_grouped_away_team, 
                   r=df_WP_grouped_away_team['point_away'], 
                   theta=df_WP_grouped_away_team['season'], 
                   color=df_WP_grouped_away_team['Month_chrono'],
                      range_r=[0,35])

    st.plotly_chart(fig)
    
    st.write("La régularité des points acquis entre 11/12 et 15/16 est un peu plus mitigé à l'extérieur qu'à domicile mais reste visible de 12/13 à 14/15.")
    
    st.title("L’impact des rencontres du Big Six")
    
    st.write("Le graphique, ci-dessous, met en avant l’importance des rencontres qui opposent les équipes du Big Six. Nous avons sur les 10 saisons, le classement des équipes en fonction du nombre de points gagnés lors des rencontres du Big Six. Chaque équipe a 10 matchs par saison contre ces équipes. Le nombre maximal de point possible est donc de 30 points lors des 10 rencontres.")
    
    st.write("Ce graphique est très important car sur les 10 saisons, à seulement 5 reprises une équipe du Big Six ne faisait pas partie des 6 premiers.")
    
    recap_big_5=recap_big_5.sort_values(by=['Classement'])
    
    fig = sns.catplot(x='Classement',y='Points gagnés',kind='swarm',hue='Equipe',data=recap_big_5,s=8, height=4, aspect=7/4);  
    st.pyplot(fig)
    
    st.write("Quelques tendances sont visibles sur ce graphique :")
    st.write("- Pour être champion, il faut gagner au moins 16 points lors de ces rencontres.")
    st.write("- On ne peut pas être sur le podium si l’on ne gagne pas un minimum de 10 points.")
    st.write("- A contrario, remporter au minimum 18 points sur ces rencontres vous permet d’être dans les 3 premiers.")
    st.write("- Lorsque que l’on gagne plus de 15 points, on se retrouve dans les 6 premiers du championnat.")

    st.write("Zoom sur Arsenal :")
    
    recap_arsenal=recap_ars_big5.loc[recap_ars_big5['Equipe']=='Arsenal']
    fig = sns.catplot(x=list(recap_arsenal.index),y='Points gagnés',kind='swarm',hue='Classement',data=recap_arsenal,s=10, height=4, aspect=7/4);
    st.pyplot(fig)
    
    st.write("Le graphique précédent permet de visualiser les points gagnés par Arsenal lors de ces rencontres au fil des saisons:") 
    st.write("- Arsenal gagne peu de points lors de ses rencontres. Ils n’ont jamais gagné plus de 13 points/30.")
    st.write("- Les 4 dernières saisons sont les pires classements d’Arsenal. Il s’agit également du pire nombres de points gagnés contre le Big Six, excepté 18/19.")
    st.write("**Constat :**")
    st.write("- Arsenal ne remporte pas des points cruciaux contre les équipes du Big Six pour performer en Premier League.")

    
    st.title("Les joueurs clefs")
    
    st.write("Arsenal a eu une période où elle a été au sommet de toute son histoire. De la saison 1997/1998 à 2003/2004, l’équipe a gagné 3 championnats sur 7 possibles et a fini 2ème sur le reste des saisons. Ils ont inscrit l’équipe dans la légende du football mondial en devenant la première équipe à remporter un championnat sans perdre le moindre match.")
    st.write("L’équipe des invincibles était composé de joueurs stars. Le meilleur buteur sur 5 des 7 saisons, Thierry Henry. Le créateur qui a dominé le classement des meilleurs passeurs sur les dernières saisons, Robert Pires. Et le légendaire gardien anglais, David Seaman.")
    st.write("Les prochains graphiques nous permettront de comprendre si l’équipe d’Arsenal a toujours ces joueurs clefs qui permettent à Arsenal de faire peur aux équipes adverses et surtout qui permettent de débloquer des matchs à eux seul.")
    
    st.write("**Buteurs**")
    #Graphique des meilleurs buteurs
    
    fig = plt.figure(figsize = (14, 5))
    plt.plot(stat_joueur_10_20.index,stat_joueur_10_20['Meilleur buteur Arsenal'],'#95190C', linewidth=3, label="Meilleur buteur d'Arsenal")
    plt.plot(stat_joueur_10_20.index,stat_joueur_10_20['Meilleur buteur de la saison'],'#044B7F', linewidth=3,label='Meilleur buteur de la saison')
    plt.axhline(y=stat_joueur_ars97_04['Meilleur buteur'].mean(),color='#E3B505',linestyle='dashed',label="Moyenne des meilleurs buteurs d'Arsenal 97 à 04")
    plt.xlabel('Saison')
    plt.ylabel('Buts marqués')
    plt.legend();
    st.pyplot(fig)
    
    st.write("Arsenal a eu deux fois le meilleur buteur du championnat sur 10 saisons (11/12 et 18/19). Ils manquent de régularité chez leurs buteurs. L’écart avec le meilleur buteur de la saison est de plus de 10 buts sur certaines saisons. Ils n’ont plus de buteurs capables d’être sur une moyenne de 21 buts par matchs comme c’était le cas des saisons 97/98 à 03/04. **Il manque un vrai buteur régulier au fil des saisons**.")
    
    st.write("**Passeurs**")
    
    #Graphique des meilleurs passeurs
    
    fig = plt.figure(figsize = (14, 5))
    plt.plot(stat_joueur_10_20.index,stat_joueur_10_20['Meilleur passeur Arsenal'],'#95190C', linewidth=3, label="Meilleur passeur d'Arsenal")
    plt.plot(stat_joueur_10_20.index,stat_joueur_10_20['Meilleur passeur de la saison'],'#044B7F', linewidth=3,label='Meilleur passeur de la saison')
    plt.axhline(y=stat_joueur_ars97_04['Meilleur passeur'].mean(),color='#E3B505',linestyle='dashed',label="Moyenne des meilleurs passeurs d'Arsenal 97 à 04")
    plt.ylabel('Nombres de passes décisives')
    plt.legend();
    st.pyplot(fig)
    
    st.write("Arsenal a eu le meilleur passeur décisif du championnat sur la saison 15/16. On voit par ailleurs qu’Arsenal a toujours eu un joueur capable de faire des passes décisives sur les saisons. A partir de 16/17, on visualise bien que ce n’est plus le cas.**Une des explications du déclin d’Arsenal à partir de 16/17 s’explique également sur le fait qu’ils n’ont plus de joueurs autant impactant et capables de distribuer un minimum de 10 passes décisives par saison**.")
    
    st.write("**Gardiens**")
    #Graphique des meilleurs gardiens
    
    fig = plt.figure(figsize = (14, 5))
    plt.plot(stat_joueur_10_20.index,stat_joueur_10_20['Gardien Arsenal'],'#95190C', linewidth=3, label="Gardien Arsenal")
    plt.plot(stat_joueur_10_20.index,stat_joueur_10_20['Meilleur gardien de la saison'],'#044B7F', linewidth=3,label='Moyenne Big5')
    plt.axhline(y=stat_joueur_ars97_04['Gardien'].mean(),color='#E3B505',linestyle='dashed',label="Moyenne des gardiens d'arsenal de 97 à 04")
    plt.xlabel('Saison')
    plt.ylabel('Buts sencaissés')
    plt.legend();
    st.pyplot(fig)
    
    st.write("Les gardiens d’Arsenal encaissent trop de buts si l’on compare à la moyenne de 97/98 à 03/04 et cette difficulté s’observe également sur les 4 dernières saisons. Malgré que l’on parle de buts encaissés et donc de défense, **on peut souligner qu’aucun gardien s’est mis en lumière en faisant une grosse saison, le mettant à l’honneur**.")
    
   
    st.title("Analyse des investisseurs")
   
    # Plot
    fig, ax = plt.subplots(figsize=(14,5), dpi= 80)
    ax.vlines(x=big5_ownership.index, ymin=0, ymax=big5_ownership['Worth'], color='firebrick', alpha=0.7, linewidth=2)
    ax.scatter(x=big5_ownership.index, y=big5_ownership['Worth'], s=150, color='firebrick', alpha=0.7)


    # Title, Label, Ticks and Ylim
    ax.set_title('Who has got the biggest?', fontdict={'size':22})
    ax.set_ylabel('Worth')
    ax.set_xticks(big5_ownership.index)
    ax.set_xticklabels(big5_ownership.Owners.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
    ax.set_ylim(0, 25000)
    st.pyplot(fig)
    
    st.write("Le propriétaire le plus riche de la Premier League est le Sheikh Mansour qui possède Man City avec une fortune estimée à 23£ milliards tandis que le propriétaire d’Arsenal, l’homme d’affaires Stan Kroenke s’élève au 3ème rang avec une fortune d’environ 6.8£ milliards.")

    st.write("D’après nos lectures des états financiers d’Arsenal, la stratégie suivie par le Board est celle d’avoir un club autosuffisant dont une partie des recettes sont réinvesties.")

    st.write("Nous avons ainsi web scrapé les montants directement investis par les propriétaires de la Premier League afin de réaliser une analyse comparative basée sur un ratio que nous avons établi à savoir le montant investi par rapport à la fortune du propriétaire.")
    
    
  # Plot investissement

    fig, ax = plt.subplots(figsize=(14,5), dpi= 80)     
    ax.vlines(x=big5_ownership_wet.index, ymin=0, ymax=big5_ownership_wet['Direct Financing / Total Worth ratio'], color='firebrick', alpha=0.7, linewidth=2)
    ax.scatter(x=big5_ownership_wet.index, y=big5_ownership_wet['Direct Financing / Total Worth ratio'], s=150, color='firebrick', alpha=0.7)


    # Title, Label, Ticks and Ylim
    ax.set_title('Who gets the most wet?', fontdict={'size':22})
    ax.set_ylabel('Ratio Direct Financing / Total Worth')
    ax.set_xticks(big5_ownership_wet.index)
    ax.set_xticklabels(big5_ownership_wet.Owners.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
    ax.set_ylim(0,0.1)
    st.pyplot(fig)
    
    st.write("Les propriétaires de Manchester United sont ceux qui ont le plus investis au vu de la taille de leur fortune avec environ 8% investis lors des 10 dernières années par rapport à leur fortune à ce jour.") 
    st.write("Stan Kroenke se retrouve loin en dernière place avec moins de 0.2% de la valeur actuelle de sa fortune investis lors des 10 dernières années.") 
    st.write("Nous notons que le propriétaire de Tottenham n’a également pas beaucoup investi et que la différence avec les 4 autres clubs du Big Six est significative et que seul ces 4 équipes (à l’exception de Leicester) ont remporté la Premier League au cours des 10 dernières saisons.")
             
    st.write("**Effet Kroenke**")
    st.write("Les recherches effectuées montrent que Stan Kroenke est entré au club lors de la saison 06/07 à hauteur de près de 10% du capital-actions. Il est devenu l’actionnaire principal lorsque Peter Denis Hill-Wood a vendu ses parts suite à des problèmes de santé. L’ancien actionnaire majoritaire était la 3ème génération aux rennes d’Arsenal.")
    st.write("Dès l’entrée au capital de Stan Kroenke, le club a enregistré une baisse de performance qui s’est accentuée avec le temps, résultant de la politique d’autosuffisance du club notamment induite suite aux investissements massifs pour la construction de l’Emirates Stadium en 2006. Nos recherches d’articles sur le web nous ont d’ailleurs montré qu’Arsène Wenger a dû adapter son style de jeu qui devenu moins porté sur le physique, l’agressivité et les contre-attaques pour s’inspirer du schéma victorieux de l’Espagne plus basé sur la possession de balle et la créativité avec Cesc Fabregas en remplacement de Patrick Viera. Le 4-4-2 a petit à petit disparu au détriment du 4-5-1 et dès lors Arsenal a perdu son ADN qui avait fait d’elle une équipe d’invincibles.")

    st.title('**Réponse à la problématique**')

    st.write("Le football est une histoire de passion. Arsenal, avec 13 titres de Premier League est un club qui s’inscrit dans ce patrimoine culturel. Le club était également un patrimoine familial depuis 3 générations mais cet héritage a pris fin lorsque Peter Hill-Wood, le dernier descendant a vendu ses parts à l’homme d’affaires Stan Kroenke en 2011 suite à des problèmes de santé.")
      
    st.write("Comme nous l’avons vu précédemment, Kroenke bien que parmi les propriétaires les plus fortunés de la Premier League, n’investit que très peu dans le club ce qui diffère grandement des autres propriétaires du Big Six (excepté pour Tottenham).")
                 
    st.write("Arsène Wenger, entraineur arrivé au club en 1996 a rapidement su trouver le chemin du succès et a réussi à créer la fameuse équipe des invincibles, à son apogée en 2004. Son équipe était réputée comme agressive, physique et qui savait repartir en contre-attaque pour aller marquer des buts.")
                         
    st.write("La politique du club décidée par Kroenke n’a pas permis à Wenger d’exploiter pleinement son schéma tactique et ce dernier a opéré des changements avec les moyens imposés et c’est alors que le déclin a commencé. Dès lors, l’équipe des invincibles a été dissoute avec la perte d’éléments clefs tels que Patrick Viera, Thierry Henry, Robert Pires ou encore David Seaman. Arsenal ne s’est alors jamais donné les moyens de remplacer ces éléments clefs dans la durée ce qui a lourdement impacté les performances de l’équipe.")
                                  
    st.write("L’écart avec les autres équipes du Big Six s’est creusé, et comme vu lors de notre analyse, les rencontres directes contres ces équipes sont cruciales afin d’espérer remporter le graal. La data récupérée indique qu’Arsenal peine à être percutante durant ces grandes rencontres puisque lors des 10 dernières saisons, elle n’a jamais réussi à repartir de ces matchs avec plus de 13 points.")                                                               
    st.write("Sortir perdant de ces importants rendez-vous peut peser sur le moral d’une équipe et nos données nous indique qu’Arsenal, au fil de la dernière décennie, est devenue moins percutante (moins d’occasions créées) tant à domicile qu’en déplacement et surtout encaisse plus de buts.")                        
                                                      
    st.write("Notre travail nous pousse à penser qu’Arsenal est entré dans un cercle vicieux qui a été initié par le manque de soutien financier mais pas uniquement puisque Kroenke ne se déplace que très rarement au stade. Ce manque d’investissement direct a induit une baisse de la compétitivité de l’équipe et cette dernière qui peine à se qualifier pour les compétitions européennes génère dès lors moins de recettes qui ne pourront ainsi être réinvesties dans le modèle d’autosuffisance insufflé par Stan Kroenke.")
                                                              
    st.write("A l’image du PSG et de Manchester City, souvent sous le feu des projecteurs avec leurs dépenses astronomiques sur le marché des transferts (Neymar 222 millions€, Grealish 118 millions€), une nouvelle ère a vu le jour avec des joueurs toujours plus chers et prétendant toujours à des salaires plus hauts.")
                                                                       
    st.write("Ainsi, est-ce qu’un modèle d’autosuffisance du club tel qu’instauré chez les Gunners permet-il encore de rester sur le devant de la scène internationale d’aujourd’hui ?")
                 


#_______________________________________________________________________________________________________________________________________
