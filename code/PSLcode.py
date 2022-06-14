#importing libraries

import os
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.graph_objs as go
import ipywidgets as wg
from scipy import stats
from IPython.display import display
from urllib.request import urlopen
from plotly.subplots import make_subplots
import plotly.express as px



#importing data file
psldata = pd.ExcelFile(r'/Users/Amar/Documents/WD/projects/PSLVisualisation/data/All deliveries.xlsx')
df = pd.read_excel(psldata)


##cleaning data
#Replaced Nas
df.iloc[:,-2:] = df.iloc[:,-2:].fillna('No dismisal')



df['wicket'] = df['Dismisal_kind'].apply(lambda x: 1 if x != "No dismissal" else 0)
df['Six'] = df['Batsman_runs'].map(lambda x: 1 if x == 6 else 0)
df['Four'] = df['Batsman_runs'].map(lambda x: 1 if x == 4 else 0)

#Batsman dataframe

#Ball-by-ball detail for each batsman

df_bat_byball = pd.merge(df.groupby(['Batsman_strike','Match_id']).sum().reset_index()[['Batsman_strike','Match_id','Batsman_runs']], df.groupby(['Batsman_strike', 'Match_id']).count().reset_index()[['Batsman_strike','Match_id', 'Ball']].rename(columns = {'Ball': 'Balls_played'}), on = ['Batsman_strike', 'Match_id'])



##Batting datafrane
df_bat = pd.merge(pd.merge(df_bat_byball.groupby('Batsman_strike').mean().reset_index()[['Batsman_strike','Batsman_runs', 'Balls_played']].rename(columns = {'Batsman_runs' : 'AvgRuns_match' , 'Balls_played': 'AvgBalls_match'}), df[['Batsman_strike', 'Batting_team']].drop_duplicates(keep = 'first'), on = 'Batsman_strike') , pd.merge(df.groupby(['Batsman_strike']).sum().reset_index()[['Batsman_strike', 'Batsman_runs', 'Six', 'Four']].rename(columns = {'Batsman_runs': 'TotalRuns', 'Six':'TotalSixes', 'Four':'TotalFours'}), df.groupby(['Batsman_strike']).count().reset_index()[['Batsman_strike' , 'Ball']].rename(columns = {'Ball': 'Balls_played'}), on ='Batsman_strike'), on = 'Batsman_strike')                         

df_bat['Runs_sq'] = 0
df_bat['Matches_played'] = 0


#To calculate variance
for i in range(len(df_bat)):
    for j in range(len(df_bat_byball)):
        if df_bat.iloc[i]['Batsman_strike'] == df_bat_byball.iloc[j]['Batsman_strike']:
            df_bat['Runs_sq'].iloc[i] += df_bat_byball.iloc[j]['Batsman_runs'] **2
            df_bat['Matches_played'].iloc[i] += 1


#Strke rate, % boundary and Variance
df_bat['Strike_rate'] = df_bat['TotalRuns'] * 100 /df_bat['Balls_played']
df_bat['PercentageRuns_boundary'] = (df_bat['TotalSixes']* 6 + df_bat['TotalFours'] * 4) * 100/df_bat['TotalRuns']

NumberMatches = df_bat['Matches_played']
AvgRuns = df_bat['AvgRuns_match']
df_bat["Var"] = 0

##Batsman Scatter

def quadrant_chart(x, y, team,xaxis,yaxis, title):
    data = pd.DataFrame({'x': x, 'y': y, 'team':team})
    data = data.sort_values(by='team', ascending=True)
    # calculate averages up front to avoid repeated calculations
    y_avg = data['y'].mean()
    x_avg = data['x'].mean()
    figQuad = px.scatter(x=data['x'], y=data['y'], color= data['team'])
    figQuad.add_hline(y= y_avg, line_color='red', line_width=1)
    figQuad.add_vline(x = x_avg, line_color='green', line_width=1)
    figQuad.update_layout(title=title, yaxis=yaxis,xaxis=xaxis,legend_title="Teams")

    plotly.offline.plot(figQuad, include_plotlyjs = False, output_type = 'div')
    # figQuad.show()

quadrant_chart(
    x=df_bat['AvgRuns_match'],
    y=df_bat['Strike_rate'],
    team=df_bat['Batting_team'],
    xaxis=dict(title="Average Batting Runs"),
    yaxis=dict(title="Strike Rate"),
    title="Batting runs vs Batting strike rate for each team")


##Batting boxplot

trace0 = go.Box(y = df_bat.loc[df_bat['Batting_team'] == 'Islamabad United', 'AvgRuns_match'], name = 'Islamabad United', boxmean = 'sd', boxpoints = 'all')
trace1 = go.Box(y = df_bat.loc[df_bat['Batting_team'] == 'Karachi Kings', 'AvgRuns_match'], name = 'Karachi Kings' , boxmean = 'sd', boxpoints = 'all')
trace2 = go.Box(y = df_bat.loc[df_bat['Batting_team'] == 'Lahore Qalandars', 'AvgRuns_match'], name = 'Lahore Qalandars' , boxmean = 'sd', boxpoints = 'all')
trace3 = go.Box(y = df_bat.loc[df_bat['Batting_team'] == 'Peshawar Zalmi', 'AvgRuns_match'], name = 'Peshawar Zalmi', boxmean = 'sd', boxpoints = 'all')
trace4 = go.Box(y = df_bat.loc[df_bat['Batting_team'] == 'Quetta Gladiators', 'AvgRuns_match'], name = 'Quetta Gladiators', boxmean = 'sd', boxpoints = 'all')

data = [trace0, trace1, trace2, trace3, trace4]

figBatBox = go.Figure(data = data)

figBatBox.update_layout(
    title=dict(text = "Batting Runs distribution for each team" ),
    xaxis=dict(title="Teams"),
    yaxis_title="Average runs scored",
    legend_title="Teams")

# figBatBox.show()
plotly.offline.plot(figBatBox, include_plotlyjs = False, output_type = 'div')

##Bowling
## Defining bowler metrics

df['Caught'] = df['Dismisal_kind'].apply(lambda x: 1 if x == "caught" else 0)
df['Bowled'] = df['Dismisal_kind'].apply(lambda x: 1 if x == "bowled" else 0)
df['Lbw'] = df['Dismisal_kind'].apply(lambda x: 1 if x == "lbw" else 0)
df['Runout'] = df['Dismisal_kind'].apply(lambda x: 1 if x == "runout" else 0)
df['Stump'] = df['Dismisal_kind'].apply(lambda x: 1 if x == "stump" else 0)
df['Caught_bowled'] = df['Dismisal_kind'].apply(lambda x: 1 if x == "caught and bowled" else 0)


## Creating a bowler dataframe
##By each match
df_bowl_match = pd.merge(
    df.groupby(['Bowler', 'Match_id']).sum().reset_index().filter(regex = "[Innings,Over,Ball]"), 
    df.groupby(['Bowler', 'Match_id']).count().reset_index()[['Bowler','Match_id', 'Ball']].rename(columns =        {'Ball': 'TotalDeliveries'}
    ), on = ['Bowler', 'Match_id'])

#Avg deliveries bye ach match
df_bowl_matchAvg = pd.merge(
    df.groupby(['Bowler', 'Match_id']).sum().reset_index().filter(regex = "[^Match_id,Innings,Over,Ball]") ,       df.groupby(['Bowler', 'Match_id']).count().reset_index()[['Bowler', 'Ball']].rename(columns = {'Ball': 'Delieveries_bowled'}), on = 'Bowler').groupby('Bowler').mean().reset_index()

#Avg tot runs by each match
df_bowl_Avg_Tot = pd.merge(
    df_bowl_match.groupby('Bowler').sum().reset_index().filter(regex = '[^Match_id,Innings,Over,Ball]'),           df_bowl_matchAvg.filter(regex = "[^Caught,Bowled,Stump,Caught_bowled, Lbw, Runout]").rename(columns =      {'Delieveries_bowled':'AvgDeliveries_bowled','Batsman_runs': 'AvgBatsman_runs', 'Extra_runs': 'AvgExtra_runs' , 'Six': 'AvgSix', 'Four': 'AvgFour', 'wicket' : 'AvgWicket'}),
 on = 'Bowler')

#Bowler Economy
dfecon = df.groupby(['Bowler','Match_id', 'Over']).sum().reset_index().groupby('Bowler').mean().reset_index()[['Bowler','Batsman_runs', 'Extra_runs']]
dfecon['Economy'] = dfecon['Batsman_runs'] + dfecon['Extra_runs']

#Bowler final dataframe
df_bowler_summ = pd.merge(df_bowl_Avg_Tot.loc[df_bowl_Avg_Tot['TotalDeliveries'] > 50], dfecon[['Bowler','Economy']], on = 'Bowler')


## Bowler variance
df_bowler_summ['Runs_sq'] = 0
df_bowler_summ['Matches_played'] = 0
for i in range(len(df_bowler_summ)):
    for j in range(len(df_bowl_match)):
        if df_bowler_summ.iloc[i]['Bowler'] == df_bowl_match.iloc[j]['Bowler']:
            df_bowler_summ['Runs_sq'].iloc[i] += (df_bowl_match.iloc[j]['Batsman_runs'] +  df_bowl_match.iloc[j]['Extra_runs'])**2
            df_bowler_summ['Matches_played'].iloc[i] += 1

AvgRuns = df_bowler_summ['AvgBatsman_runs']+df_bowler_summ['AvgExtra_runs']
NumberMatches = df_bowler_summ['Matches_played']

df_bowler_summ['Var'] = 0

for i in range(len(df_bowler_summ)):
    if NumberMatches.loc[i] > 0:
        df_bowler_summ["Var"].loc[i] = (df_bowler_summ['Runs_sq'].loc[i] - NumberMatches.loc[i]*(AvgRuns.loc[i]**2))/(NumberMatches.loc[i] - 1)

df_bowler = pd.merge(df_bowler_summ, df[['Bowler', 'Bowling_team']].drop_duplicates(keep = 'first'), on = 'Bowler')
df_bowler.dropna(inplace=True)

##bowler scatter

quadrant_chart(
    x=df_bowler['Economy'],
    y=df_bowler['wicket'],
    team=df_bowler['Bowling_team'],
    xaxis=dict(title="Average number of wickets"),
    yaxis=dict(title="Economy"),
    title=dict(text = "Bowler wicket vs Bowler Economy for each team")
)

##bowler box

trace0 = go.Box(y = df_bowler.loc[df_bowler['Bowling_team'] == 'Islamabad United', 'Economy'], name = 'Islamabad United', boxmean = 'sd', boxpoints = 'all')
trace1 = go.Box(y = df_bowler.loc[df_bowler['Bowling_team'] == 'Karachi Kings', 'Economy'], name = 'Karachi Kings' , boxmean = 'sd', boxpoints = 'all')
trace2 = go.Box(y = df_bowler.loc[df_bowler['Bowling_team'] == 'Lahore Qalandars', 'Economy'], name = 'Lahore Qalandars' , boxmean = 'sd', boxpoints = 'all')
trace3 = go.Box(y = df_bowler.loc[df_bowler['Bowling_team'] == 'Peshawar Zalmi', 'Economy'], name = 'Peshawar Zalmi', boxmean = 'sd', boxpoints = 'all')
trace4 = go.Box(y = df_bowler.loc[df_bowler['Bowling_team'] == 'Quetta Gladiators', 'Economy'], name = 'Quetta Gladiators', boxmean = 'sd', boxpoints = 'all')

data = [trace0, trace1, trace2, trace3, trace4]

figBowlerBox = go.Figure(data = data)

figBowlerBox.update_layout(
    title=dict(text = "Bowler Economy distribution for each team" ),
    xaxis=dict(title="Teams"),
    yaxis_title="Economy",
    legend_title="Teams")

#figBowlerBox.show()
plotly.offline.plot(figBowlerBox, include_plotlyjs = False, output_type = 'div')


#Creating player score dataframe

dfscore = df_bat[['Batsman_strike', 'Batting_team','TotalRuns','AvgRuns_match', 'Strike_rate', 'Var']].rename(columns = {'Batsman_strike': 'Player', 'Batting_team':'Team','Var': 'Bat_var'}).merge(df_bowler[['Bowler', 'Economy','wicket' ,'AvgWicket','AvgDeliveries_bowled', 'TotalDeliveries', 'Var', 'Bowling_team']].rename(columns = {'Bowler':'Player','Var':'Bowl_var','Bowling_team':'Team'}), on = ['Player','Team'], how = 'outer').fillna(value = 0)     

metricsPostiveIncrement = ['Strike_rate','TotalRuns', 'wicket']
metricsNegativeIncrement = ['Economy', 'Bat_var', 'Bowl_var']


for positiveMetric in metricsPostiveIncrement:
            dfscore[positiveMetric+'Rank'] = np.zeros(len(dfscore))
for negativeMetric in metricsNegativeIncrement:
            dfscore[negativeMetric+'Rank'] = np.zeros(len(dfscore))

def awardingPositiveIncrement(metric, index):
    #Comparing against only bowlers
    x = dfscore[metric].iloc[index]
    not_bat_bowl = False
    if metric == 'wicket':
        quantile = dfscore.loc[dfscore['TotalDeliveries'] > 50, metric] 
        not_bat_bowl = dfscore['TotalDeliveries'].iloc[index] < 50
    else:
        #Comparing against only batsman
        quantile = dfscore.loc[dfscore['TotalRuns'] > 50, metric]
        not_bat_bowl  = dfscore['TotalRuns'].iloc[index] < 50
    if (x == 0) | (not_bat_bowl):
        return 0
    if x <= np.quantile(quantile,0.25):
        return 1
    if x<=  np.quantile(quantile,0.5):
        return  2
    if x<=  np.quantile(quantile,0.75):
        return  3
    else:
        return 4

def awardingNegativeIncrement(metric, index):
    x = dfscore[metric].iloc[index]
    not_bat_bowl = False
    #Comparing against only batsman
    if metric == 'Bat_var':
        mean = dfscore.loc[dfscore['TotalRuns'] > 50, metric].mean()
        not_bat_bowl = dfscore['TotalRuns'].iloc[index] < 50
    #Comparing against only bowler
    elif metric == "Bowl_var":
        mean = dfscore.loc[dfscore['TotalDeliveries'] > 50, metric].mean()
        not_bat_bowl = dfscore['TotalDeliveries'].iloc[index] < 50
    else:
        mean = dfscore.loc[dfscore['TotalDeliveries'] > 50, metric].mean()
    if (x == 0) | (not_bat_bowl):
        return 0
    elif x >=  1.5*mean:
        return 1
    elif ((x >=  mean) & (x < 1.5*mean)):
        return 2
    elif x <= 0.25*mean:
        return 4
    else:
        return 3  

def awardingMetric(index):
    for positiveMetric in metricsPostiveIncrement:
            dfscore[positiveMetric+'Rank'].iloc[index] = awardingPositiveIncrement(positiveMetric, index) 
    for negativeMetric in metricsNegativeIncrement:
            dfscore[negativeMetric+'Rank'].iloc[index] = awardingNegativeIncrement(negativeMetric , index) 
    return dfscore 

for index in range(len(dfscore)):
    awardingMetric(index)

##Creating PlayerScore column as the sum of all metrics
dfscore['PlayerScore'] = dfscore['TotalRunsRank'] + dfscore['Strike_rateRank'] + dfscore['wicketRank'] + dfscore['EconomyRank'] + dfscore['Bat_varRank'] + dfscore['Bowl_varRank']

dfscore= dfscore.sort_values(by=['Team', 'PlayerScore'], ascending=True)

##Histogram with distribution
team_list = list(dfscore['Team'].unique())

trace0 = go.Histogram(x= dfscore.loc[dfscore['Team'] == team_list[0], 'PlayerScore'], opacity = 0.75, name = team_list[0])
trace1 = go.Histogram(x= dfscore.loc[dfscore['Team'] == team_list[1], 'PlayerScore'], opacity = 0.75, name = team_list[1])
trace2 = go.Histogram(x= dfscore.loc[dfscore['Team'] == team_list[2], 'PlayerScore'], opacity = 0.75, name = team_list[2])
trace3 = go.Histogram(x= dfscore.loc[dfscore['Team'] == team_list[3], 'PlayerScore'], opacity = 0.75, name = team_list[3])
trace4 = go.Histogram(x= dfscore.loc[dfscore['Team'] == team_list[4], 'PlayerScore'], opacity = 0.75, name = team_list[4])

figHistogram = make_subplots(rows =2, cols = 3, horizontal_spacing=0.2, vertical_spacing=0.24)
figHistogram.append_trace(trace0,1,1)
figHistogram.append_trace(trace1,1,2)
figHistogram.append_trace(trace2,2,1)
figHistogram.append_trace(trace3,2,2)
figHistogram.append_trace(trace4,2,3)


# Update xaxis properties
figHistogram.update_xaxes(title_text="Number of players", row=1, col=1)
figHistogram.update_xaxes(title_text="Number of players", row=1, col=2)
figHistogram.update_xaxes(title_text="Number of players", row=2, col=1)
figHistogram.update_xaxes(title_text="Number of players", row=2, col=2)
figHistogram.update_xaxes(title_text="Number of players", row=2, col=3)


# Update yaxis properties
figHistogram.update_yaxes(title_text="Player Score", row=1, col=1)
figHistogram.update_yaxes(title_text="Player Score", row=1, col=2)
figHistogram.update_yaxes(title_text="Player Score", row=2, col=1)
figHistogram.update_yaxes(title_text="Player Score", row=2, col=2)
figHistogram.update_yaxes(title_text="Player Score", row=2, col=3)

figHistogram.update_layout(
    title='Distibution of Player Scores for each team',
    legend_title="Team")
 
figHistogram.show()
plotly.offline.plot(figHistogram, include_plotlyjs = False, output_type = 'div')

#Peshawar the clear standout players

Heatmap

team_list = list(dfscore['Team'].unique())

scoreList = []


for team in team_list:
    scoreTeamList = []
    for index in range(len(dfscore)):
        if dfscore['Team'].iloc[index] == team:
            scoreTeamList.append(dfscore['PlayerScore'].iloc[index])
    scoreList.append(scoreTeamList)

    figHeatmap = go.Figure(data=go.Heatmap(
                   z=scoreList,
                   y=team_list,
                   colorbar_title_text= "Player Score",
                   hoverinfo="y+z",
                   hoverongaps = True, ))


figHeatmap.update_layout(
    title='Heatmap of Player Score for each team',
    xaxis_title="Number of Players",
    yaxis_title="Teams",
    legend_title="Player Score",
    xaxis_nticks=18)

# figHeatmap.show()
plotly.offline.plot(figHeatmap, include_plotlyjs = False, output_type = 'div')
