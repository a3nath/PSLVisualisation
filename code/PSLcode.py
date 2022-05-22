
##importing libraries

import os
import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import ipywidgets as wg
from scipy import stats
from IPython.display import display
from urllib.request import urlopen


#importing data file
psldata = pd.ExcelFile(r'/Users/Amar/Documents/WD/projects/PSLVisualisation/data/All deliveries.xlsx')
df = pd.read_excel(psldata)


##cleaning data
#Replaced Nas
df.iloc[:,-2:] = df.iloc[:,-2:].fillna('No dismisal')


# In[324]:


def wick(x):
    if x != "No dismisal":
        return 1
    else:
        return 0
df['wicket'] = df['Dismisal_kind'].apply(lambda x: wick(x))




def isSix(x):
    if x == 6:
        return 1
    else:
        return 0
    
def isFour(x):
    if x == 4:
        return 1
    else:
        return 0
df['Six'] = df['Batsman_runs'].map(lambda x: isSix(x))
df['Four'] = df['Batsman_runs'].map(lambda x: isFour(x))



#Ball-by-ball detail for each batsman

df_bat = pd.merge(df.groupby(['Batsman_strike','Match_id']).sum().reset_index()[['Batsman_strike','Match_id','Batsman_runs']], df.groupby(['Batsman_strike', 'Match_id']).count().reset_index()[['Batsman_strike','Match_id', 'Ball']].rename(columns = {'Ball': 'Balls_played'}), on = ['Batsman_strike', 'Match_id'])



##Batting datafrane
df_batAvg = pd.merge(pd.merge(df_bat.groupby('Batsman_strike').mean().reset_index()[['Batsman_strike','Batsman_runs', 'Balls_played']].rename(columns = {'Batsman_runs' : 'AvgRuns_match' , 'Balls_played': 'AvgBalls_match'}), df[['Batsman_strike', 'Batting_team']].drop_duplicates(keep = 'first'), on = 'Batsman_strike') , pd.merge(df.groupby(['Batsman_strike']).sum().reset_index()[['Batsman_strike', 'Batsman_runs', 'Six', 'Four']].rename(columns = {'Batsman_runs': 'TotalRuns', 'Six':'TotalSixes', 'Four':'TotalFours'}), df.groupby(['Batsman_strike']).count().reset_index()[['Batsman_strike' , 'Ball']].rename(columns = {'Ball': 'Balls_played'}), on ='Batsman_strike'), on = 'Batsman_strike')
                                                                                                                                     
                                                                                                                                     

df_batAvg['Runs_sq'] = 0
df_batAvg['Matches_played'] = 0


#To calculate variance
for i in range(len(df_batAvg)):
    for j in range(len(df_bat)):
        if df_batAvg.iloc[i]['Batsman_strike'] == df_bat.iloc[j]['Batsman_strike']:
            df_batAvg['Runs_sq'].iloc[i] += df_bat.iloc[j]['Batsman_runs'] **2
            df_batAvg['Matches_played'].iloc[i] += 1


#Strke rate, % boundary and Variance
df_batAvg['Strike_rate'] = df_batAvg['TotalRuns'] * 100 /df_batAvg['Balls_played']
df_batAvg['PercentageRuns_boundary'] = (df_batAvg['TotalSixes']* 6 + df_batAvg['TotalFours'] * 4) * 100/df_batAvg['TotalRuns']
df_batAvg["Var"] = (df_batAvg['Runs_sq'] - df_batAvg['Matches_played']*(df_batAvg['AvgRuns_match']**2))/(df_batAvg['Matches_played'] - 1)
df_batAvg = df_batAvg.sort_values(by = 'TotalRuns', ascending = False)


##Batsman Scatter

z = df_batAvg['Batsman_strike']
trace = go.Scatter(y = df_batAvg['Strike_rate'], x= df_batAvg['AvgRuns_match'], mode = 'markers', showlegend = False , text = df_batAvg['Batsman_strike'] + "  " + df_batAvg['Batting_team'], hoverinfo =  "text", marker = dict(size = 20, symbol = 'circle' , cmin = 0, autocolorscale = True, reversescale = True, colorbar = {'title' : {'text': 'Score variance'}}, color = df_batAvg['Var']))
data = [trace]

layout = go.Layout(paper_bgcolor = 'white',plot_bgcolor = 'white' , yaxis = dict(title = dict(text = "Strike rate" , font = dict(color ='black')) ,showgrid = True, zeroline = False, color = 'black'), xaxis = dict( title= dict( text = 'Average runs'), showgrid = False, zeroline= False), title = dict(text = "All batsman's performance - Average runs, Strike rate, Score variance"))
fig = go.Figure(data = data, layout = layout)
plotly.offline.init_notebook_mode(connected=True)
plotly.offline.plot(fig, include_plotlyjs = False, output_type = 'div')


#EBatsman Histogram


team_list = list(df_batAvg['Batting_team'].unique())

trace0 = go.Histogram(x= df_batAvg.loc[df_batAvg['Batting_team'] == team_list[0], 'AvgRuns_match'], opacity = 0.75, name = team_list[0])
trace1 = go.Histogram(x= df_batAvg.loc[df_batAvg['Batting_team'] == team_list[1], 'AvgRuns_match'], opacity = 0.75, name = team_list[1])
trace2 = go.Histogram(x= df_batAvg.loc[df_batAvg['Batting_team'] == team_list[2], 'AvgRuns_match'], opacity = 0.75, name = team_list[2])
trace3 = go.Histogram(x= df_batAvg.loc[df_batAvg['Batting_team'] == team_list[3], 'AvgRuns_match'], opacity = 0.75, name = team_list[3])
trace4 = go.Histogram(x= df_batAvg.loc[df_batAvg['Batting_team'] == team_list[4], 'AvgRuns_match'], opacity = 0.75, name = team_list[4])

fig4 = plotly.tools.make_subplots(rows =3, cols = 2)
fig4.append_trace(trace0,1,1)
fig4.append_trace(trace1,2,1)
fig4.append_trace(trace2,2,2)
fig4.append_trace(trace3,3,1)
fig4.append_trace(trace4,3,2)

fig4.layout.title = dict(text = 'Distribution of average batsman scores')
fig4.layout.xaxis = dict(tickmode = 'linear', dtick = 10)


plotly.offline.plot(fig4, include_plotlyjs = False, output_type = 'div')



def runlevel(x):
    if x >= 1.25*df_batAvg['AvgRuns_match'].mean():
        return 'High'
    elif x <= 0.75 * df_batAvg['AvgRuns_match'].mean():
        return 'Low'
    else:
        return 'Average'
    
def strikelevel(x):
    if x >= 1.25*df_batAvg['Strike_rate'].mean():
        return 'High'
    elif x <= 0.75 * df_batAvg['Strike_rate'].mean():
        return 'Low'
    else:
        return 'Average'
    
def varlevel(x):
    if x <= 0.75*df_batAvg['Var'].mean():
        return 'High Consistency'
    elif x >= 1.25 * df_batAvg['Var'].mean():
        return 'Inconsistent'
    else:
        return 'Average'
    

df_batAvg['Runs_level'] = df_batAvg['AvgRuns_match'].map(lambda x: runlevel(x))
df_batAvg['Strike_level'] = df_batAvg['Strike_rate'].map(lambda x: strikelevel(x))
df_batAvg['Var_level'] = df_batAvg['Var'].map(lambda x: varlevel(x))


dfteam = df.groupby(['Batting_team','Bowling_team', 'Match_id', 'Innings', 'Over']).sum().reset_index()
over = 0
total = 0
Total_runs = []

for i in range(len(dfteam)):
    if dfteam.iloc[i]['Over'] >= over:
        total =+ total + dfteam.iloc[i]['Batsman_runs'] + dfteam.iloc[i]['Extra_runs']
        over =+ 1
        Total_runs.append(total)
    else:
        total = dfteam.iloc[i]['Batsman_runs'] + dfteam.iloc[i]['Extra_runs']
        Total_runs.append(total)
        over = 1


dfteam['Total_runs'] = pd.Series(Total_runs)


##TEAM Innings performance
#Batting first vs second


team_list = wg.Dropdown(options = list(df['Batting_team'].unique()))

def response(team):
                        
    dfteamid = dfteam.loc[dfteam['Batting_team'] == team].groupby(['Batting_team','Match_id','Innings','Over']).sum().reset_index().copy()
    match_list = list(dfteamid['Match_id'].unique())

    datainn1 = []
    datainn2 = []

    for match in match_list:
        traceinn1 = go.Scatter(x = pd.Series(range(20)), y = dfteamid.loc[(dfteamid['Match_id'] == match) & (dfteamid['Innings'] == 1),'Total_runs'], marker = dict(color = 'grey') , line = dict(dash = 'dot') ,name = "Match num: " + str(match), hoverinfo = "all" , showlegend = True)
        datainn1.append(traceinn1)
        traceinn2 = go.Scatter(x = pd.Series(range(20)), y = dfteamid.loc[(dfteamid['Match_id'] == match) & (dfteamid['Innings'] == 2),'Total_runs'], marker = dict(color = 'indigo'), line = dict(dash = 'dash'), text = str(match), name = "Match num: " + str(match), showlegend = True)
        datainn2.append(traceinn2)

    slope1, intercept1, r_value, p_value, std_err = stats.linregress(dfteamid.loc[dfteamid['Innings'] == 1,'Over'], dfteamid.loc[dfteamid['Innings'] == 1,'Total_runs'])                       
    traceinn1bf = go.Scatter(x = pd.Series(range(20)), y = slope1*pd.Series(range(20)) + intercept1 , marker = dict(color = 'red'), name = '1stinningsbestfit', showlegend = True)

    slope2, intercept2, r_value, p_value, std_err = stats.linregress(dfteamid.loc[dfteamid['Innings'] == 2,'Over'], dfteamid.loc[dfteamid['Innings'] == 2,'Total_runs'])
    traceinn2bf = go.Scatter(x = pd.Series(range(20)), y = slope2*pd.Series(range(20)) + intercept2, marker = dict(color = 'green'), name = '2ndinnningsbestfit', showlegend = True)

    data = datainn1+datainn2+[traceinn1bf,traceinn2bf]

    anno1 = [dict(x = 14, y = slope1*14 + intercept1, xref = 'x', yref = 'y', text = str(team) + ' Average 1st inn'), dict(x = 19.2, y = slope1*19 + intercept1, xref = 'x', yref = 'y', text = 'Total Score: ' + str(round(slope1*19 + intercept1)), showarrow = False, font = dict(color = 'white'), bgcolor = 'black', xanchor = 'left')]
    anno2 = [dict(x = 18, y = slope2*18 + intercept2, xref = 'x', yref = 'y', text = str(team) + ' Average 2nd inn'), dict(x = 19.2, y = slope2*19 + intercept2, xref = 'x', yref = 'y', text = 'Total Score: ' + str(round(slope2*19 + intercept2)), showarrow = False, font = dict(color = 'white'), bgcolor = 'black', xanchor = 'left')]

    updatemenus = list([
        dict(type = 'buttons', active = -1, buttons = list([ 
            dict(visible = True, label = '1st Innings', method = 'update', args = [{'visible' : [True] * len(datainn1) + [False] * len(datainn2) +  [True,False]},
                                                                  {'title': str(team) + " First Innings", 'annotations': anno1}]),
            dict(visible = True, label = '2nd Innings', method = 'update', args = [{'visible' : [False] * len(datainn1) + [True] * len(datainn2) +  [False,True]},
                                                                   {'title': str(team) + ' Second Innings', 'annotations': anno2}]),
            dict(visible = True, label = 'Both', method = 'update', args = [{'visible': [True] * len(datainn1) + [True] * len(datainn2) +  [True,True]},
                                                           {'title': 'Both', 'annotations': anno1+anno2}])]))])

    layout = dict(title =  str(team) + ' Innings', updatemenus = updatemenus, xaxis = {'title' : {'text' : 'Overs'}, 'ticks': 'inside', 'tickson': 'labels', 'tickcolor' : 'grey', 'tickmode': 'linear', 'showticklabels': True, 'showgrid': False}, yaxis = {'title' : {'text': 'Total runs'}, 'nticks': 10, 'showgrid': False})
    fig11 = dict(data = data, layout = layout)
    plotly.offline.plot(fig11, include_plotlyjs = False, output_type = 'div')
    

wg.interact(response, team=team_list)


##Head to head Innings performance
#Team 1 vs Team 2 - Batting first and second

team_list1 = wg.Dropdown(options = list(df['Batting_team'].unique()), description = "Team1 Name")
team_list2 = wg.Dropdown(options = df['Batting_team'].unique().tolist(), description = "Team2 Name ")

def head_head(team1,team2):
    
    data = []
    
    dfteam1 = dfteam.loc[(dfteam['Batting_team'] == team1) & (dfteam['Bowling_team'] == team2)].groupby(['Batting_team','Match_id','Innings','Over']).sum().reset_index().copy()
    dfteam2 = dfteam.loc[(dfteam['Batting_team'] == team2) & (dfteam['Bowling_team'] == team1)].groupby(['Batting_team', 'Match_id', 'Innings', 'Over']).sum().reset_index().copy()
    
    try:
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(dfteam1.loc[dfteam1['Innings'] == 1,'Over'], dfteam1.loc[dfteam1['Innings'] == 1,'Total_runs'])                       
        traceteam1bat1 = go.Scatter(x = pd.Series(range(20)), y = slope1*pd.Series(range(20)) + intercept1 , marker = dict(color = 'red'), name = str(team1) + ' Innings', showlegend = True)
        data.append(traceteam1bat1)
        anno1 = [dict(x = 17, y = slope1*17 + intercept1, xref = 'x', yref = 'y', text = 'Total Score: ' + str(round(slope1*19 + intercept1)), showarrow = False, font = dict(color = 'white'), bgcolor = 'black', xanchor = 'left')]  
    except ValueError or UnboundLocalError:  
        data.append({})
        anno1 = ""
        
    try:
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(dfteam2.loc[dfteam2['Innings'] == 2,'Over'], dfteam2.loc[dfteam2['Innings'] == 2,'Total_runs'])
        traceteam2bat2 = go.Scatter(x = pd.Series(range(20)), y = slope2*pd.Series(range(20)) + intercept2, marker = dict(color = 'green'), name = str(team2) + ' Innings', showlegend = True)
        data.append(traceteam2bat2)
        anno2 = [dict(x = 20, y = slope2*20 + intercept2, xref = 'x', yref = 'y', text = 'Total Score: ' + str(round(slope2*19 + intercept2)), showarrow = False, font = dict(color = 'white'), bgcolor = 'black', xanchor = 'left')]
    except ValueError or UnboundLocalError:
        data.append({})
        anno2 = ""
    
    try:
        slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(dfteam1.loc[dfteam1['Innings'] == 2,'Over'], dfteam1.loc[dfteam1['Innings'] == 2,'Total_runs'])                       
        traceteam1bat2 = go.Scatter(x = pd.Series(range(20)), y = slope3*pd.Series(range(20)) + intercept3 , marker = dict(color = 'red'), name = str(team1) + ' Innings', showlegend = True)
        data.append(traceteam1bat2)
        anno3 = [dict(x = 17, y = slope3*17 + intercept3, xref = 'x', yref = 'y', text = 'Total Score: ' + str(round(slope3*19 + intercept3)), showarrow = False, font = dict(color = 'white'), bgcolor = 'black', xanchor = 'left')]
    except ValueError or UnboundLocalError:
        data.append({})
        anno3 = ""
            
    try:
        slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(dfteam2.loc[dfteam2['Innings'] == 1,'Over'], dfteam2.loc[dfteam2['Innings'] == 1,'Total_runs'])
        traceteam2bat1 = go.Scatter(x = pd.Series(range(20)), y = slope4*pd.Series(range(20)) + intercept4, marker = dict(color = 'green'), name = str(team2) + ' Innings', showlegend = True)
        data.append(traceteam2bat1)
        anno4 = [dict(x = 20, y = slope4*20 + intercept4, xref = 'x', yref = 'y', text = 'Total Score: ' + str(round(slope4*19 + intercept4)), showarrow = False, font = dict(color = 'white'), bgcolor = 'black', xanchor = 'left')]
    except ValueError or UnboundLocalError:
        data.append({})
        anno4 = ""
        
   
        
        
    updatemenus = list([
        dict(type = 'buttons', active = -1, buttons = list([ 
            dict(visible = True, label = str(team1) + ' Bat first', method = 'update', args = [{'visible' : [True,True,False,False]},
                                                                                               {'title': str(team1) + ' vs ' + str(team2) + " - " + str(team1) + " Batting First", 'annotations': anno1+anno2}]),
            dict(visible = True, label = str(team1) + ' Bat second', method = 'update', args = [{'visible' : [False,False,True,True]},
                                                        {'title': str(team1) + ' vs ' + str(team2) + " - " + str(team1) + " Batting Second", 'annotations': anno3 + anno4}]),
            ]))])

    layout = dict(title =  str(team1) + ' vs ' + str(team2) + ' innings', updatemenus = updatemenus, xaxis = {'title' : {'text' : 'Overs'}, 'ticks': 'inside', 'tickcolor' : 'grey', 'tickmode': 'array', 'showticklabels': True, 'showgrid': False, 'tickvals': list(range(21))}, yaxis = {'title' : {'text': 'Total runs'}, 'nticks': 10, 'showgrid': False})
    
    fig12 = dict(data = data, layout = layout)
    plotly.offline.plot(fig12, include_plotlyjs = False)
    
    
wg.interact(head_head, team1 = team_list1, team2 = team_list2)


wg.interact(head_head, team1 = team_list1, team2 = team_list2)


## Defining bowler metrics
def caught(x):
    if x == 'caught':
        return 1
    else:
        return 0
def bowled(x):
    if x == bowled:
        return 1
    else:
        return 0
def runout(x):
    if x == 'runout':
        return 1
    else:
        return 0
def lbw(x):
    if x == 'lbw':
        return 1
    else:
        return 0
def stump(x):
    if x == 'stump':
        return 1
    else:
        return 0
def candb(x):
    if x == 'caught and bowled':
        return 1
    else:
        return 0


df['Caught'] = df['Dismisal_kind'].apply(lambda x: caught(x))
df['Bowled'] = df['Dismisal_kind'].apply(lambda x: bowled(x))
df['Lbw'] = df['Dismisal_kind'].apply(lambda x: lbw(x))
df['Runout'] = df['Dismisal_kind'].apply(lambda x: runout(x))
df['Stump'] = df['Dismisal_kind'].apply(lambda x: stump(x))
df['Caught_bowled'] = df['Dismisal_kind'].apply(lambda x: candb(x))


## Creating a bowler dataframe
df_bowl = pd.merge(df.groupby(['Bowler', 'Match_id']).sum().reset_index().filter(regex = "[Innings,Over,Ball]"), df.groupby(['Bowler', 'Match_id']).count().reset_index()[['Bowler','Match_id', 'Ball']].rename(columns = {'Ball': 'TotalDeliveries'}), on = ['Bowler', 'Match_id'])

df_bowlAvg = pd.merge(df.groupby(['Bowler', 'Match_id']).sum().reset_index().filter(regex = "[^Match_id,Innings,Over,Ball]") ,  df.groupby(['Bowler', 'Match_id']).count().reset_index()[['Bowler', 'Ball']].rename(columns = {'Ball': 'Delieveries_bowled'}), on = 'Bowler').groupby('Bowler').mean().reset_index()


df_bowl_Avg_merge = pd.merge(df_bowl.groupby('Bowler').sum().reset_index().filter(regex = '[^Match_id,Innings,Over,Ball]'), df_bowlAvg.filter(regex = "[^Caught,Bowled,Stump,Caught_bowled, Lbw, Runout]").rename(columns = {'Delieveries_bowled':'AvgDeliveries_bowled','Batsman_runs': 'AvgBatsman_runs', 'Extra_runs': 'AvgExtra_runs' , 'Six': 'AvgSix', 'Four': 'AvgFour', 'wicket' : 'AvgWicket'}),
 on = 'Bowler')

dfecon = df.groupby(['Bowler','Match_id', 'Over']).sum().reset_index().groupby('Bowler').mean().reset_index()[['Bowler','Batsman_runs', 'Extra_runs']]
dfecon['Economy'] = dfecon['Batsman_runs'] + dfecon['Extra_runs']



df_bowler = pd.merge(df_bowl_Avg_merge, dfecon[['Bowler','Economy']], on = 'Bowler')


## Bowler variance
df_bowler['Runs_sq'] = 0
df_bowler['Matches_played'] = 0
for i in range(len(df_bowler)):
    for j in range(len(df_bowl)):
        if df_bowler.iloc[i]['Bowler'] == df_bowl.iloc[j]['Bowler']:
            df_bowler['Runs_sq'].iloc[i] += (df_bowl.iloc[j]['Batsman_runs'] +  df_bowl.iloc[j]['Extra_runs'])**2
            df_bowler['Matches_played'].iloc[i] += 1


df_bowler['Var'] = (df_bowler['Runs_sq'] - (df_bowler['Matches_played']*((df_bowler['AvgBatsman_runs']+df_bowler['AvgExtra_runs'])**2)))/(df_bowler['Matches_played'] -1)

 

df_bowler = pd.merge(df_bowler, df[['Bowler', 'Bowling_team']].drop_duplicates(keep = 'first'), on = 'Bowler')
df_bowler.dropna(inplace = True)


##Bowler Scatter chart

trace1 = go.Scatter(x = df_bowler['wicket'], y = df_bowler['Economy'], mode = 'markers', hoverinfo = 'text' , text = df_bowler['Bowler'] + "   " + df_bowler['Bowling_team'] , marker = dict(size = 25, symbol = 'circle' , cmin = 0, autocolorscale = True, reversescale = True, colorbar = {'title' : {'text': 'Bowler Variance'}}, color = df_bowler['Var']))

layout = go.Layout(paper_bgcolor = 'white',plot_bgcolor = 'white' , hovermode = 'closest',  yaxis = dict(title = dict(text = "Economy" , font = dict(color ='black')) ,showgrid = True, zeroline = False, color = 'black'), xaxis = dict(title = dict( text = 'Total wickets'), showgrid = False, zeroline = False), title = dict(text = "All Bowler's Performance - Economy, Total Wickets, Variance"))


data = [trace1]

fig22 = go.Figure(data = data, layout = layout)

plotly.offline.init_notebook_mode(connected=True)
plotly.offline.plot(fig22, include_plotlyjs = False, output_type = 'div')


##Bowler histogram


team_list = ['Peshawar Zalmi', 'Karachi Kings', 'Islamabad United', 'Quetta Gladiators', 'Lahore Qalandars']

trace0 = go.Histogram(x= df_bowler.loc[df_bowler['Bowling_team'] == team_list[0], 'Economy'], opacity = 0.75, name = team_list[0])
trace1 = go.Histogram(x= df_bowler.loc[df_bowler['Bowling_team'] == team_list[1], 'Economy'], opacity = 0.75, name = team_list[1])
trace2 = go.Histogram(x= df_bowler.loc[df_bowler['Bowling_team'] == team_list[2], 'Economy'], opacity = 0.75, name = team_list[2])
trace3 = go.Histogram(x= df_bowler.loc[df_bowler['Bowling_team'] == team_list[3], 'Economy'], opacity = 0.75, name = team_list[3])
trace4 = go.Histogram(x= df_bowler.loc[df_bowler['Bowling_team'] == team_list[4], 'Economy'], opacity = 0.75, name = team_list[4])

fig4 = plotly.tools.make_subplots(rows = 3, cols = 2)
fig4.append_trace(trace0,1,1)
fig4.append_trace(trace1,1,2)
fig4.append_trace(trace2,2,1)
fig4.append_trace(trace3,2,2)
fig4.append_trace(trace4,3,1)

fig4.layout.title = dict(text = "Distribution of average bowler economy for each team")

plotly.offline.plot(fig4, include_plotlyjs = False, output_type = 'div')


##Setting up a ranking datafrane

dfscore = df_batAvg[['Batsman_strike', 'Batting_team','AvgRuns_match', 'Strike_rate', 'Var']].rename(columns = {'Batsman_strike': 'Player', 'Batting_team':'Team','Var': 'Bat_var'}).merge(df_bowler[['Bowler', 'Economy', 'AvgWicket','AvgDeliveries_bowled', 'Var', 'Bowling_team']].rename(columns = {'Bowler':'Player','Var':'Bowl_var','Bowling_team':'Team'}), on = ['Player','Team'], how = 'outer').fillna(value = 0)                                                                                                          


##Player Ranking Metrics

f = 'Strike_rate'
g = 'AvgRuns_match'
h = 'AvgWicket'


Batsman = np.quantile(dfscore['AvgRuns_match'], 0.5)
Batvarmean = dfscore.loc[dfscore['AvgRuns_match'] >= Batsman, "Bat_var"].mean()


Bowlvarmean = dfscore.loc[dfscore['AvgDeliveries_bowled'] >= 0, "Bowl_var"].mean()

dfscore['BatVarRank'] = np.zeros(len(dfscore))

dfscore['BowlVarRank'] = np.zeros(len(dfscore))


def greaterf(x):
        if x == 0:
            return 0
        if x >= 1.25 * dfscore[f].mean():
            return 3
        if x<= 0.75 *dfscore[f].mean():
            return  1
        else:
            return 2

def greaterg(x):
        if x == 0:
            return 0
        if x >= 1.25 * dfscore[g].mean():
            return 3
        if x<= 0.75 *dfscore[g].mean():
            return  1
        else:
            return 2

def greaterh(x):
        if x == 0:
            return 0
        if x >= 1.25 * dfscore[h].mean():
            return 3
        if x<= 0.75 *dfscore[h].mean():
            return  1
        else:
            return 2


dfscore[f + str('Rank')] = dfscore[f].map(lambda x: greaterf(x))    
dfscore[g + str('Rank')] = dfscore[g].map(lambda x: greaterg(x))
dfscore[h + str('Rank')] = dfscore[h].map(lambda x: greaterh(x))

def ecorank(x):
    if x == 0:
        return 0
    elif x >= 1.25 * dfscore.loc[dfscore['AvgDeliveries_bowled'] >= 0, "Economy"].mean():
        return 1
    elif x <= 0.75 * dfscore.loc[dfscore['AvgDeliveries_bowled'] >= 0, "Economy"].mean():
        return 3
    else:
        return 2
    
dfscore['EconomyRank'] = dfscore['Economy'].map(lambda x: ecorank(x))


for i in range(len(dfscore)):
    if dfscore.iloc[i]['AvgRuns_match'] <= Batsman:
        dfscore['BatVarRank'].iloc[i] = 0
    elif dfscore['Bat_var'].iloc[i] >= (1.25 * Batvarmean):  
        dfscore['BatVarRank'].iloc[i] = 1
    elif dfscore.iloc[i]['Bat_var'] <= 0.75 * Bowlvarmean:
        dfscore['BatVarRank'].iloc[i] = 3
    else:
        dfscore['BatVarRank'].iloc[i] = 2
        
        
        
for i in range(len(dfscore)):
    if dfscore.iloc[i]['AvgDeliveries_bowled'] == 0:
        dfscore['BowlVarRank'].iloc[i] = 0
    elif dfscore['Bowl_var'].iloc[i] >= (1.25 * Bowlvarmean):  
        dfscore['BowlVarRank'].iloc[i] = 1
    elif dfscore.iloc[i]['Bowl_var'] <= 0.75 * Bowlvarmean:
        dfscore['BowlVarRank'].iloc[i] = 3
    else:
        dfscore['BowlVarRank'].iloc[i] = 2



##Creating PlayerScore column as the sum of all metrics

dfscore['PlayerScore'] = dfscore['AvgRuns_matchRank'] + dfscore['Strike_rateRank'] + dfscore['AvgWicketRank'] + dfscore['BatVarRank'] + dfscore['BowlVarRank'] + dfscore['EconomyRank']


##Ranking levels

def mvprank(x):
    if x >= np.quantile(dfscore['PlayerScore'], 0.90):
        return 4
    elif x >= np.quantile(dfscore['PlayerScore'], 0.75):
            return 3
    elif x <= np.quantile(dfscore['PlayerScore'], 0.50):
            return 1
    else:
        return 2
    
dfscore['PlayerRank'] = dfscore['PlayerScore'].map(lambda x: mvprank(x))
dfscore['MVP'] = np.zeros(len(dfscore))
dfscore.loc[dfscore['PlayerRank']==4,'MVP'] = 'MVP'
dfscore.loc[dfscore['PlayerRank']==3, 'MVP'] = 'VP'
dfscore.loc[dfscore['PlayerRank'] <3, 'MVP'] = 'Below Average'


## Top 10 Players
dfscore.sort_values(by = 'PlayerScore', ascending = False).head(10)


#Boxplot

trace0 = go.Box(y = dfscore.loc[dfscore['Team'] == 'Peshawar Zalmi', 'PlayerScore'], name = 'Peshawar Zalmi', boxmean = 'sd', boxpoints = 'all')
trace1 = go.Box(y = dfscore.loc[dfscore['Team'] == 'Islamabad United', 'PlayerScore'], name = 'Islamabad United', boxmean = 'sd', boxpoints = 'all')
trace2 = go.Box(y = dfscore.loc[dfscore['Team'] == 'Quetta Gladiators', 'PlayerScore'], name = 'Quetta Gladiators', boxmean = 'sd', boxpoints = 'all')
trace3 = go.Box(y = dfscore.loc[dfscore['Team'] == 'Karachi Kings', 'PlayerScore'], name = 'Karachi Kings' , boxmean = 'sd', boxpoints = 'all')
trace4 = go.Box(y = dfscore.loc[dfscore['Team'] == 'Lahore Qalandars', 'PlayerScore'], name = 'Lahore Qalandars' , boxmean = 'sd', boxpoints = 'all')

data = [trace0, trace1, trace2, trace3, trace4]

fig111 = go.Figure(data = data)
fig111.layout.title = dict(text = "Player Score distribution for each team" )

import bs4
from bs4 import BeautifulSoup

plotly.offline.plot(fig111, include_plotlyjs = False, output_type = 'div')
