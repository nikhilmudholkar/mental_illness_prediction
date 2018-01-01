import pandas as pd
import numpy as np

df = pd.read_csv('survey.csv')
#print(df.isnull().sum())
# result shows that there are a lot of missing values in state, work_interference and comments, so we will remove them from the dataframe

drop_columns = ['state', 'work_interfere','comments','Timestamp']
drop_rows = ['queer/she/they','p','Nah','All','Enby','fluid','Genderqueer','Androgyne','non-binary','Agender','male leaning androgynous','Neuter','queer','A little about you','ostensibly male, unsure what that really means']
df = df.drop(drop_columns, axis = 1)
for element in drop_rows :
    df = df[df.Gender!=element]

df['Gender'].replace(['male','m','Male','Cis Male','something kinda male?','Male-ish','maile','Mal','Man','msle','Male (CIS)','Cis Man','Male ','cis male','Make','Malr','Mail','Guy (-ish) ^_^'],'M', inplace = True)
df['Gender'].replace(['Female','female','woman','Femail','f','femail','Woman','Female (trans)','Female ','Female (cis)','Trans woman','cis-female/femme','Femake','Cis Female','Trans-female',''],'F', inplace = True)
#a = range(1,1244)
#for i in a :
#    print(df.iloc[i][1])

print(df['self_employed'].value_counts())

#above command shows that count of no is 1085 and count of yes is 141. Therefore we can safely replace the NA with NO
df['self_employed'] = df['self_employed'].fillna(value = 'No')
#print(df['self_employed'].value_counts())
print(df.nunique())
#print(df.isnull().sum())
#doing this shows that there are no NaN values in data now

df['Age'].replace([-29,329],29,inplace = True)
df['Age'].replace([-1726],26,inplace = True)
df = df[df.Age!=5]

column_names = ["Age","Gender","self_employed","family_history","treatment","no_employees","remote_work","tech_company","benefits","care_options","wellness_program","seek_help","anonymity","leave","mental_health_consequence","phys_health_consequence","coworkers","supervisor","mental_health_interview","phys_health_interview","mental_vs_physical","obs_consequence"]

#removing country colum because the data is very very highly skewed. more that 90% of countries have less than 5 entires. Thereofre it does not seems right to train the data on the besisi of country
df = df.drop('Country', axis = 1)

for element in column_names :
    print(df[element].value_counts())

df.to_csv('survey_cleaned.csv', index = False)


