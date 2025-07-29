import pandas as pd
from sklearn.preprocessing import OneHotEncoder

##reading the csv file
data = pd.read_csv('digital_diet_mental_health.csv')


##one hot encoding to convert categorical data (gender/location) to numeric
encoder = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
encoded_data = encoder.fit_transform(data[['gender','location_type']])

#joining the two tables based on column 
data = pd.concat([data,encoded_data],axis=1)

#dropping the unnecessary columns gender and location_type
data = data.drop(columns=['gender','location_type'])

#converting the cleaned data to csv
data.to_csv("cleaned_data.csv", index = False)





