import pandas as pd

def groupairline(airline):
	tweets = pd.read_csv("Tweets.csv")
	t_gb = tweets.groupby(["airline", "airline_sentiment" ]).count()["name"]
	if airline == 'US':
		airline = 'US Airways'
	return t_gb[airline,'positive'], t_gb[airline,'negative'], t_gb[airline,'neutral']