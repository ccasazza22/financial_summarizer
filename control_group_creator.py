from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
from dtw import *

# Assume you have a DataFrame 'df' with columns 'url', 'timestamp', 'clicks', 'impressions', 'ctr', 'average_rank'
df = pd.read_csv('/Users/ccasazza/Downloads/energy_Geo_controlgroup_data.csv')

# Group by url and timestamp to get time series data of each URL
url_time_series = df.groupby(['url', 'timestamp']).sum().reset_index()

# Pivot the data to have a column for each URL
url_time_series = url_time_series.pivot(index='timestamp', columns='url')
#print(url_time_series)

# Fill NA values with 0
url_time_series.fillna(0, inplace=True)
url_time_series.to_csv('/Users/ccasazza/Downloads/url_time_series.csv')



def calculate_dtw2(url1, url2):
    alignment = dtw(url1, url2, keep_internals=True)
    return alignment.distance



# Assume that 'test_group_urls' is a list of URLs in the test group
test_group_urls = ["https://www.cnet.com/home/energy-and-utilities/california-solar-panels/", "https://www.cnet.com/home/energy-and-utilities/florida-solar-panels/","https://www.cnet.com/home/energy-and-utilities/michigan-solar-panels/","https://www.cnet.com/home/energy-and-utilities/maryland-solar-panels/","https://www.cnet.com/home/energy-and-utilities/pennsylvania-solar-panels/"]

# All unique URLs in the DataFrame
all_urls = df['url'].unique().tolist()

# URLs in the larger group are all URLs that are not in the test group
larger_group_urls = [url for url in all_urls if url not in test_group_urls]


# Calculate DTW distance for each pair of URLs
# Calculate DTW distance for each pair of URLs
distances = {}
for url1 in test_group_urls:
    for url2 in larger_group_urls:
        print(url1)
        print(url2)
        if url2 not in distances:
            distances[url2] = []
        distances[url2].append(calculate_dtw2(url_time_series[('clicks', url1)].values, url_time_series[('clicks', url2)].values))

# Average the distances for each URL in the larger group
average_distances = {url: np.mean(distances[url]) for url in distances}

# Find the URLs with the smallest average distances
control_group = sorted(average_distances, key=average_distances.get)[:len(test_group_urls)]

print(control_group)