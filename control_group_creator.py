from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
from dtw import *

# Assume you have a DataFrame 'df' with columns 'url', 'timestamp', 'clicks', 'impressions', 'ctr', 'average_rank'
df = pd.read_csv('/Users/ccasazza/Downloads/review_data_sleep.csv')

# Group by url and timestamp to get time series data of each URL
url_time_series = df.groupby(['url', 'timestamp']).sum().reset_index()

# Pivot the data to have a column for each URL
url_time_series = url_time_series.pivot(index='timestamp', columns='url')
#print(url_time_series)

# Fill NA values with 0
url_time_series.fillna(0, inplace=True)
url_time_series.to_csv('/Users/ccasazza/Downloads/url_time_series.csv')



def calculate_dtw2(url1, url2):
    alignment = dtw(url1_data, url2_data, keep_internals=True)
    return alignment.distance


url1_data = [5.93, 7.0, 7.03, 6.92, 7.13, 7.48, 7.55, 6.92, 7.12]
url2_data = [7.81, 8.16, 8.13, 8.08, 8.38, 7.48, 7.22, 7.15, 7.44]
distance = calculate_dtw2(url1_data, url2_data)
#print(distance)

# Assume that 'test_group_urls' is a list of URLs in the test group
test_group_urls = ["https://www.cnet.com/health/sleep/nectar-mattress-review/", "https://www.cnet.com/health/sleep/beautyrest-mattress-reviews-premium-beds-from-top-rated-industry-veteran/", "https://www.cnet.com/health/sleep/tempurpedic-mattress-review/"]

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
        distances[url2].append(calculate_dtw2(url_time_series[('average_rank', url1)].values, url_time_series[('average_rank', url2)].values))

# Average the distances for each URL in the larger group
average_distances = {url: np.mean(distances[url]) for url in distances}

# Find the URLs with the smallest average distances
control_group = sorted(average_distances, key=average_distances.get)[:len(test_group_urls)]

print(control_group)