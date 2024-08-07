import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import json



def fetch_duration(lon, lat):
    response = requests.get(base_url + f"{base_lon},{base_lat};{lon},{lat}?overview=false")
    if response.status_code == 200:
        # Assuming the API returns a JSON with a 'duration' field
        json_data = response.json()
        return json_data.get('routes', None)[0].get('duration')  # Adjust according to actual API response structure
    else:
        return None

def getIndexFromClusterAsc(cluster):
    df_order = data_encoded.groupby('cluster')['duration_plus'].mean()
    df_order = df_order.sort_values(ascending=True) 

    return df_order.index[cluster]

def main():
    # Load dictionary from command-line arguments
    df = pd.read_csv('Data/customer-dummy.csv')
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df['customer_id'] = df['customer_id'].astype(int)

    data = df.copy()
    data.set_index('customer_id', inplace=True)

    data['location_lon'] = data['location'].str.extract(r'(\d+\.\d+),\s*(\d+\.\d+)', expand=True)[0]
    data['location_lat'] = data['location'].str.extract(r'(\d+\.\d+),\s*(\d+\.\d+)', expand=True)[1]
    data.drop('location', axis=1, inplace=True)

    base_url = f'https://routing.openstreetmap.de/routed-car/route/v1/driving/'

    base_lon = '27.164404'
    base_lat = '31.198769'

    data_encoded = data.copy()

    data_encoded['duration'] = data_encoded.apply(lambda row: fetch_duration(row['location_lon'], row['location_lat']), axis=1)

    data_encoded['stops_s'] = data_encoded['stops'] * 12 * 60 * 60
    data_encoded['duration_plus'] = data_encoded['duration'] + data_encoded['stops_s']

    X = data_encoded[['duration_plus']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    data_encoded['cluster'] = kmeans.fit_predict(X_scaled)

    df_order = data_encoded.groupby('cluster')['duration_plus'].mean()
    df_order = df_order.sort_values(ascending=True)

    dict_str = sys.argv[1]
    data_dict = json.loads(dict_str)
    
    # Process the dictionary
    print("Received dictionary:", data_dict)

    return_df = pd.DataFrame()

    return_df = pd.DataFrame(columns=['customer_id', 'apples_amount', 'bananas_amount', 'carrots_amount', 'tomatoes_amount', 'apple_batch', 'banana_batch', 'carrot_batch', 'tomato_batch', 'duration_plus', 'cluster'])

    for index, row in data_encoded.iterrows():
        customer_id = index
        apple_amount = row['apple']
        banana_amount = row['banana']
        carrot_amount = row['carrot']
        tomato_amount = row['tomato']
        duration_plus = row['duration_plus']
        cluster = row['cluster']
        apple_batch = 0 if row['apple'] == 0 else batchs['apple'][getIndexFromClusterAsc(cluster)]
        banana_batch = 0 if row['banana'] == 0 else batchs['banana'][getIndexFromClusterAsc(cluster)]
        carrot_batch = 0 if row['carrot'] == 0 else batchs['carrot'][getIndexFromClusterAsc(cluster)]
        tomato_batch = 0 if row['tomato'] == 0 else batchs['tomato'][getIndexFromClusterAsc(cluster)]
        return_df = return_df.append({'customer_id': customer_id, 'apple_amount': apple_amount, 'banana_amount': banana_amount, 'carrot_amount': carrot_amount, 'tomato_amount': tomato_amount, 'apple_batch': apple_batch, 'banana_batch': banana_batch, 'carrot_batch': carrot_batch, 'tomato_batch': tomato_batch, 'duration_plus': duration_plus, 'cluster': cluster}, ignore_index=True)

    return_df.drop(['apples_amount',	'bananas_amount','carrots_amount',	'tomatoes_amount'], axis=1, inplace=True)
    return_df

if __name__ == "__main__":
    main()