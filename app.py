import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from azure.storage.blob import BlobServiceClient
import json
from io import StringIO
import os

# Configuration
st.set_page_config(layout="wide")

# Load data from Azure Blob Storage
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    # Use this EXACT format (replace placeholders with your real values)
    connection_string = (
        "DefaultEndpointsProtocol=https;"
        "AccountName=iesstsabdbaa;"  # Just the name, no URLs
        "AccountKey=yfqMW8gf8u+M5pOW33Q5gtRTFBJQXStVK4K2rlCVVzxlrRG21Sh7MVj06uExoL86Npb7HWWgxYUe+ASthUr6/g==;"
        "EndpointSuffix=core.windows.net"
    )
    container_name = "group7"
    # Rest of your function...
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Load ride data
    ride_blob_client = blob_service_client.get_blob_client(container=container_name, blob="ride_data.json")
    ride_data = json.loads(ride_blob_client.download_blob().readall().decode())
    ride_df = pd.DataFrame(ride_data)
    
    # Load driver data
    driver_blob_client = blob_service_client.get_blob_client(container=container_name, blob="driver_data.json")
    driver_data = json.loads(driver_blob_client.download_blob().readall().decode())
    driver_df = pd.DataFrame(driver_data)
    
    return ride_df, driver_df

# Basic Analytics
def basic_analytics(ride_df, driver_df):
    st.header("Basic Analytics")
    
    # Ride status counts
    status_counts = ride_df['status'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Requested Rides", status_counts.get('requested', 0))
    col2.metric("Accepted Rides", status_counts.get('accepted', 0))
    col3.metric("Completed Rides", status_counts.get('completed', 0))
    
    # Rides by type
    st.subheader("Rides by Type")
    fig1, ax1 = plt.subplots()
    ride_df['ride_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
    st.pyplot(fig1)
    
    # Rides by vehicle type (merged with driver data)
    st.subheader("Rides by Vehicle Type")
    merged_df = pd.merge(ride_df, driver_df, left_on='passenger_id', right_on='driver_id', how='left')
    fig2, ax2 = plt.subplots()
    sns.countplot(x='vehicle_type', data=merged_df, ax=ax2)
    st.pyplot(fig2)

# Intermediate Analytics
def intermediate_analytics(ride_df):
    st.header("Intermediate Analytics")
    
    # Cancellation rates
    st.subheader("Cancellation Rates by Ride Type")
    total_rides = ride_df.groupby('ride_type').size()
    cancelled_rides = ride_df[ride_df['status'] == 'cancelled'].groupby('ride_type').size()
    cancellation_rates = (cancelled_rides / total_rides * 100).fillna(0)
    
    fig, ax = plt.subplots()
    cancellation_rates.plot.bar(ax=ax)
    ax.set_ylabel("Cancellation Rate (%)")
    st.pyplot(fig)

# Advanced Analytics
def advanced_analytics(ride_df):
    st.header("Advanced Analytics")
    
    # Surge pricing prediction
    st.subheader("Surge Pricing Zone Prediction")
    
    # Bin coordinates
    ride_df['lat_bin'] = (ride_df['pickup_lat'] / 0.1).astype(int)
    ride_df['lng_bin'] = (ride_df['pickup_lng'] / 0.1).astype(int)
    
    demand_df = ride_df.groupby(['lat_bin', 'lng_bin']).agg(
        demand=('ride_type', 'size'),
        avg_demand_level=('demand_level', 'mean')
    ).reset_index().sort_values('demand', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        demand_df['lng_bin'], 
        demand_df['lat_bin'],
        c=demand_df['avg_demand_level'],
        s=demand_df['demand']/10,
        cmap='Reds',
        alpha=0.6
    )
    
    plt.colorbar(scatter, label='Demand Level')
    ax.set_xlabel("Longitude Bins")
    ax.set_ylabel("Latitude Bins")
    st.pyplot(fig)
    
    # Show top surge zones
    st.write("Top Potential Surge Pricing Zones:")
    surge_zones = demand_df.head(5).copy()
    surge_zones['lat_range'] = surge_zones['lat_bin'].apply(lambda x: f"{x*0.1:.1f}-{(x+1)*0.1:.1f}")
    surge_zones['lng_range'] = surge_zones['lng_bin'].apply(lambda x: f"{x*0.1:.1f}-{(x+1)*0.1:.1f}")
    st.dataframe(surge_zones[['lat_range', 'lng_range', 'demand', 'avg_demand_level']])

# Main App
def main():
    st.title("Ride Sharing Analytics Dashboard")
    
    # Load data
    try:
        ride_df, driver_df = load_data()
        
        # Basic Analytics
        basic_analytics(ride_df, driver_df)
        
        # Intermediate Analytics
        intermediate_analytics(ride_df)
        
        # Advanced Analytics
        advanced_analytics(ride_df)
        
        # Sidebar stats
        st.sidebar.header("Statistics")
        st.sidebar.write(f"Total Rides: {len(ride_df)}")
        st.sidebar.write(f"Unique Drivers: {driver_df['driver_id'].nunique()}")
        st.sidebar.write(f"Latest Ride: {ride_df['datetime'].max()}")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
