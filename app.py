import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Configuration
st.set_page_config(layout="wide")

# Azure Blob Storage config
ACCOUNT_URL = "https://iesstsabdbaa.blob.core.windows.net/"
CONTAINER_NAME = "group7"
RIDE_PATH = "optimized_ride_data"
DRIVER_PATH = "optimized_driver_data"
ACCESS_KEY = "yfqMW8gf8u+M5pOW33Q5gtRTFBJQXStVK4K2rlCVVzxlrRG21Sh7MVj06uExoL86Npb7HWWgxYUe+ASthUr6/g=="

# Connect to Azure Blob Storage
blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=ACCESS_KEY)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_parquet_from_blob(path_prefix):
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_list = container_client.list_blobs(name_starts_with=path_prefix)
    dataframes = []
    for blob in blob_list:
        blob_client = container_client.get_blob_client(blob.name)
        stream = BytesIO(blob_client.download_blob().readall())
        try:
            df = pd.read_parquet(stream)
            dataframes.append(df)
        except Exception:
            continue
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_data():
    ride_df = load_parquet_from_blob(RIDE_PATH)
    driver_df = load_parquet_from_blob(DRIVER_PATH)
    return ride_df, driver_df

# Basic Analytics
def basic_analytics(ride_df, driver_df):
    st.header("Basic Analytics")
    
    if not ride_df.empty:
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
        if not driver_df.empty:
            st.subheader("Rides by Vehicle Type")
            merged_df = pd.merge(ride_df, driver_df, left_on='passenger_id', right_on='driver_id', how='left')
            fig2, ax2 = plt.subplots()
            sns.countplot(x='vehicle_type', data=merged_df, ax=ax2)
            st.pyplot(fig2)
    else:
        st.warning("No ride data available")

# Intermediate Analytics
def intermediate_analytics(ride_df):
    st.header("Intermediate Analytics")
    
    if not ride_df.empty:
        # Cancellation rates
        st.subheader("Cancellation Rates by Ride Type")
        total_rides = ride_df.groupby('ride_type').size()
        cancelled_rides = ride_df[ride_df['status'] == 'cancelled'].groupby('ride_type').size()
        cancellation_rates = (cancelled_rides / total_rides * 100).fillna(0)
        
        fig, ax = plt.subplots()
        cancellation_rates.plot.bar(ax=ax)
        ax.set_ylabel("Cancellation Rate (%)")
        st.pyplot(fig)
    else:
        st.warning("No ride data available")

# Advanced Analytics
def advanced_analytics(ride_df):
    st.header("Advanced Analytics")
    
    if not ride_df.empty and 'pickup_lat' in ride_df.columns and 'pickup_lng' in ride_df.columns:
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
    else:
        st.warning("Required data columns not available")

# Main App
def main():
    st.title("Ride Sharing Analytics Dashboard")
    
    # Load data
    try:
        ride_df, driver_df = load_data()
        
        if not ride_df.empty:
            # Basic Analytics
            basic_analytics(ride_df, driver_df)
            
            # Intermediate Analytics
            intermediate_analytics(ride_df)
            
            # Advanced Analytics
            advanced_analytics(ride_df)
            
            # Sidebar stats
            st.sidebar.header("Statistics")
            st.sidebar.write(f"Total Rides: {len(ride_df)}")
            if not driver_df.empty:
                st.sidebar.write(f"Unique Drivers: {driver_df['driver_id'].nunique()}")
            if 'datetime' in ride_df.columns:
                st.sidebar.write(f"Latest Ride: {ride_df['datetime'].max()}")
        else:
            st.error("No ride data loaded successfully")
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
