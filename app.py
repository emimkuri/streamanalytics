import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, avg, sum as spark_sum, window, expr
from datetime import datetime, timedelta
import json

# Initialize Spark Session
def init_spark():
    spark = SparkSession.builder \
    .appName("AzureBlobDashboard") \
    .config("spark.jars.packages",
            "org.apache.hadoop:hadoop-azure:3.3.1,"
            "com.microsoft.azure:azure-storage:8.6.6") \
    .config("fs.azure.account.key.iesstsabdbaa.blob.core.windows.net",
            "yfqMW8gf8u+M5pOW33Q5gtRTFBJQXStVK4K2rlCVVzxlrRG21Sh7MVj06uExoL86Npb7HWWgxYUe+ASthUr6/g==") \
    .config("fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem") \
    .config("fs.azure.account.keyprovider.iesstsabdbaa.blob.core.windows.net",
            "org.apache.hadoop.fs.azure.SimpleKeyProvider") \
    .getOrCreate()
    
    # Configure Azure Blob Storage access
    spark.conf.set("fs.azure.account.key.iesstsabdbaa.blob.core.windows.net", "yfqMW8gf8u+M5pOW33Q5gtRTFBJQXStVK4K2rlCVVzxlrRG21Sh7MVj06uExoL86Npb7HWWgxYUe+ASthUr6/g==")
    return spark

# Load data from Blob Storage
def load_data(spark):
    ride_data_path = "wasbs://group7@iesstsabdbaa.blob.core.windows.net/optimized_ride_data/*"
    driver_data_path = "wasbs://group7@iesstsabdbaa.blob.core.windows.net/optimized_driver_data/*"
    
    ride_df = spark.read.json(ride_data_path)
    driver_df = spark.read.json(driver_data_path)
    
    return ride_df, driver_df

# Basic Analytics
def basic_analytics(ride_df):
    st.header("Basic Analytics")
    
    # Ride status counts
    status_counts = ride_df.groupBy("status").agg(count("*").alias("count")).toPandas()
    
    col1, col2, col3 = st.columns(3)
    requested = status_counts[status_counts['status'] == 'requested']['count'].values[0]
    accepted = status_counts[status_counts['status'] == 'accepted']['count'].values[0]
    completed = status_counts[status_counts['status'] == 'completed']['count'].values[0]
    
    col1.metric("Requested Rides", requested)
    col2.metric("Accepted Rides", accepted)
    col3.metric("Completed Rides", completed)
    
    # Rides by type
    st.subheader("Rides by Type")
    ride_type_counts = ride_df.groupBy("ride_type").agg(count("*").alias("count")).toPandas()
    fig1, ax1 = plt.subplots()
    ax1.pie(ride_type_counts['count'], labels=ride_type_counts['ride_type'], autopct='%1.1f%%')
    st.pyplot(fig1)
    
    # Rides by vehicle type
    st.subheader("Rides by Vehicle Type")
    vehicle_type_counts = ride_df.join(driver_df, ride_df.passenger_id == driver_df.driver_id) \
                               .groupBy("vehicle_type").agg(count("*").alias("count")).toPandas()
    fig2, ax2 = plt.subplots()
    sns.barplot(x='vehicle_type', y='count', data=vehicle_type_counts, ax=ax2)
    st.pyplot(fig2)

# Intermediate Analytics
def intermediate_analytics(ride_df):
    st.header("Intermediate Analytics")
    
    # Cancellation rates
    st.subheader("Cancellation Rates by Ride Type")
    total_rides = ride_df.groupBy("ride_type").agg(count("*").alias("total_rides")).toPandas()
    cancelled_rides = ride_df.filter(col("status") == "cancelled") \
                           .groupBy("ride_type").agg(count("*").alias("cancelled_rides")).toPandas()
    
    cancellation_rates = pd.merge(total_rides, cancelled_rides, on="ride_type", how="left")
    cancellation_rates['cancellation_rate'] = (cancellation_rates['cancelled_rides'] / cancellation_rates['total_rides']) * 100
    cancellation_rates = cancellation_rates.fillna(0)
    
    fig, ax = plt.subplots()
    sns.barplot(x='ride_type', y='cancellation_rate', data=cancellation_rates, ax=ax)
    ax.set_ylabel("Cancellation Rate (%)")
    st.pyplot(fig)

# Advanced Analytics
def advanced_analytics(ride_df):
    st.header("Advanced Analytics")
    
    # Surge pricing prediction
    st.subheader("Surge Pricing Zone Prediction")
    
    # Calculate demand by area (simplified using lat/lng bins)
    demand_df = ride_df.withColumn("lat_bin", (col("pickup_lat") / 0.1).cast("int")) \
                      .withColumn("lng_bin", (col("pickup_lng") / 0.1).cast("int")) \
                      .groupBy("lat_bin", "lng_bin") \
                      .agg(count("*").alias("demand"), 
                           avg("demand_level").alias("avg_demand_level")) \
                      .orderBy("demand", ascending=False)
    
    demand_pd = demand_df.limit(20).toPandas()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(demand_pd['lng_bin'], demand_pd['lat_bin'], 
                        c=demand_pd['avg_demand_level'], s=demand_pd['demand']/10,
                        cmap='Reds', alpha=0.6)
    
    plt.colorbar(scatter, label='Demand Level')
    ax.set_xlabel("Longitude Bins")
    ax.set_ylabel("Latitude Bins")
    ax.set_title("Potential Surge Pricing Zones (Size = Demand, Color = Demand Level)")
    st.pyplot(fig)
    
    # Show top surge zones
    st.write("Top Potential Surge Pricing Zones:")
    surge_zones = demand_pd[['lat_bin', 'lng_bin', 'demand', 'avg_demand_level']].copy()
    surge_zones['lat_range'] = surge_zones['lat_bin'].apply(lambda x: f"{x*0.1:.1f}-{(x+1)*0.1:.1f}")
    surge_zones['lng_range'] = surge_zones['lng_bin'].apply(lambda x: f"{x*0.1:.1f}-{(x+1)*0.1:.1f}")
    st.dataframe(surge_zones[['lat_range', 'lng_range', 'demand', 'avg_demand_level']].head(5))

# Main App
def main():
    st.title("Ride Sharing Analytics Dashboard")
    
    # Initialize Spark
    spark = init_spark()
    
    # Load data
    ride_df, driver_df = load_data(spark)
    
    # Register DataFrames as temporary views for SQL queries
    ride_df.createOrReplaceTempView("ride_data")
    driver_df.createOrReplaceTempView("driver_data")
    
    # Basic Analytics
    basic_analytics(ride_df)
    
    # Intermediate Analytics
    intermediate_analytics(ride_df)
    
    # Advanced Analytics
    advanced_analytics(ride_df)
    
    # Add some streaming statistics
    st.sidebar.header("Streaming Statistics")
    st.sidebar.write(f"Total Rides Processed: {ride_df.count()}")
    st.sidebar.write(f"Total Drivers: {driver_df.select('driver_id').distinct().count()}")
    
    # Data freshness
    latest_ride = ride_df.selectExpr("max(datetime)").collect()[0][0]
    st.sidebar.write(f"Latest Ride: {latest_ride}")

if __name__ == "__main__":
    main()
