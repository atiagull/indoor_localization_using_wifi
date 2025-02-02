import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib

st.title("Data Table")

df = pd.read_csv("./data/JUIndoorLoc-Training-data.csv")
st.write(df.head())

test_df = pd.read_csv("./test_data.csv")

if 'Cid_encoded' in test_df.columns:
    y_test = test_df['Cid_encoded']
    X_test = test_df.drop(columns=['Cid_encoded'])
else:
    st.error("Error: 'Cid_encoded' column not found in test_data.csv")
    X_test, y_test = None, None

def plot_missing_values_heatmap(df):
    """Function to visualize missing values across selected APs using a heatmap."""
    # Extract AP columns
    ap_columns = [col for col in df.columns if col.startswith("AP")]

    # Multiselect for APs
    selected_aps = st.multiselect("Select APs for the heatmap", ap_columns, default=ap_columns[:5])

    if selected_aps:
        # Replace -110 with NaN (missing values)
        df_missing = df[selected_aps].replace(-110, pd.NA)

        st.write("### Missing Values Heatmap (Selected AP Signal Strength)")

        plt.figure(figsize=(12, 6))
        sns.heatmap(df_missing.isna(), cmap="viridis", cbar=False, yticklabels=False)

        plt.xlabel("Selected Access Points (AP)")
        plt.ylabel("Records")
        plt.title("Heatmap of Missing Values Across Selected APs")

        st.pyplot(plt)

        st.write(
            "#### Drop columns: ",
            "The checkbox below allows you to drop Access Point (AP) columns where more than 95% of values are missing (represented by -110 dBm)."
        )

        drop_aps = st.checkbox("Drop APs with more than 95% missing values", value=False)

        return drop_aps, df_missing

    else:
        st.warning("Please select at least one AP to visualize.")
        return False, df

def drop_missing_aps(df, drop_aps):
    """Function to drop APs with more than 95% missing values."""
    if drop_aps:
        # Extract AP columns
        ap_columns = [col for col in df.columns if col.startswith("AP")]
        
        # Remove APs with more than 95% of values at -110 dBm (missing data)
        ap_missing_ratio = (df[ap_columns] == -110).mean()  # Calculate % of -110 values
        valid_aps = ap_missing_ratio[ap_missing_ratio < 0.95].index  # Keep APs with <95% missing
        df_filtered = df[["Ts", "Hpr","Cid","Did"] + list(valid_aps)]  # Keep valid APs only
        return df_filtered
    else:
        return df

def plot_signal_strength(df):
    """Function to plot signal strength variations across different APs."""
    # Extract AP columns
    ap_columns = [col for col in df.columns if col.startswith("AP")]

    # Sidebar selection for APs
    selected_aps = st.multiselect("Select APs to visualize", ap_columns, default=ap_columns[:5])

    
    st.write("### Signal Strength Variations Across Selected APs")

    if selected_aps:
        plt.figure(figsize=(12, 6))
        for ap in selected_aps:
            plt.plot(df.index, df[ap], label=ap, alpha=0.7)

        plt.xlabel("Index")
        plt.ylabel("Signal Strength (dBm)")
        plt.title("Signal Strength Variations Across APs")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.warning("Please select at least one AP to visualize.")

def plot_signal_strength_over_time(df):
    """Function to plot average signal strength over time."""
    # Extract AP columns
    ap_columns = [col for col in df.columns if col.startswith("AP")]

    # Compute average signal strength across all APs
    df["Avg_Signal_Strength"] = df[ap_columns].mean(axis=1)

    # Sort by timestamp
    df = df.sort_values(by="Ts")
    df["Ts"] = pd.to_datetime(df["Ts"], unit='ms')

    # Plot Time Series as a Scatter Plot
    st.write("### Average Signal Strength Over Time\n This graph shows the time intervals during which data is collected and average value of signal strength across various timestamps ")

    plt.figure(figsize=(12, 6))
    plt.scatter(df["Ts"], df["Avg_Signal_Strength"], alpha=0.7, color='y', edgecolors='k')

    plt.xlabel("Timestamp")
    plt.ylabel("Average Signal Strength (dBm)")
    plt.title("Average Signal Strength Over Time (Scatter Plot)")
    plt.grid(True)
    st.pyplot(plt)

def plot_filtered_ap_strength_day_night(df):
    """Function to compare AP signal strength between day and night after filtering missing APs."""
    # Convert Timestamp to datetime format
    df["datetime"] = pd.to_datetime(df["Ts"], unit='ms')

    # Extract hour from the datetime
    df["hour"] = df["datetime"].dt.hour

    # Define day (6 AM - 6 PM) and night (6 PM - 6 AM)
    df["time_period"] = df["hour"].apply(lambda x: "Day" if 6 <= x < 18 else "Night")

    # Extract AP columns
    ap_columns = [col for col in df.columns if col.startswith("AP")]

    # Multiselect to allow user to select AP columns
    selected_aps = st.multiselect("Select APs for the analysis", ap_columns, default=ap_columns[:30])


    if selected_aps:
        # Compute average signal strength per time period for the selected APs
        avg_signal_by_period = df.groupby("time_period")[selected_aps].mean().T

        # Sort APs by highest day signal strength
        avg_signal_by_period = avg_signal_by_period.sort_values(by="Day", ascending=False)

        
        st.write("### AP Signal Strength: Day vs. Night")

        plt.figure(figsize=(12, 6))
        avg_signal_by_period.plot(kind="bar", figsize=(12, 6), alpha=0.7)

        plt.xlabel("Access Points (AP)")
        plt.ylabel("Average Signal Strength (dBm)")
        plt.title("Comparison of AP Signal Strength Between Day and Night (Filtered)")
        plt.xticks(rotation=45)  # Rotate labels for readability
        plt.grid(True)

        st.pyplot(plt)
    else:
        st.warning("Please select at least one AP to visualize.")

def plot_human_presence_trends(df):
    """Function to plot human presence trends over time with date range selection."""
    
    # Convert 'Ts' column to datetime format if not already
    df["Ts"] = pd.to_datetime(df["Ts"], unit='ms')
    
    # Allow user to select a date range
    st.write("### Human Presence Trends\n This graph shows human presence (1 for presence, 0 for absence) over time.")
    min_date = df["Ts"].min().date()
    max_date = df["Ts"].max().date()

    # Date range picker
    start_date, end_date = st.date_input(
        "Select a time interval", 
        [min_date, max_date], 
        min_value=min_date, 
        max_value=max_date
    )

    # Filter data based on the selected date range
    df_filtered = df[(df["Ts"].dt.date >= start_date) & (df["Ts"].dt.date <= end_date)]

    # Check if the filtered data is empty
    if df_filtered.empty:
        st.warning("No data available for the selected time range.")
        return

    # Plot Time Series for Human Presence (Scatter Plot)
    plt.figure(figsize=(12, 6))
    plt.scatter(df_filtered["Ts"], df_filtered["Hpr"], alpha=0.7, color='g', edgecolors='k')

    plt.xlabel("Timestamp")
    plt.ylabel("Human Presence (Hpr)")
    plt.title(f"Human Presence Over Time (From {start_date} to {end_date})")
    plt.grid(True)
    st.pyplot(plt)

def plot_human_presence_overlay(df):
    
    
    # check to see if 'Hpr' and 'Cid' columns are in the dataframe
    if "Hpr" not in df.columns or "Cid" not in df.columns:
        st.warning("'Hpr' or 'Cid' column is missing from the dataframe. Please check the data processing steps.")
        return

    # Group by 'Cid' and calculate the sum of human presence (Hpr) across each location (Cid)
    human_presence_per_location = df.groupby("Cid")["Hpr"].sum().reset_index()

    # Sort the data by human presence (sum of Hpr)
    human_presence_per_location = human_presence_per_location.sort_values(by="Hpr", ascending=False)

    # Create a heatmap or bar plot to visualize the distribution of human presence across Cid
    st.write("### Human Presence Across Locations (Cid)\n This graph shows human presence trend across various locations in the building.")

    # Plot as a heatmap
    plt.figure(figsize=(12, 6))
    # Pivot the data to create a 2D structure (locations as rows, Hpr as values)
    heatmap_data = human_presence_per_location.pivot_table(index="Cid", values="Hpr", aggfunc="sum")
    
    
    sns.heatmap(heatmap_data.T, cmap="YlGnBu", annot=True, cbar=True, fmt="d", linewidths=0.5)

    plt.xlabel("Cid (Location)")
    plt.ylabel("Human Presence (Hpr)")
    plt.title("Human Presence Overlay Across Locations (Cid)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.grid(True)

    st.pyplot(plt)

def plot_device_usage(df):
    """Function to plot the number of fingerprints collected from each device."""
    # Group by device (Did) and count the occurrences
    device_usage = df['Did'].value_counts().reset_index()
    device_usage.columns = ['Device', 'Fingerprint Count']

    
    st.write("### Bar Chart of Device Usage")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=device_usage, x='Device', y='Fingerprint Count', palette='viridis')

    plt.xlabel("Device ID (Did)")
    plt.ylabel("Fingerprint Count")
    plt.title("Count of Fingerprints Collected from Each Device")
    plt.xticks(rotation=90)
    plt.grid(True)
    st.pyplot(plt)

def plot_device_performance_comparison(df):
    """Function to compare device performance using boxplots for signal strength."""
    # Extract AP columns for signal strength
    ap_columns = [col for col in df.columns if col.startswith("AP")]
    
    # Reshape the data to long format for seaborn
    df_long = pd.melt(df, id_vars=["Did"], value_vars=ap_columns, var_name="AP", value_name="Signal Strength")

    # Plotting the boxplot to compare signal strength per device
    st.write("### Device Performance Comparison (Boxplot of Signal Strength)")
  
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_long, x='Did', y='Signal Strength', palette='coolwarm', showfliers=False)

    plt.xlabel("Device ID (Did)")
    plt.ylabel("Signal Strength (dBm)")
    plt.title("Device Performance: Signal Strength Comparison")
    plt.xticks(rotation=90)
    plt.grid(True)
    st.pyplot(plt)

# Show the missing values heatmap and ask if the user wants to drop APs with >95% missing values
drop_aps, df_missing = plot_missing_values_heatmap(df)

# Drop the APs if the user selected to do so
df_filtered = drop_missing_aps(df, drop_aps)

# Plot the other graphs with the filtered dataset
# plot_signal_strength(df_filtered)
# plot_signal_strength_over_time(df_filtered)
# plot_filtered_ap_strength_day_night(df_filtered)
# plot_human_presence_trends(df_filtered)
# plot_human_presence_overlay(df_filtered)
# plot_device_usage(df_filtered)
# plot_device_performance_comparison(df_filtered)
# Function to train and test KNN

def load_and_test_model(model_name):
    """Load a trained model and evaluate it on the test set."""
    model_path = f"models/{model_name}.joblib"
    
    try:
        model = joblib.load(model_path)  # Load the model
        y_pred = model.predict(X_test) 
        accuracy = accuracy_score(y_test, y_pred)  
        st.write(f"### Model: {model_name} - Test Accuracy: {accuracy:.4f}")
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")

def main():
    
    st.write("## Choose Analysis Type")
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        options=["Signal Strength Analysis", "Human Presence Trend", "Device Analysis"],
        key="analysis_type_select"
    )

    if analysis_type == "Signal Strength Analysis":
        plot_signal_strength(df_filtered)
        plot_signal_strength_over_time(df_filtered)
        plot_filtered_ap_strength_day_night(df_filtered)
    elif analysis_type == "Human Presence Trend":
        plot_human_presence_trends(df_filtered)
        plot_human_presence_overlay(df_filtered)
    elif analysis_type == "Device Analysis":
        plot_device_usage(df_filtered)
        plot_device_performance_comparison(df_filtered)

    st.write("## Model Selection")
    model_choice = st.selectbox("Select a model to test", ["KNN", "XGBoost"])

    if X_test is not None and y_test is not None:
        model_filenames = {
            "KNN": "knn_model",
            "XGBoost": "xgb_model"
        }
        load_and_test_model(model_filenames[model_choice])


if __name__ == "__main__":
    main()