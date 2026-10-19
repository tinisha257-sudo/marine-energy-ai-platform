import pandas as pd
import numpy as np

# --------------------------------------------------
# MARINE ENERGY DATA PROCESSING
# --------------------------------------------------

FILE_NAME = "WEC_Sydney_100.csv"


def load_dataset():
    """Load the marine energy dataset."""
    df = pd.read_csv(FILE_NAME)
    return df


def get_wec_power_data(df):
    """
    Convert Power1 ... Power100
    into WEC-wise analysis.
    """

    power_columns = [
        col for col in df.columns
        if col.startswith("Power")
    ]

    wec_data = []

    for column in power_columns:

        wec_id = column.replace("Power", "")

        power_values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        wec_data.append({
            "WEC_ID": f"WEC_{wec_id}",
            "Average_Power": power_values.mean(),
            "Maximum_Power": power_values.max(),
            "Minimum_Power": power_values.min(),
            "Power_Std": power_values.std()
        })

    return pd.DataFrame(wec_data)


def get_system_summary(df, wec_data):

    total_power = pd.to_numeric(
        df["Total_Power"],
        errors="coerce"
    )

    return {
        "Total_WECs": len(wec_data),
        "Data_Records": len(df),
        "Average_Total_Power": total_power.mean(),
        "Maximum_Total_Power": total_power.max(),
        "Minimum_Total_Power": total_power.min()
    }


def get_top_wecs(wec_data, n=10):

    return wec_data.sort_values(
        by="Average_Power",
        ascending=False
    ).head(n)


def get_low_wecs(wec_data, n=10):

    return wec_data.sort_values(
        by="Average_Power",
        ascending=True
    ).head(n)


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("\nMARINE ENERGY DATA PROCESSING")
    print("=" * 50)

    # Load dataset
    df = load_dataset()

    print("Dataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Process WEC data
    wec_data = get_wec_power_data(df)

    print("\nWEC Analysis")
    print("-" * 50)

    print(wec_data.head())

    # System summary
    summary = get_system_summary(
        df,
        wec_data
    )

    print("\nSystem Summary")
    print("-" * 50)

    for key, value in summary.items():

        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")

        else:
            print(f"{key}: {value}")

    # Top WECs
    print("\nTop 10 Performing WECs")
    print("-" * 50)

    print(
        get_top_wecs(wec_data).to_string(index=False)
    )

    # Lowest WECs
    print("\nLowest 10 Performing WECs")
    print("-" * 50)

    print(
        get_low_wecs(wec_data).to_string(index=False)
    )