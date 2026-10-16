"""
Data Loader and Ingestion Module for Hotel Booking Demand Dataset.
IT3091 Machine Learning - Domain 6: Tourism & Hospitality
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_URLS = [
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
    "https://raw.githubusercontent.com/jokecamp/hotel-booking-demand/master/hotel_bookings.csv",
    "https://raw.githubusercontent.com/KeithGalli/hotel-booking-demand/master/hotel_bookings.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/hotel_bookings.csv"
]

DEFAULT_RAW_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "hotel_bookings.csv")


def download_dataset(destination_path: str = DEFAULT_RAW_PATH, force: bool = False) -> str:
    """
    Downloads the hotel bookings dataset if not already present.
    Verifies data integrity upon download.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    if os.path.exists(destination_path) and not force:
        print(f"[DataLoader] Dataset already exists at {destination_path} ({os.path.getsize(destination_path):,} bytes)")
        return destination_path

    print(f"[DataLoader] Downloading Hotel Booking Demand dataset...")
    success = False
    last_error = None
    
    for url in DATA_URLS:
        try:
            print(f"[DataLoader] Attempting download from: {url}")
            urllib.request.urlretrieve(url, destination_path)
            # Verify file validity
            df_test = pd.read_csv(destination_path, nrows=100)
            if "is_canceled" in df_test.columns or "hotel" in df_test.columns:
                print(f"[DataLoader] Successfully downloaded and verified dataset from {url}")
                success = True
                break
        except Exception as e:
            print(f"[DataLoader] Download failed from {url}: {e}")
            last_error = e

    if not success:
        raise RuntimeError(f"Failed to download Hotel Booking Demand dataset from all mirrors. Last error: {last_error}")
        
    return destination_path


def load_raw_data(data_path: str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Loads the raw Hotel Booking Demand dataset into a pandas DataFrame.
    """
    if not os.path.exists(data_path):
        download_dataset(data_path)
        
    df = pd.read_csv(data_path)
    print(f"[DataLoader] Loaded dataset with shape {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def get_data_dictionary() -> dict:
    """
    Returns the comprehensive 32-attribute Data Dictionary with business descriptions,
    types, and expected domain values.
    """
    return {
        "hotel": {"type": "Categorical", "desc": "Hotel type: 'Resort Hotel' (H1) or 'City Hotel' (H2)"},
        "is_canceled": {"type": "Binary Target", "desc": "Value indicating if booking was canceled (1) or not (0)"},
        "lead_time": {"type": "Numerical (Integer)", "desc": "Number of days between the entering date of the booking into PMS and the arrival date"},
        "arrival_date_year": {"type": "Numerical (Integer)", "desc": "Year of arrival date (2015-2017)"},
        "arrival_date_month": {"type": "Categorical", "desc": "Month of arrival date with 12 categories: 'January' to 'December'"},
        "arrival_date_week_number": {"type": "Numerical (Integer)", "desc": "Week number of year for arrival date"},
        "arrival_date_day_of_month": {"type": "Numerical (Integer)", "desc": "Day of the month of the arrival date"},
        "stays_in_weekend_nights": {"type": "Numerical (Integer)", "desc": "Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay"},
        "stays_in_week_nights": {"type": "Numerical (Integer)", "desc": "Number of week nights (Monday to Friday) the guest stayed or booked to stay"},
        "adults": {"type": "Numerical (Integer)", "desc": "Number of adults"},
        "children": {"type": "Numerical (Float/Int)", "desc": "Number of children (missing in 4 records)"},
        "babies": {"type": "Numerical (Integer)", "desc": "Number of babies"},
        "meal": {"type": "Categorical", "desc": "Type of meal booked: Undefined/SC (no meal), BB (Bed & Breakfast), HB (Half Board), FB (Full Board)"},
        "country": {"type": "Categorical", "desc": "Country of origin in ISO 3166-1 alpha-3 format (e.g. PRT, GBR, FRA)"},
        "market_segment": {"type": "Categorical", "desc": "Market segment designation: Direct, Corporate, Online TA, Offline TA/TO, Groups, Complementary, Aviation"},
        "distribution_channel": {"type": "Categorical", "desc": "Booking distribution channel: Direct, Corporate, TA/TO, GDS, Undefined"},
        "is_repeated_guest": {"type": "Binary", "desc": "Value indicating if the booking name was from a repeated guest (1) or not (0)"},
        "previous_cancellations": {"type": "Numerical (Integer)", "desc": "Number of previous bookings that were cancelled by the customer prior to the current booking"},
        "previous_bookings_not_canceled": {"type": "Numerical (Integer)", "desc": "Number of previous bookings not cancelled by the customer prior to the current booking"},
        "reserved_room_type": {"type": "Categorical", "desc": "Code of room type reserved (A to P)"},
        "assigned_room_type": {"type": "Categorical", "desc": "Code for the type of room assigned to the booking"},
        "booking_changes": {"type": "Numerical (Integer)", "desc": "Number of changes/amendments made to the booking from entering PMS to check-in or cancellation"},
        "deposit_type": {"type": "Categorical", "desc": "Deposit guarantee made: No Deposit, Non Refund, Refundable"},
        "agent": {"type": "Categorical / ID", "desc": "ID of the travel agency that made the booking (missing = direct booking)"},
        "company": {"type": "Categorical / ID", "desc": "ID of the company/entity that made or paid the booking (missing = private booking)"},
        "days_in_waiting_list": {"type": "Numerical (Integer)", "desc": "Number of days the booking was in the waiting list before it was confirmed"},
        "customer_type": {"type": "Categorical", "desc": "Type of booking: Contract, Group, Transient, Transient-Party"},
        "adr": {"type": "Numerical (Continuous)", "desc": "Average Daily Rate (total lodging revenue divided by total number of staying nights)"},
        "required_car_parking_spaces": {"type": "Numerical (Integer)", "desc": "Number of car parking spaces requested by the customer"},
        "total_of_special_requests": {"type": "Numerical (Integer)", "desc": "Number of special requests made by the customer (e.g. high floor, twin bed)"},
        "reservation_status": {"type": "Categorical (TARGET LEAKAGE)", "desc": "Final reservation status: Canceled, Check-Out, No-Show. MUST BE DROPPED BEFORE MODELING"},
        "reservation_status_date": {"type": "Date (TARGET LEAKAGE)", "desc": "Date at which the last status was set. MUST BE DROPPED BEFORE MODELING"}
    }


if __name__ == "__main__":
    df = load_raw_data()
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info Summary:")
    print(df.info())
