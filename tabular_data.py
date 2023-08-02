import pandas as pd
import ast

def remove_rows_with_missing_columns(df):
    """
    Remove rows with missing values in specific rating columns.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: DataFrame with rows containing missing ratings dropped.
    """
    df = df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating',
                           'Location_rating', 'Check-in_rating', 'Value_rating'])
    return df


def combine_description_settings(df):
    """
    Combine data from a problematic row to fix shifting issue, and process description and amenities columns.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: DataFrame with fixed values in the problematic row and processed description and amenities columns.
    """
    # There seems to be a problematic row where every element of a column has shifted to the next column. (except for ID)
    # Fixing this manually and then moving ahead.
    problematic_row = df.loc[df['Title'] == 'Stunning Cotswolds Water Park'].index[0]
    df.at[problematic_row, 'Title'] = 'Stunning Cotswolds Water Park, sleeps 6 with pool'
    df.at[problematic_row, 'Description'] = df['Amenities'][problematic_row]
    df.at[problematic_row, 'Amenities'] = df['Location'][problematic_row]
    df.at[problematic_row, 'Location'] = df['guests'][problematic_row]
    df.at[problematic_row, 'guests'] = df['beds'][problematic_row]
    df.at[problematic_row, 'beds'] = df['bathrooms'][problematic_row]
    df.at[problematic_row, 'bathrooms'] = df['Price_Night'][problematic_row]
    df.at[problematic_row, 'Price_Night'] = df['Cleanliness_rating'][problematic_row]
    df.at[problematic_row, 'Cleanliness_rating'] = df['Accuracy_rating'][problematic_row]
    df.at[problematic_row, 'Accuracy_rating'] = df['Communication_rating'][problematic_row]
    df.at[problematic_row, 'Communication_rating'] = df['bedrooms'][problematic_row]
    df.at[problematic_row, 'bedrooms'] = df['Unnamed: 19'][problematic_row]

    
    def clean_and_combine_list_items(item):
        """
        Helper function to clean and combine list items into a single string.

        Parameters:
            item (str): The string containing list items.

        Returns:
            str: The cleaned and combined string.
        """
        if isinstance(item, str):
            return ', '.join(v.strip() for v in ast.literal_eval(item) if isinstance(v, str))
        return ''

    # Cleaning and combining description and amenities columns
    df['Description'] = df['Description'].str.replace('About this space', '').str.strip().apply(clean_and_combine_list_items)
    df['Amenities'] = df['Amenities'].str.replace('What this place offers', '').str.strip().apply(clean_and_combine_list_items)
    # Removing the first comma
    df['Amenities'] = df['Amenities'].str.replace(',', '', 1)
    return df
    

def set_default_feature_values(df):
    """
    Set default feature values for empty entries in the guests, beds, bathrooms, and bedrooms columns.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: DataFrame with default values set for empty entries in certain columns.
    """
    df["guests"].fillna(1, inplace=True)
    df["beds"].fillna(1, inplace=True)
    df["bathrooms"].fillna(1, inplace=True)
    df["bedrooms"].fillna(1, inplace=True)
    return df

def convert_dtypes(df):
    """
    Convert specific columns to the appropriate data types.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: DataFrame with appropriate data types for specific columns.
    """
    # Convert the DataFrame columns to appropriate data types
    df = df.convert_dtypes()
    
    # Handle the issue where a value has been interchanged between 'url' and 'Communication_rating'
    # as the value in communication_rating is a url and the value in url is '46'
    problematic_row = df.loc[df['url']=='46'].index[0]
    temp_value = df.at[problematic_row, 'Communication_rating']
    df.at[problematic_row, 'Communication_rating'] = df.at[problematic_row, 'url']  
    df.at[problematic_row, 'url'] = temp_value
    
    # Assume the rating in 'Communication_rating' is 4.6 and not '46' and safely convert the datatype to float
    df.at[problematic_row, 'Communication_rating'] = 4.6
    df['Communication_rating'] = df['Communication_rating'].astype('Float64')

    # Fix the column 'guests'
    df['guests'] = pd.to_numeric(df['guests'], errors='coerce')
    df['guests'] = df['guests'].astype('Int64')

    # Fix the column 'bedrooms'
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bedrooms'] = df['bedrooms'].astype("Int64")

    # Drop the column Unnamed 19:
    df = df.drop('Unnamed: 19', axis=1)
    
    # Reset the index
    df = df.reset_index(drop=True)

    return df


def clean_tabular_data(df):
    """
    Apply a series of data cleaning steps to the DataFrame.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: Cleaned DataFrame with selected columns and missing values removed/filled.
    """
    # Step 1: Remove rows with missing columns in rating
    df1 = remove_rows_with_missing_columns(df)
    # Step 2: Combine data from a problematic row and process description and amenities columns
    df2 = combine_description_settings(df1)
    # Step 3: Set default feature values for empty entries in guests, beds, bathrooms, and bedrooms columns
    df3 = set_default_feature_values(df2)
    # Step 4: Correct the data_types for the df and clean it thoroughly making it presentable
    df4 = convert_dtypes(df3)

    return df4


def load_airbnb(df, label):
    """
    Extract features and labels from the DataFrame.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        label (str): The name of the column representing the label.
    
    Returns:
        DataFrame, Series: Features DataFrame and labels Series.
    """
    labels = df[label]
    # Drop irrelevant columns (e.g., 'Category', 'ID', 'Title', 'Description', 'Amenities', 'Location', 'url') to get the features DataFrame
    features = df.drop(columns=[label, 'Category', 'ID', 'Title', 'Description', 'Amenities', 'Location', 'url'])
    return features, labels


if __name__ == "__main__":
    # Load the data from the CSV file
    df = pd.read_csv("C:/Users/Anany/OneDrive/Desktop/Github/AIcore/Modelling_Airbnb's_property_listing_dataset/airbnb-property-listing/tabular_data/listing.csv")
    # Clean the data
    clean_df = clean_tabular_data(df)
    # Save the cleaned data to a new CSV file
    clean_df.to_csv("C:/Users/Anany/OneDrive/Desktop/Github/AIcore/Modelling_Airbnb's_property_listing_dataset/airbnb-property-listing/tabular_data/clean_listing.csv")
