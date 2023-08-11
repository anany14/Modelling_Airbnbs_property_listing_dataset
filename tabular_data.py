import pandas as pd
import ast

def remove_rows_with_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def fix_problematic_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixing problematic rows manually as these errors are very specific to this dataframe.

    Parameters:
        df (DataFrame): The Input Dataframe.

    Returns:
        DataFrame : DataFrame with problematic rows fixed. 
    
    """

    # There seems to be a problematic row where every element of a column has shifted to the next column. (except for ID)
    #fixing this manually 

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

    # Handle the issue where a value has been interchanged between 'url' and 'Communication_rating'
    # as the value in communication_rating is a url and the value in url is '46'
    problematic_row = df.loc[df['url']=='46'].index[0]
    temp_value = df.at[problematic_row, 'Communication_rating']
    df.at[problematic_row, 'Communication_rating'] = df.at[problematic_row, 'url']  
    df.at[problematic_row, 'url'] = temp_value

    # Assume the rating in 'Communication_rating' is 4.6 and not '46' as the value can't be greater than 5 in that column
    df.at[problematic_row, 'Communication_rating'] = 4.6

    return df

def combine_description_settings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine data from a problematic row to fix shifting issue, and process description and amenities columns.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: DataFrame with processed description and amenities columns.
    """
    
    def clean_and_combine_list_items(item: str) -> str:
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
    

def set_default_feature_values(df: pd.DataFrame) -> pd.DataFrame:
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

def convert_dtypes_and_optimise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert specific columns to the appropriate data types.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: DataFrame with appropriate data types for specific columns.
    """

    # Convert the DataFrame columns to appropriate data types
    df = df.convert_dtypes()
    
    df['Communication_rating'] = df['Communication_rating'].astype('Float64')

    # Fix the column 'guests'
    df['guests'] = pd.to_numeric(df['guests'], errors='coerce')
    df['guests'] = df['guests'].astype('Int64')

    # Fix the column 'bedrooms'
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bedrooms'] = df['bedrooms'].astype("Int64")

    print(df.dtypes)

    # Drop the column Unnamed 19:
    df = df.drop('Unnamed: 19', axis=1)
    
    # Reset the index
    df = df.reset_index(drop=True)

    return df


def clean_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of data cleaning steps to the DataFrame.
    
    Parameters:
        df (DataFrame): The input DataFrame.
    
    Returns:
        DataFrame: Cleaned DataFrame with selected columns and missing values removed/filled.
    """
    # Step 1: Remove rows with missing columns in rating
    df1 = remove_rows_with_missing_columns(df)
    # Step 2: Combine data from a problematic row
    df2 = fix_problematic_rows(df1)
    # Step 3: Process description and amenities columns
    df3 = combine_description_settings(df2)
    # Step 3: Set default feature values for empty entries in guests, beds, bathrooms, and bedrooms columns
    df4 = set_default_feature_values(df3)
    # Step 4: Correct the data_types for the df and clean it thoroughly making it presentable
    df5 = convert_dtypes_and_optimise_df(df4)

    return df5


def load_airbnb(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame,pd.Series]:
    """
    Extract features and labels from the DataFrame.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        label (str): The name of the column representing the label.
    
    Returns:
        tuple[pd.DataFrame,pd.Series]: Features DataFrame and labels Series.
    """
    labels = df[label]
    # Drop irrelevant columns (e.g., 'Category', 'ID', 'Title', 'Description', 'Amenities', 'Location', 'url') to get the features DataFrame
    features = df.drop(columns=[label, 'Category', 'ID', 'Title', 'Description', 'Amenities', 'Location', 'url'])
    return features, labels


if __name__ == "__main__":
    # Load the data from the CSV file
    df = pd.read_csv("C:/Users/Anany/OneDrive/Desktop/Github/AIcore/Modelling_Airbnbs_property_listing_dataset/airbnb-property-listing/tabular_data/listing.csv")
    # Clean the data
    clean_df = clean_tabular_data(df)
    # Save the cleaned data to a new CSV file
    clean_df.to_csv("C:/Users/Anany/OneDrive/Desktop/Github/AIcore/Modelling_Airbnbs_property_listing_dataset/airbnb-property-listing/tabular_data/clean_listing.csv")
    # printing features and labels
    features,labels = load_airbnb(clean_df,label='Price_Night')
    print("printing first 10 features and labels \n",features[:10],labels[:10])