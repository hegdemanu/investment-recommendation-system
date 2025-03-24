# Data Directory

This directory stores all data files used by the Investment Recommendation System, including raw data, processed data, and uploaded files.

## Structure

### `/uploads`
- Contains CSV files uploaded by users through the web interface or API
- Files are automatically renamed with timestamps to prevent overwriting
- Supports historical stock data in CSV format

### `/raw`
- Stores raw data fetched from external sources like Yahoo Finance
- Maintains original data without any modifications
- Organized by data source and date of acquisition

### `/processed`
- Contains cleaned and preprocessed datasets ready for model training
- Features standardized format with consistent column names
- Includes derived features and technical indicators

## Data Format

### Stock Data Format
The system works with stock data in the following format:

| Date       | ticker   | Open  | High  | Low   | Price | Vol.     | Change % |
|------------|----------|-------|-------|-------|-------|----------|----------|
| 2023-01-01 | INFY.NS  | 1500  | 1520  | 1495  | 1510  | 2500000  | 0.5      |
| 2023-01-02 | INFY.NS  | 1510  | 1530  | 1505  | 1525  | 3000000  | 0.99     |
| 2023-01-01 | TCS.NS   | 3300  | 3350  | 3290  | 3320  | 1200000  | 0.6      |

### Minimum Required Columns
- `Date`: Trading date (YYYY-MM-DD format)
- `ticker`: Stock ticker symbol
- `Price`: Closing price for the day
- Additional columns like Open, High, Low, Vol. (Volume) are recommended

## Data Sources

The system can acquire data from multiple sources:
1. **User Uploads**: CSV files uploaded directly by users
2. **Yahoo Finance**: Via the yfinance Python package
3. **RapidAPI**: Using various financial data APIs
4. **NSE India**: For Indian stocks and indices

## Data Retention

- Raw data is retained indefinitely for historical analysis
- Processed data is regenerated when preprocessing parameters change
- Uploaded files are preserved unless explicitly deleted by users

## Data Privacy

- This directory may contain sensitive financial data
- No personally identifiable information should be stored here
- Internal use only - do not share raw data outside the organization

## Adding New Data

To add new data:
1. Place CSV files in the `/uploads` folder
2. Use the API endpoint `/api/upload-csv` to upload files programmatically
3. Use the `/api/fetch-data` endpoint to fetch from external sources

Data will be automatically processed and made available for model training. 