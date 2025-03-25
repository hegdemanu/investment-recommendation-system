# Data Directory

This directory contains the data files used by the Investment Recommendation System.

## Structure

- **stocks/**: Contains historical stock price data in CSV format
  - Format: `TICKER_Sorted.csv` (e.g., `ADNA_Historical_Sorted.csv`)
  - Each file contains columns for Date, Price, Open, High, Low, Vol., Change %, ticker

- **mutual_funds/**: Contains mutual fund NAV data in CSV/Excel format
  - Format: `FUND_NAME.xlsx` (e.g., `Tata_MF.xlsx`)
  - Each file contains columns for Date, NAV

- **raw/**: Contains original, unprocessed data files
  - Used for backup and reference

- **processed/**: Contains preprocessed data files
  - Contains data after cleaning, feature engineering, and technical indicator calculation

- **uploads/**: Contains user-uploaded data files for analysis
  - Temporary storage for user provided data through the web interface

## Data Format Requirements

### Stock Data Format

Required columns:
- Date (YYYY-MM-DD format)
- Price (closing price)
- Open (opening price)
- High (day's high)
- Low (day's low)
- Vol. (trading volume)
- Change % (percentage change)
- ticker (stock symbol)

### Mutual Fund Data Format

Required columns:
- Date (YYYY-MM-DD format)
- NAV (Net Asset Value)

## Sample Data

The system comes with sample data for testing and demonstration purposes.
The sample data includes historical prices for Indian stocks from multiple sectors:

- Technology: Infosys, Wipro
- Banking: ICICI
- Oil & Gas: ONGC
- Consumer: Nestle, Titan
- Pharma: Apollo
- Financial Services: Bajaj Finance

## Data Sources

The data is sourced from public financial APIs and stock market databases. For complete and up-to-date data, consider connecting to a real-time financial data API. 