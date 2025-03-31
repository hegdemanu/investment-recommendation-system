# Uploads Directory

This directory is used to store CSV files uploaded by users through the web interface or API endpoints. The files contain historical stock data for analysis and model training.

## File Handling

- Files uploaded via `/api/upload-csv` endpoint are stored here
- Original filenames are preserved when possible, but sanitized for security
- Duplicate filenames are handled by adding timestamps
- CSV files are validated for proper format before being processed

## File Format Requirements

Uploaded CSV files must conform to the following format:

1. **Required columns**:
   - `Date`: Trading date in YYYY-MM-DD format
   - Ticker identifier (can be column name or provided in request)
   - Price data (at least one of: `Close`, `Price`, `Adj Close`)

2. **Recommended columns**:
   - `Open`: Opening price
   - `High`: Highest price during the period
   - `Low`: Lowest price during the period
   - `Volume`: Trading volume
   - `Change %`: Percentage price change

3. **Example CSV format**:

```
Date,Open,High,Low,Close,Adj Close,Volume
2023-01-01,100.5,105.2,99.8,104.3,104.3,1500000
2023-01-02,104.5,108.7,103.6,107.2,107.2,2000000
2023-01-03,107.0,109.1,105.3,106.8,106.8,1800000
```

## File Size Limits

- Maximum file size: 16MB
- Recommended row count: <100,000 rows per file
- For larger datasets, consider splitting into multiple files

## Data Security

- This directory should have appropriate permissions set
- Files may contain sensitive financial data
- No personally identifiable information should be included
- Access should be restricted to authorized system users only

## File Lifecycle

1. Files are uploaded to this directory
2. System validates file format and content
3. Data processor loads and preprocesses the data
4. Processed data is used for model training and analysis
5. Original files remain in this directory for reference
6. Files may be periodically archived or deleted based on retention policy

## Uploading Files Programmatically

Example using curl:
```bash
curl -X POST http://localhost:5000/api/upload-csv \
  -F "file=@/path/to/your/stock_data.csv"
```

Example using Python requests:
```python
import requests

files = {'file': open('stock_data.csv', 'rb')}
response = requests.post('http://localhost:5000/api/upload-csv', files=files)
```

## Troubleshooting

If you encounter issues with file uploads:
1. Check that the file is a valid CSV
2. Ensure required columns are present
3. Verify the file size is within limits
4. Check file permissions and disk space
5. Review server logs for specific error messages 