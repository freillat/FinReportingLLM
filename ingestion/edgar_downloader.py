import os
import argparse
from sec_edgar_downloader import Downloader

def download_10k_filings(ticker, num_years):
    """
    Downloads the 10-K filings for a given ticker for a specified number of years.
    """
    print(f"Setting up downloader...")
    # Initialize downloader with your company name and email
    dl = Downloader("YourCompanyName", "your.email@example.com")
    
    output_path = "data"
    os.makedirs(output_path, exist_ok=True)
    
    try:
        print(f"Starting download for ticker {ticker}...")
        dl.get("10-K", ticker, limit=num_years, download_details=True)
        print(f"Successfully downloaded 10-K filings for {ticker} to '{output_path}/sec-edgar-filings'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 10-K filings from SEC EDGAR.")
    parser.add_argument("--ticker", type=str, default="MSTR", help="Stock ticker symbol (e.g., MSTR for MicroStrategy).")
    parser.add_argument("--years", type=int, default=5, help="Number of recent years to download filings for.")
    
    args = parser.parse_args()
    
    download_10k_filings(args.ticker, args.years)