# Twitter API Crawler

A Python-based Twitter crawler using twitterapi.io to collect tweets based on keywords and date ranges.

## Features

- Crawl tweets for specific keywords across multiple dates
- Two modes: Test mode and Formal crawl mode
- Automatic state management for resuming interrupted crawls
- Progress tracking with tqdm
- Comprehensive logging
- Configurable via external config file

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example config file
cp config.example.py config.py

# Edit config.py and add your API key
# Change: API_KEY = "YOUR_API_KEY_HERE"
# To:     API_KEY = "your_actual_api_key"
```

### 3. Customize Configuration (Optional)

Edit `config.py` to customize:
- Keywords to search for
- Date range (START_DATE, END_DATE, TARGET_DAYS)
- Number of tweets per date
- File paths for output
- Language filter
- Request delay

## Usage

### Test Mode

Test the crawler with 1 API call for both start and end dates:

```bash
python tweet_crawler.py test
```

This will:
- Test dates 2024-03-01 and 2025-03-20
- Make 1 API call per date
- Save results to `test.jsonl` (JSON Lines, one tweet per line)
- Validate your API key and query strategy

### Formal Crawl Mode

Run the full crawler:

```bash
python tweet_crawler.py crawl
```

This will:
- Crawl all dates in reverse order (from END_DATE to START_DATE)
- Collect tweets for TARGET_DAYS of each month (default: 1, 10, 20)
- Save each date to a separate JSON file in `tweet_data/`
- Track progress in `crawler_state.json`
- Log all activities to `tweet_crawler.log`

## Resuming Interrupted Crawls

If the crawler is interrupted, simply run the crawl command again:

```bash
python tweet_crawler.py crawl
```

The crawler will:
- Read the state file (`crawler_state.json`)
- Skip completed dates
- Resume from where it left off

## Output Files

- `test.jsonl` - Test mode results (JSON Lines)
- `tweet_data/tweets_YYYY-MM-DD.jsonl` - Formal crawl results (JSON Lines, one tweet per line)
- `crawler_state.json` - Progress tracking state
- `tweet_crawler.log` - Detailed logs

## Configuration Options

See `config.example.py` for all available configuration options:

- **API Configuration**: URL and API key
- **Crawling Configuration**: Tweets per date, pages, calls
- **File Paths**: Output directories and filenames
- **Keywords**: List of search terms
- **Date Range**: Start/end dates and target days
- **Search Configuration**: Language and query type
- **Rate Limiting**: Delay between requests

## Project Structure

```
.
├── tweet_crawler.py      # Main crawler script
├── config.py             # Your configuration (not in git)
├── config.example.py     # Configuration template
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── crawler_state.json   # State tracking (generated)
├── tweet_crawler.log    # Logs (generated)
├── test.jsonl           # Test results (generated)
└── tweet_data/          # Crawled data (generated)
    └── tweets_*.jsonl
```

## Notes

- The `config.py` file is excluded from git to protect your API key
- Always run test mode first to validate your setup
- The crawler includes automatic delays to avoid rate limiting
- All progress is saved, so you can safely interrupt and resume
