# Hawkish vs Dovish Sentiment Analysis on FOMC Data

Get Data (1996 - 10/15/2022 period)
1. Run create_master_files.py
2. Run the following (they all use respective master file except Press Confernce)
- Meeting Minutes (1996 - 2022)
  - Run mm_extract_data 
    - gets all raw data for mm
  - Run dictionary_filter
    - gets filtered mm sentences by key words
- Press Conference (Manually downloaded files, 2011-2022)
  - Run press_conf_extract_data
    - gets all raw data for press conf
      - all: returns all sentences from press conf
      - select: returns non-question statements made by Fed Chair
  - Run dictionary_filter
    - gets filtered press conf sentences by key words
- Speech (1996 - 2022)
  - Run speech_extract_data.py
    - gets all speech files using selenium driver
      - downloads the files as .htm if title contains a key words
      - convert .htm to text files
  - Run dictionary_filter
    - gets filtered speech sentences by key words

Key Words:
- financial terms that were outlined in Yuriy that we use to isolate relevant sentences
- If sentence contains any following word we consider them relevant

A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate", "price", "economic activity", "inflation",
      "employment"]

B1 = ["unemployment", "growth", "exchange rate", "productivity", "deficit", "demand", "job market", "monetary policy"]

Details:
  - Meeting Minutes:
  - Press Conference:
    - Total Lines: **19,068** 
    - Total Words: **468,941**
    - Filtered Lines: **5,086**
    - Filtered Words: **160,574**
  - Speech:
    - Total Lines: **23,503** 
    - Total Words: **735,222**
    - Filtered Lines: **12,428**
    - Filtered Words: **446,873**