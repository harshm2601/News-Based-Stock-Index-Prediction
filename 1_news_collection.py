from pynytimes import NYTAPI
import datetime
import pandas as pd
import numpy as np

def get_news(year, month, day):
    """
    get top 10 most relevant finance news headings on each day from NY Times
    """
    try:
        # Initialize the NYTAPI object
        # nyt = NYTAPI("FMWU5sNkUErTLUf8zjHM2hWr9MkWbmlT", parse_dates=True)
        nyt = NYTAPI("zWyggCXFeOgLZgZj5evSUfxj8eYWb6mE", parse_dates=True)
        list = []
        
        print(f"Fetching news for {year}-{month}-{day}...")

        # Fetch articles from the API
        articles = nyt.article_search(
            results=10,
            dates={
                "begin": datetime.datetime(year, month, day),
                "end": datetime.datetime(year, month, day)
            },
            options={
                "sort": "relevance",
                "news_desk": [
                    "Business", "Business Day", "Entrepreneurs", "Financial", "Technology"
                ],
                "section_name": [
                    "Business", "Business Day", "Technology"
                ]
            }
        )

        # Check if articles are fetched successfully
        if not articles:
            print(f"No articles found for {year}-{month}-{day}.")
        
        for i in range(len(articles)):
            # Extract the abstract and remove commas
            abstract = articles[i].get('abstract', 'No Abstract Available')
            list.append(abstract.replace(',', ""))
        
        return list
    except Exception as e:
        print(f"Error occurred while fetching news for {year}-{month}-{day}: {e}")
        return []

df = pd.DataFrame()

def generate_news_file():
    """
    Store news headings for each day in a given date range and save to a CSV file
    """
    try:
        start = '2024-03-27'
        end = '2025-03-27'
        
        print(f"Generating news file from {start} to {end}...")

        # Generate a list of dates
        mydates = pd.date_range(start, end)
        dates = [mydates[i].strftime("%Y-%m-%d") for i in range(len(mydates))]

        # Initialize a matrix to store the data
        matrix = np.zeros((len(dates) + 1, 11), dtype=object)
        matrix[0, 0] = "Date"

        # Set column headers for the news
        for i in range(10):
            matrix[0, i + 1] = f"News {i + 1}"

        # Loop through each date and fetch news
        for i in range(len(dates)):
            matrix[i + 1, 0] = dates[i]
            y, m, d = dates[i].split("-")
            print(f"Fetching news for {dates[i]}...")
            news_list = get_news(int(y), int(m), int(d))
            
            if news_list:
                for j in range(len(news_list)):
                    matrix[i + 1, j + 1] = news_list[j]
            else:
                print(f"No news found for {dates[i]}. Filling with empty values.")
                for j in range(10):
                    matrix[i + 1, j + 1] = "No News"
        
        # Convert matrix to DataFrame
        df = pd.DataFrame(matrix)

        # Save to CSV
        df.to_csv("news1.csv", index=False)
        print("News file generated and saved as news1.csv")
    except Exception as e:
        print(f"Error occurred while generating the news file: {e}")

# Start the news file generation
generate_news_file()