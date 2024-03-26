function fetchNewsData(ticker) {
    fetch(`/news_data/${ticker}`)
        .then(response => response.json())
        .then(data => {
            // Process the news data here
            console.log(data); // Example: Log the data to the console
        })
        .catch(error => {
            console.error('Error fetching news data:', error);
        });
}