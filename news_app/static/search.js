function fetchNewsData(ticker) {
    fetch(`/news_data/${ticker}`)
        .then(response => response.json())
        .then(data => {
            // Process the news data here
            console.log(data); // Example: Log the data to the console

            // Display the news data on the page
            const newsContainer = document.getElementById('news-container');
            newsContainer.innerHTML = ''; // Clear any existing news

            data.forEach(article => {
                const articleElement = document.createElement('div');
                articleElement.classList.add('article');

                const titleElement = document.createElement('h3');
                titleElement.textContent = article.title;

                const linkElement = document.createElement('a');
                linkElement.href = article.link;
                linkElement.textContent = 'Read more';

                const timestampElement = document.createElement('p');
                timestampElement.textContent = article.timestamp;

                articleElement.appendChild(titleElement);
                articleElement.appendChild(linkElement);
                articleElement.appendChild(timestampElement);

                newsContainer.appendChild(articleElement);
            });
        })
        .catch(error => {
            console.error('Error fetching news data:', error);
        });
}

// Assuming you have a form with an input field for the ticker symbol and a submit button
const searchForm = document.getElementById('search-form');

searchForm.addEventListener('submit', (event) => {
    event.preventDefault(); // Prevent the form from submitting normally

    const ticker = document.getElementById('ticker-input').value;
    fetchNewsData(ticker);
});