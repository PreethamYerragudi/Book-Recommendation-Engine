import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, url_for

app = Flask(__name__)


@app.route("/")
def hello_world():
    redirect = False
    book = ""
    str = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Book Recommendation Page</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }

        .search-container {
            text-align: center;
            padding: 20px;
            background-color: tomato;
            position: relative;
        }

        #search-bar {
            width: 50%;
            padding: 10px;
            border-radius: 5px;
            border: none;
            font-size: 16px;
        }

        .suggestions-box {
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 50%;
            background-color: #fff;
            border: 1px solid #ccc;
            border-top: none;
            max-height: 150px;
            overflow-y: auto;
            z-index: 1000;
        }

        .suggestion-item {
            padding: 10px;
            cursor: pointer;
        }

        .suggestion-item:hover {
            background-color: #ddd;
        }

        .recommendation-section {
            padding: 20px;
        }

        .book-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }

        .book {
            width: 150px;
            text-align: center;
            background-color: #fff;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        .book img {
            width: 100%;
            height: auto;
            border-radius: 5px;
        }

        .book-title {
            margin-top: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="search-container">
        <input type="text" id="search-bar" placeholder="Search for books..." oninput="showSuggestions()">
        <button id="myBtn" onclick="searchBooks()">Search</button>

    </div>
    <div class="recommendation-section">
        <h2>Books Similar to Your Favorite Book</h2>
        <div id="similar-books" class="book-row"></div>
    </div>
    <div class="recommendation-section">
        <h2>Books You Might Like</h2>
        <div id="books-you-might-like" class="book-row"></div>
    </div>
    <div class="recommendation-section">
        <h2>Search Results</h2>
        <div id="search-results" class="book-row"></div>
    </div>

    <script>
        const similarBooks = [
        {
"""
    for i in range(len(titles)):
        str += f'title: "{titles[i]}", img: "{images[i]}"'
        str += "},\n"
        if (i != len(titles) - 1):
            str += '{'
    str += """];

        const booksYouMightLike = [
            { title: "Book A", img: "https://via.placeholder.com/150" },
            { title: "Book B", img: "https://via.placeholder.com/150" },
            { title: "Book C", img: "https://via.placeholder.com/150" },
        ];
        const books = [
            """

    for i in range(len(books["title"])):
        if not i == len(books["title"]) - 1:
            str += f'\"{books["title"][i]}\", '
        else:
            str += f'\"{books["title"][i]}\"'
    str += "];"

    str += """

        function displayBooks(books, containerId) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            books.forEach(book => {
                const bookDiv = document.createElement('div');
                bookDiv.className = 'book';
                bookDiv.innerHTML = `<img src="${book.img}" alt="${book.title}">
                                     <div class="book-title">${book.title}</div>`;
                container.appendChild(bookDiv);
            });
        }

        function searchBooks() {
            const query = document.getElementById('search-bar').value.toLowerCase();
            const searchResults = similarBooks.concat(booksYouMightLike).filter(book => book.title.toLowerCase().includes(query));
            displayBooks(searchResults, 'search-results');
            document.getElementById('suggestions').innerHTML = '';
        }

        function showSuggestions() {
            const query = document.getElementById('search-bar').value.toLowerCase();
            const suggestionsBox = document.getElementById('suggestions');
            suggestionsBox.innerHTML = '';

            if (query) {
                const filteredBooks = books.filter(book => book.toLowerCase().includes(query));
                filteredBooks.forEach(book => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.className = 'suggestion-item';
                    suggestionItem.textContent = book;
                    suggestionItem.onclick = () => {
                        document.getElementById('search-bar').value = book;
                        suggestionsBox.innerHTML = '';
                        searchBooks();
                    };
                    suggestionsBox.appendChild(suggestionItem);
                });
            }
        }

        // Display initial recommendations
        displayBooks(similarBooks, 'similar-books');
        displayBooks(booksYouMightLike, 'books-you-might-like');
        var input = document.getElementById("search-bar");
        input.addEventListener("keypress", function(event) {
          if (event.key === "Enter") {
            event.preventDefault();
            document.getElementById("myBtn").click();
          }
        });
    </script>
</body>
</html>
"""
    if redirect:
        return redirect(url_for('rate_book', book_name=book))
    else:
        return f"{str}"


@app.route('/rate/<book_name>')
def rate_book(book_name):
    return f"Rating {book_name}"


# Load the data
books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
book_tags = pd.read_csv('book_tags.csv')
tags = pd.read_csv('tags.csv')

# Merge book_tags and tags
book_tags = pd.merge(book_tags, tags, on='tag_id', how='inner')

# Create a combined tags string for each book
book_tags['tag_name'] = book_tags['tag_name'].str.replace('-', '')
book_tags_agg = book_tags.groupby('book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()

# Merge with the books dataframe
books = pd.merge(books, book_tags_agg, on='book_id', how='left')

# Fill NaN values in the tag_name column
books['tag_name'] = books['tag_name'].fillna('')

# Combine book title, authors, and tags for content-based filtering
books['content'] = books['title'] + ' ' + books['authors'] + ' ' + books['tag_name']

# TF-IDF Vectorizer for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['content'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to get book recommendations based on content similarity
def get_content_recommendations(book_id, cosine_sim=cosine_sim):
    idx = books.index[books['book_id'] == book_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the top 10 similar books
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices]


# Create a user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

# Fit the KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix_sparse)


# Function to get KNN-based recommendations
def get_knn_recommendations(user_ratings, knn_model, user_item_matrix, n_neighbors=10):
    user_ratings_df = pd.DataFrame(user_ratings)
    user_vector = np.zeros((1, user_item_matrix.shape[1]))
    for rating in user_ratings:
        user_vector[0, rating['book_id'] - 1] = rating['rating']

    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=n_neighbors)
    similar_users = indices.flatten()
    similar_users_ratings = user_item_matrix.iloc[similar_users].mean(axis=0)

    rec_books = similar_users_ratings.sort_values(ascending=False).head(10)
    rec_books = rec_books[rec_books.index.isin(
        user_item_matrix.columns[~user_item_matrix.columns.isin([rating['book_id'] for rating in user_ratings])])]

    return books[books['book_id'].isin(rec_books.index)]


# Example usage of content-based recommendations
recommended_books = get_content_recommendations(book_id=744)
titles = recommended_books["title"].to_list()
images = recommended_books["image_url"].to_list()

