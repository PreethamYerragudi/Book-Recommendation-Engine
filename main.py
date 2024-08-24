import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, url_for, jsonify

app = Flask(__name__)
user_ratings = [];
id = 1;

@app.route("/")
def init_page():
    updateColabRecs()
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
        
        .suggestions-bok:hover{
            cursor: text;
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
        <button id="myBtn" onclick="searchBooks()" style="display: none;">Search</button>
        <div id="suggestions" class="suggestions-box"></div>
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
            { """

    for i in range(len(titles2)):
        str += f'title: "{titles2[i]}", img: "{images2[i]}"'
        str += "},\n"
        if (i != len(titles2) - 1):
            str += '{'

    str += """    ];
        const books = [
            """

    for i in range(len(books["title"])):
        if not i == len(books["title"]) - 1:
            str += f'{{ title: \"{books["title"][i]}\", img: \"{books["image_url"][i]}\" }}, '
        else:
            str += f'{{ title: \"{books["title"][i]}\", img: \"{books["image_url"][i]}\" }}'
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
            const searchResults = books.filter(book => book.title.toLowerCase().includes(query));
            displayBooks(searchResults, 'search-results');
            document.getElementById('suggestions').innerHTML = '';
            console.log(searchResults.length);
            if (searchResults.length === 1) {
                window.location.href = `/rate/${searchResults[0].title}`;
            }
        }

        function showSuggestions() {
            const query = document.getElementById('search-bar').value.toLowerCase();
            const suggestionsBox = document.getElementById('suggestions');
            suggestionsBox.innerHTML = '';

            if (query) {
                const filteredBooks = books.filter(book => book.title.toLowerCase().includes(query));
                filteredBooks.forEach(book => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.className = 'suggestion-item';
                    suggestionItem.textContent = book.title;
                    suggestionItem.onclick = () => {
                        document.getElementById('search-bar').value = book.title;
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
    return f"{str}"


@app.route('/rate/<book_name>')
def rate_book(book_name):
    str = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Book Detail</title>
    <style>
        body {
    font - family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .book-detail {
    text - align: center;
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        .book-detail img {
    width: 200px;
            height: auto;
            border-radius: 5px;
        }

        .book-title {
    margin - top: 10px;
            font-size: 24px;
        }

        .rating {
    margin - top: 20px;
        }

        .rating span {
    font - size: 30px;
            cursor: pointer;
        }

        .rating span:hover,
        .rating span.selected {
    color: gold;
        }
    </style>
</head>
<body>
    <div class="book-detail">
        <img id="book-image" src="" alt="Book Image">
        <div id="book-title" class="book-title"></div>
        <div class="rating">
            <span onclick="rateBook(1)">★</span>
            <span onclick="rateBook(2)">★</span>
            <span onclick="rateBook(3)">★</span>
            <span onclick="rateBook(4)">★</span>
            <span onclick="rateBook(5)">★</span>
        </div>
    </div>

    <script>

        document.getElementById('book-title').textContent = """
    str += f"\"{book_name}\";"
    str += """
        document.getElementById('book-image').src = """
    str += f"\"{books['image_url'][books['title'].to_list().index(book_name)]}\";"
    str += """
    function rateBook(rating) {
        const bookName = """ + f"\"{book_name}\";" + """
        const bookId = """ + f"{books['title'].to_list().index(book_name)};" + """ // Get the book's ID based on its title

        // AJAX call to send the rating to the server
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/submit_rating", true);
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                window.location.href = '/';
            }
        };

        const data = JSON.stringify({
            book_id: bookId,
            rating: rating
        });
        xhr.send(data);
    }
    </script>
</body>
</html>

    """
    return str;


@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    global id
    data = request.get_json()
    book_id = data['book_id']
    rating = data['rating']

    user_ratings.append({'book_id': book_id, 'rating': rating})
    id = book_id
    return jsonify({"message": "Rating submitted successfully!"})


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
recommended_books = get_content_recommendations(book_id=id)
titles = recommended_books["title"].to_list()
images = recommended_books["image_url"].to_list()
recommended_books2 = get_knn_recommendations(user_ratings, knn, user_item_matrix)
titles2 = recommended_books2["title"].to_list()
images2 = recommended_books2["image_url"].to_list()



def updateColabRecs():
    global titles2, images2, titles, images
    recommended_books2 = get_knn_recommendations(user_ratings, knn, user_item_matrix)
    titles2 = recommended_books2["title"].to_list()
    images2 = recommended_books2["image_url"].to_list()
    recommended_books = get_content_recommendations(book_id=id)
    titles = recommended_books["title"].to_list()
    images = recommended_books["image_url"].to_list()
