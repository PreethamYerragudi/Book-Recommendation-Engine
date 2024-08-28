# Book-Recommendation-Engine
**Overview**

This Book Recommendation System is a Flask-based web application that provides users with personalized book recommendations using both content-based filtering and collaborative filtering (KNN). Users can search for books, receive recommendations based on their favorite books, and rate books to improve future suggestions.
____________________________________________________________________________________________________________________
**Features**

Search Functionality: Users can search for books and view detailed information about them.

Content-Based Recommendations: Get recommendations for similar books based on the content of a selected book, including its title, authors, and associated tags.

Collaborative Filtering: Personalized recommendations based on user ratings and preferences using KNN (K-Nearest Neighbors).

Real-Time Suggestions: Dynamic search suggestions appear as the user types in the search bar.
____________________________________________________________________________________________________________________
**Tech Stack**

Backend: Flask, Python, Pandas, Scikit-learn

Frontend: HTML, CSS, JavaScript 

Machine Learning: TF-IDF Vectorizer, Cosine Similarity, KNN (K-Nearest Neighbors) 

Data: Book ratings, metadata, and tags from CSV files 
______________________________________________________________________________________________________________________
**Usage**

Search for Books: Use the search bar to find books by title.
Rate Books: Click on a book from the search results to rate it. Ratings help in improving future recommendations.
View Recommendations: Explore books similar to your favorite books and discover new reads based on your ratings.
