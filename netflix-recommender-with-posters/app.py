from flask import Flask, render_template, request
from recommender import recommend_movies

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    user_id = None

    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        recommendations = recommend_movies(user_id)

    return render_template('index.html', recommendations=recommendations, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)
