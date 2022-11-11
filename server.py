from flask import Flask, render_template
app = Flask(__name__)

# two decorators, same function
@app.route('/')
def home():
    return render_template('home.html', the_title='Tiger Home Page')

@app.route('/recorder')
def record():
    return render_template('record.html', the_title='Tiger Home Page')

if __name__ == '__main__':
    app.run(debug=True)
