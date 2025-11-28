from flask import Flask

app=Flask(__name__)

@app.route("/")
def home():
    return "hello,flask"

@app.route("/about")
def about():
    return "about flask"

@app.route("/triangle")
def triangle():
    n=5
    pattern=""
    for i in range(n):
        for j in range(i):
            pattern+="*"
        pattern+="\n"
    return "<pre>" + pattern + "</pre>"


app.run()