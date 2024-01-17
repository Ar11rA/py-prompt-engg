from flask import Flask, request

from services import location_service

app = Flask(__name__)


@app.get("/query")
def get_location():
    location = request.args.get('location')
    return location_service.get_location_summary(location)


if __name__ == "__main__":
    app.run(port=8081)
