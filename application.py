from flask import Flask, render_template, request
from NN_sache.image_class import *
from keras.applications import ResNet50

application = Flask(__name__)
model = None


def load_model():
    """
    Load Model Function:
    Pretrained ImageNet and provided by Keras,
    To be substituted with self model
    """

    global model
    model = ResNet50(weights="imagenet")


# Catch type errors when submitting something other than an image
@application.errorhandler(TypeError)
def internal_error(error):
    return render_template('500.html'), 500


@application.errorhandler(404)
def internal_error(error):
    return render_template('404.html'), 404


@application.route('/')
def index():
    return render_template('home.html')


@application.route('/human_error')
def musik():
    return render_template('human_error.html')


@application.route('/classify')
def classify():
    return render_template('classify.html')


@application.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    print(data)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):

            # read the image in PIL format
            image = request.files["image"].read()
            print(image)
            image = open_image(image)

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": round(float(prob), 2)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True
        else:
            raise TypeError
    # return the data dictionary as a JSON response
    return render_template('predictions.html', data=data["predictions"])


if __name__ == '__main__':
    load_model()
    application.run(debug=True)
