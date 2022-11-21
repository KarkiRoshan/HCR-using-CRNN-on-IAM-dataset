import model
import cv2
import resizer
import numpy as np
import tensorflow.keras.backend as K
import string

char_list = (
    "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
new_model = model.make_model()

new_model.load_weights("model.hdf5")


def listToString(s):

    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1


def word_predictor(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rescaled_image = resizer.process_image(img)
    added_dimension_image = np.expand_dims(rescaled_image, axis=2)
    # print(added_dimension_image.shape)
    final_image = added_dimension_image.reshape(1, 32, 128, 1)

    prediction = new_model.predict(final_image)
    decoded = K.get_value(
        K.ctc_decode(
            prediction,
            input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
            greedy=True,
        )[0][0]
    )
    out = K.get_value(decoded)
    predicted_array = []

    for x in out:
        print("predicted text = ", end="")
        for p in x:
            if int(p) != -1:
                predicted_array.append(char_list[int(p)])
    predicted_string = listToString(predicted_array)
    return predicted_string
