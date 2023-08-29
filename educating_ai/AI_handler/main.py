from keras.models import load_model
import cv2
import numpy as np
import openai 

import pyttsx3

# engine = pyttsx3.init()
# engine.say("I will speak this text")
# engine.runAndWait()
# engine.stop()

def allinone():

    openai.api_key = "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"

    def result(class_name):
        model_engine = "text-davinci-003"
        prompt="explain a student what is " + class_name + " ,what is it used for or any other info under 100 words"

        completion = openai.Completion.create(
            engine = model_engine,
            prompt = prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        response = completion.choices[0].text

        print(response)
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()
        engine.stop()

        return

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # CAMERA can be 0 or 1 based on the default camera of your computer
    camera = cv2.VideoCapture(0)

    capture_image = False  # Flag to indicate if an image should be captured

    while True:
        # Grab the web camera's image.
        ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # Check if the 'c' key is pressed to capture the image
        if keyboard_input & 0xFF == ord('c'):
            capture_image = True

        if capture_image:
            # Make the image a numpy array and reshape it to the model's input shape.
            image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            image_array = (image_array / 127.5) - 1

            # Predict the model
            prediction = model.predict(image_array)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Prepare text to display on the image
            text = f"Class: {class_name[2:]}, Confidence Score: {int(confidence_score * 100)}%"

            print(class_name,text)
            result(class_name=class_name)

            # Display text on the image
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            

            # Show the image with prediction
            cv2.imshow("Webcam Image", image)

            capture_image = False  # Reset the capture flag

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()