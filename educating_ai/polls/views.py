from django.shortcuts import render , redirect
from keras.models import load_model
import cv2
import numpy as np
import base64
import json
from django.views.decorators.csrf import csrf_exempt
# from AI_handler import chat
# Create your views here.
from django.http import HttpResponse
from django.http import JsonResponse


# Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
def cap_helper(request):
    x=""
    if request.method == 'POST':
        image_data = request.POST.get('image')
        x=capture_image(image_data)
        c=str(x.get("class_name"))
        # d=str(chat.chat_info(c))
        x['chatgpt_response'] = d
        return JsonResponse(x)
        return render(request, 'capture.html',context)
    context={'x':x}
    return render(request, 'capture.html',context)

def capture_image(image_data):
    # if request.method == 'POST':
        # image_data = request.POST.get('image')
        if image_data:
            # Decode the base64 image data
            _, img_encoded = image_data.split(',')
            img_decoded = base64.b64decode(img_encoded)

            # Save the image to a file
            image_path = 'captured_image.png'
            with open(image_path, 'wb') as f:
                f.write(img_decoded)

            # Load the model
            model = load_model("./AI_handler/keras_Model.h5", compile=False)

            # Load the labels
            class_names = open("./AI_handler/labels.txt", "r").readlines()

            # Read and preprocess the captured image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1

            # Predict the model
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            # print(class_name,confidence_score)


            # Prepare the prediction result
            result = {
                'class_name': class_name[2:],
                'confidence_score': int(confidence_score * 100),
            }
            # return JsonResponse(result)
            return result

            return JsonResponse(result)

        return JsonResponse({'error': 'Invalid request.'}, status=400)

def index(request):
    a="a"
    b=int(2)
    if request.method == "POST":

        result={}
        return redirect('cap',slug=result)
    context={'a':a,'b':b}
    return render(request, 'x.html',context)

def capture(request):
    context={}
    # context={'class_name':slug1,'confidense':slug2}
    return render(request, 'capture.html',{})