import openai 

openai.api_key = "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"

def chat_info(class_name):
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

        # print(response)
        # engine = pyttsx3.init()
        # engine.say(response)
        # engine.runAndWait()
        # engine.stop()

        return response

print(chat_info("ultrasonic sensor"))