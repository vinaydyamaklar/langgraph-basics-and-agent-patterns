import os
import json
import pyttsx3
import speech_recognition as sr
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI

from  schemas import HotelBooking


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# speak function to convert text to speech
def speak(text: str):
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 160)
    tts_engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
    tts_engine.say(text)
    tts_engine.runAndWait()


# speech-to-text
def listen():
    # initialize text-to-speech, and speech Recognition
    r = sr.Recognizer()

    with sr.Microphone() as mic:
        print("Listening...")
        r.adjust_for_ambient_noise(mic, duration=1)
        audio = r.listen(mic)

        try:
            user_input = r.recognize_google(audio)
            print(f"User: {user_input}")
            return user_input
        except sr.UnknownValueError:
            speak("I couldn't understand that.")
            return None
        except sr.RequestError:
            speak("Could not request the result.")
            return None


def book_hotel_room(booking_data: HotelBooking):
    """Simulates booking a hotel room with extracted data, CONFIRMATION"""
    speak("I have the following data:")
    speak(f"Name: {booking_data.full_name}")
    speak(f"Check-In date: {booking_data.checkin_data}")
    speak(f"Check-out date: {booking_data.checkout_data}")
    speak(f"Rooms: {booking_data.number_of_rooms}")
    speak(f"Number of Guests: {booking_data.number_of_guests}")

    if booking_data.special_request:
        speak(f"I also have your special request which is, {booking_data.special_request}")

    speak("Do you confirm the details and booking? sat yes or no!")

    confirmation = listen()
    if "yes" in confirmation.lower():
        speak("Great!, your booking is confirmed! Enjoy your stay")
        return True
    else:
        speak("Okay, Let me start again")
        return False


def get_booking_details():
    """Main conversational loop to get booking details."""
    speak("Hello, I am your hotel reservation assistant!")

    messages = [
        {
            "role": "system",
            "content": """You are a polite hotel reservation assistant! 
                        your primary goal is to ask the user the hotel room booking details in a conversational way
                        You will ask one detail at a time don't ask everything once  
                        and you will collect all the required booking details as below:
                        - Name 
                        - Checkin date 
                        - Checkout date
                        - Number of rooms
                        - Number of Guests
                        - Any special request 
                        """
        },
        {
            "role": "user",
            "content": f"I want to book a room in your hotel!"
        }
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "book_hotel_room",
            "description": "Books a hotel room with the provided data.",
            "parameters": HotelBooking.model_json_schema()
        }
    }]

    while True:
        try:
            # Let's start calling open ai with messages(prompts) and tools(resource for open ai)
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name

                    if function_name == 'book_hotel_room':
                        function_args = json.loads(tool_call.function.arguments)

                        try:
                            booking_details = HotelBooking(**function_args)
                            if book_hotel_room(booking_details):
                                return booking_details
                            else:
                                # if user rejects the booking restart the conversation
                                messages.clear()
                                messages.append({
                                    "role": "system",
                                    "content": """You are a polite hotel reservation assistant! 
                                                    your primary goal is to ask the user the hotel room booking details in a conversational way  
                                                    and you will collect all the required booking details as below:
                                                    - Name 
                                                    - Checkin date 
                                                    - Checkout date
                                                    - Number of rooms
                                                    - Number of Guests
                                                    - Any special request 
                                                    """
                                })

                                speak("Let's start over again! What's your name?")
                                user_name = listen()
                                if user_name:
                                    messages.append({
                                        "role": "user",
                                        "content": f"My name is {user_name} and I want to book a room!"
                                    })
                                else:
                                    speak("Okay good bye!")
                                    return None

                        except Exception as e:
                            speak(f"I am missing some information can you provide your name, check-in date, check-out date, number of rooms, number of guests, any special request!")
                            messages.append({
                                "role": "assistant",
                                "content": "I'm missing some required information. Can you provide your full name, check-in and check-out dates, number of rooms, and number of guests?"
                            })
                            user_response = listen()
                            if user_response:
                                messages.append({
                                    "role": "user",
                                    "content": user_response
                                })
                            else:
                                return None
            else:
                # The model responded conversationally, not with a function call
                assistant_response = response_message.content
                print(f"assistant: {assistant_response}")
                speak(assistant_response)

                user_response = listen()
                if user_response:
                    messages.append({"role": "user", "content": user_response})
                else:
                    speak("I am unable to continue. Goodbye.")
                    return None

        except Exception as e:
            speak(f"An error occurred: {e}. Please try again.")
            return None


if __name__ == "__main__":
    booking_data = get_booking_details()