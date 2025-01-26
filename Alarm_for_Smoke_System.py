from gtts import gTTS
from playsound import playsound

def trigger_alarm(volume=50):
    alarm_message = "Warning: Smoke is Detected. Please evacuate immediately"
    tts = gTTS(text=alarm_message, lang='en')
    tts.save("alarm_smoke.mp3")
    playsound("alarm_smoke.mp3")

# now we will call the function
trigger_alarm(volume=50)
