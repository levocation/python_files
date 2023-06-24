import speech_recognition as sr
from playsound import playsound

r = sr.Recognizer() # Recognizer 객체 생성
with sr.Microphone() as source: # 마이크를 source로 사용
    print("녹음 시작")
    audio = r.record(source, duration=5) # 5초 동안 마이크를 사용하여 녹음
    print("녹음 끝")

file_name = "voice.wav" # 저장할 오디오 파일의 이름
with open(file_name, "wb") as f:
    f.write(audio.get_wav_data()) # 녹음한 데이터를 저장

playsound(file_name) # 오디오 파일 재생