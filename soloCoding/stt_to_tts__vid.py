from moviepy.editor import VideoFileClip
import os
from gtts import gTTS # gtts 라이브러리의 gTTS 클래스 import
from playsound import playsound # playsound 모듈의 playsound 함수 import
import speech_recognition as sr # SpeechRecognition 라이브러리 import하고 sr이라는 별칭으로 사용
from pydub import AudioSegment

folder_link = "C:\minimalize_source\python_files\mp4"

file_list = os.listdir(folder_link) # 디렉토리의 모든 파일을 리스트에 담아서 리턴

mp4_file_list = [file for file in file_list if file.endswith("mp4")] # 맨 끝에 mp4로 끝나는(확장자가 mp4인) 파일들만 리스트에 담는다
#print(mp4_file_list) # debug

r = sr.Recognizer() # Recognizer 객체 생성

for file in mp4_file_list:
    mp3_link = f"{folder_link}\{file.split('.')[0]}.mp3"
    wav_link = f"{folder_link}\{file.split('.')[0]}.wav"
    #print(mp3_link) # debug

    if not os.path.exists(mp3_link):
        clip = VideoFileClip(f"{folder_link}\{file}")
        clip.audio.write_audiofile(mp3_link)

    if not os.path.exists(wav_link):
        AudioSegment.from_mp3(mp3_link).export(wav_link, format="wav")

    AudioSegment.ffmpeg = ""
    with sr.AudioFile(wav_link) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio, language='ko') # 오디오 데이터를 텍스트로 변환
        print(text) # 결과 출력

        speech = "movie_text.mp3" # 오디오를 저장할 파일 이름
        tts = gTTS(text, lang="ko") # 텍스트를 오디오로 변환
        tts.save(speech) # 파일로 저장
        playsound(speech) # 오디오 파일 재생

    except sr.UnknownValueError: # 음성 인식이 실패한 경우
        print("인식 실패")
        
    except sr.RequestError as e: # 요청이 실패한 경우
        print(f"요청 실패 : {e}")