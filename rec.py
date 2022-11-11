import sounddevice as sd
from scipy.io.wavfile import write

def record(duration):
    fs = 44100  # this is the frequency sampling; also: 4999, 64000
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    print("Starting: Speak now!")
    sd.wait()  # Wait until recording is finished
    print("recording finished")
    write('output.wav', fs, myrecording)  # Save as MP3 file
    
record(10) # 10 seconds