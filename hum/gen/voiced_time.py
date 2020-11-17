from datetime import datetime as dt
import time

from hum.util import ModuleNotFoundErrorNiceMessage

with ModuleNotFoundErrorNiceMessage("You'll need to pip install pyttsx3 to use voiced_time"):
    import pyttsx3

DFLT_TIME_FORMAT = '%H %M %S'


class Voicer:
    def __init__(self, voice=None, volume=None, rate=None, engine_kwargs=None, time_format=DFLT_TIME_FORMAT,
                 **kwargs):
        engine_kwargs = engine_kwargs or {}
        self.engine = pyttsx3.init(**engine_kwargs)  # object creation

        if voice is None:
            for voice_dict in self.voices:
                if 'en_US' in voice_dict['languages']:
                    voice = voice_dict['id']
                    break
            else:
                print("Didn't find a voice!!")
        else:
            voice_id = self.voice_id_for_name(voice)
            if voice_id is not None:
                voice = voice_id

        for attr, val in dict(kwargs, volume=volume, rate=rate).items():
            if val is not None:
                self.engine.setProperty(attr, val)

        self.time_format = time_format

    def say(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()

    @property
    def voices_df(self):
        import pandas as pd
        df = pd.DataFrame(self.voices)
        df = df.set_index('name')
        df['language'] = [x[0] for x in df['languages']]
        df['gender'] = ['male' if x == 'VoiceGenderMale' else 'female' for x in df['gender']]
        del df['languages']
        return df

    def __getattribute__(self, attr):
        if attr in {'volume', 'rate', 'voice'}:
            return self.engine.getProperty(attr)
        else:
            return super().__getattribute__(attr)

    @property
    def voices(self):
        return [x.__dict__ for x in self.engine.getProperty('voices')]

    def voice_id_for_name(self, name):
        for voice in self.voices:
            if voice['name'] == name:
                return voice['id']

    def say_the_time(self, verbose=False):
        s = dt.utcnow().strftime(self.time_format)
        if verbose:
            print(s)
        self.say(s)

    def tell_time_continuously(self, every_secs=5, verbose=False):
        try:
            while True:
                tic = time.time()
                self.say_the_time(verbose=verbose)
                elapsed = time.time() - tic
                time.sleep(max(0, every_secs - elapsed))
        except KeyboardInterrupt:
            print("KeyboardInterrupt!!!")
            self.engine.stop()


def tell_time_continuously(every_secs=3, voice=None, volume=1, rate=200, engine_kwargs=None,
                           time_format=DFLT_TIME_FORMAT, verbose=False):
    """
    Loop, and say the time regularly.

    Args:
        every_secs: How often to tell the time (in seconds)
        voice: What voice to use (or "help" to get a list of names (and their properties))
        volume: Volume of voice
        rate: How quickly the time is uttered
        engine_kwargs: Stuff you don't need to worry about
        time_format: In what format the time (or even date) is transformed to text that is read.
            See https://www.programiz.com/python-programming/datetime/strftime

    Returns:
        Nothing, just loops and says the time regularly
    """
    if voice == 'help':
        print(Voicer().voices_df)
    else:
        voicer = Voicer(voice=voice, volume=volume, rate=rate, engine_kwargs=engine_kwargs, time_format=time_format)
        voicer.tell_time_continuously(every_secs, verbose=verbose)


if __name__ == '__main__':
    import argh

    argh.dispatch_command(tell_time_continuously)
