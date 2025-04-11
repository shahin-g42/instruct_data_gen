import stable_whisper

model = stable_whisper.load_model('turbo')
result = model.transcribe("./res/8fb4b111-429b-463f-a3f1-3e72c9b8ecf2.wav")
print(result.language)
result.to_srt_vtt("./res/8fb4b111-429b-463f-a3f1-3e72c9b8ecf2.vtt", vtt=True, word_level=False)


result = model.transcribe("./res/015fc24c-b17f-4ec7-bd35-4a4f25b1067c.wav")
result.to_srt_vtt("./res/015fc24c-b17f-4ec7-bd35-4a4f25b1067c.vtt", vtt=True, word_level=False)
