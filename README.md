# Live Assistant

Simple example to create locally a live AI assistant.

Mic → Whisper (STT) → LM Studio (LLM) → Piper (TTS)


## Setup

```
pip install numpy sounddevice soundfile faster-whisper openai piper-tts
```

## Whisper
Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. 
See [Whisper](https://huggingface.co/openai/whisper-medium)

```
git lfs install
git clone https://huggingface.co/guillaumekln/faster-whisper-medium
```

## LM Studio
Run AI models, locally and privately. 
See [LM Studio](https://lmstudio.ai/)


## Piper
Piper is a local neural text-to-speech engine, see [Piper](https://github.com/OHF-Voice/piper1-gpl/tree/main)

```
python3 -m piper.download_voices en_US-hfc_female-medium
```


## License

See [LICENSE](LICENSE) file.