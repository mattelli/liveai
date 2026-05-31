# Live Assistant

Simple example to create locally a live AI assistant.

Mic → Whisper (STT) → LM Studio (LLM) → Piper (TTS)


## Whisper
Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. 
See [Whisper](https://huggingface.co/openai/whisper-medium)

```
git lfs install
git clone https://huggingface.co/guillaumekln/faster-whisper-medium
```

## LM Studio SDK
Run AI models, locally and privately. See [LM Studio](https://lmstudio.ai/)
`lmstudio-python` provides a set APIs to interact with LLMs, embeddings models, and agentic flows.

### CLI 
To interact with the LM Studio Server use `lms -h` in the CLI.


## Piper
Piper is a local neural text-to-speech engine, see [Piper](https://github.com/OHF-Voice/piper1-gpl/tree/main)

```
python3 -m piper.download_voices en_US-hfc_female-medium
```


## Diffusers
Text to Image and Image to Image generation through Diffusion pipelins.

Considered models

- runwayml/stable-diffusion-v1-5
- UnfilteredAI/NSFW-gen-v2


## Setup (Python 3.12.10)

OS Windows 11 (powershell)

```
# Ai agent based on LM Studio API
.\lms-cpu-setup.ps1

# Stable-Diffusers pipeline
.\sd-cpu-setup.ps1  

```

## License

See [LICENSE](LICENSE) file.