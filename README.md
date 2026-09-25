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
Text to Image and Image to Image generation through Diffusion pipelines.

Considered models

- runwayml/stable-diffusion-v1-5
- UnfilteredAI/NSFW-gen-v2


## Setup (Python 3.12.10)

OS Windows 11 (powershell)

```
# Ai agent based on LM Studio API
.\setup-lms-cpu.ps1

# Stable-Diffusers pipeline
.\setup-sd-cpu.ps1  

```


## System Notes

```
CPU - AMD Ryzen 7 7735HS with Radeon Graphics - 8 Cores
RAM - 24 GB

APU - AMD Radeon(TM) Graphics - Primary/Integrated
VRAM - 4096 MB - LPDDR5 800 MHz

Windows Edition - Windows 11 Professional (64 bit)
Windows Version - 25H2
```

The processing is done through CPU.
GPU processing is supported only through Direct ML (very limited - not usable)


## License

See [LICENSE](LICENSE) file.