# OPAC — Objective-driven Planning and Action Coordinator

> Fully local AI agent running on the Intel NPU.  
> No cloud. No API keys. No data leaves your device.

---

## What is implemented (Phase 1 & 2)

| Feature | Status |
|---|---|
| LLM inference on Intel NPU via OpenVINO GenAI | ✅ Phase 1 |
| INT4 quantised model (Phi-3 Mini / Qwen3-4B) | ✅ Phase 1 |
| Auto chunking for long documents | ✅ Phase 1 |
| PDF summarisation (PyMuPDF + pypdf fallback) | ✅ Phase 2 |
| Word .docx summarisation | ✅ Phase 2 |
| PowerPoint .pptx summarisation | ✅ Phase 2 |
| Excel .xlsx summarisation | ✅ Phase 2 |
| Web page / URL summarisation | ✅ Phase 2 |
| HTML file reading | ✅ Phase 2 |
| Plain text / Markdown reading | ✅ Phase 2 |
| Interactive text REPL | ✅ Phase 1 |
| Cross-platform: Windows + Linux | ✅ Both phases |
| Voice input / output | 🔜 Phase 3 |
| App launcher | 🔜 Phase 5 |

---

## Hardware requirements

- Intel Core Ultra 5/7/9 with **Intel AI Boost NPU**  
- 16 GB RAM (recommended model uses ~3 GB)  
- Windows 11 or Linux (Ubuntu 22.04+, Arch, Fedora)  
- Intel NPU driver installed (see setup below)

---

## Quick start

### 1. Install dependencies

```bash
python install.py
```

This installs all Python packages and checks your NPU driver.

### 2. Download and compile the model

```bash
python opac.py --setup
```

Downloads `Phi-3-mini-int4` from Hugging Face (~2.5 GB) and compiles it to NPU format.  
**First compile takes ~60 seconds** — result is cached, subsequent starts take ~5 s.

### 3. Run OPAC

```bash
python opac.py                         # interactive mode
python opac.py --file report.pdf       # summarise a PDF directly
python opac.py --file slides.pptx      # summarise a PowerPoint
python opac.py --url https://...       # summarise a web page
python opac.py --info                  # check NPU and system status
```

### 4. Run tests

```bash
python tests/test_phase1_2.py
```

Phase 2 (document) tests run on any machine.  
Phase 1 (NPU) tests run only when OpenVINO and the NPU driver are installed.

---

## NPU driver installation

### Windows 11

1. Go to: https://www.intel.com/content/www/us/en/download/794734/
2. Download and run **Intel NPU Driver for Windows**
3. Reboot
4. Verify: open Task Manager → Performance → you should see "NPU" listed

### Linux (Ubuntu 22.04 / 24.04)

```bash
# Download the three .deb packages from:
# https://github.com/intel/linux-npu-driver/releases

sudo dpkg -i intel-driver-compiler-npu_*.deb
sudo dpkg -i intel-fw-npu_*.deb
sudo dpkg -i intel-level-zero-npu_*.deb
sudo reboot

# Verify
python opac.py --info
```

### Linux (Arch / Manjaro)

```bash
yay -S intel-npu-driver-bin     # AUR package
sudo reboot
```

---

## Changing the model

Edit `config/settings.py`:

```python
# Fastest (recommended for 16 GB RAM):
DEFAULT_MODEL_REPO = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"   # 2.5 GB, ~12 tok/s

# Slightly smarter:
DEFAULT_MODEL_REPO = "OpenVINO/Qwen3-4B-int4-cw-ov"              # 3.0 GB, ~10 tok/s

# Best quality (needs ~6 GB RAM free):
DEFAULT_MODEL_REPO = "OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov"
```

Then re-run `python opac.py --setup` to download and compile the new model.

---

## Changing the inference device

In `config/settings.py`:

```python
INFERENCE_DEVICE = "NPU"   # Intel AI Boost — recommended (frees CPU/GPU)
INFERENCE_DEVICE = "GPU"   # Intel Arc — faster tokens, uses GPU
INFERENCE_DEVICE = "CPU"   # Fallback — works everywhere, slow
```

---

## Project structure

```
opac/
├── opac.py                  ← main entry point
├── install.py               ← one-command installer
├── requirements.txt
├── config/
│   └── settings.py          ← all configuration in one place
├── core/
│   ├── npu_engine.py        ← Phase 1: OpenVINO GenAI on NPU
│   ├── model_setup.py       ← Phase 1: model download + verification
│   ├── summarizer.py        ← Phase 1+2: chunking + LLM summarisation
│   └── agent.py             ← orchestrator + interactive REPL
├── documents/
│   └── loader.py            ← Phase 2: PDF, DOCX, PPTX, XLSX, HTML, web
├── utils/
│   ├── chunker.py           ← smart text chunking
│   ├── logger.py            ← centralised logging
│   └── platform_info.py     ← NPU/system status
└── tests/
    └── test_phase1_2.py     ← full test suite
```

---

## Example session

```
  You: /home/user/documents/quarterly_report.pdf

  [OPAC] Read: [PDF] Q3 Financial Report — 8420 words, 12 page(s)  (0.4s)
  [OPAC] Long document — summarising in 3 sections …
  [OPAC] Processing section 1/3 … done (52 words)
  [OPAC] Processing section 2/3 … done (48 words)
  [OPAC] Processing section 3/3 … done (61 words)
  [OPAC] Combining section summaries …

  OPAC: The Q3 report shows revenue of $2.4M, up 18% year-on-year.
  Operating costs increased by 7% due to headcount expansion. The
  company expects Q4 revenue of $2.8M driven by new product launches
  in the APAC region. Key risks identified include supply chain delays
  and currency fluctuation.


  You: what were the key risks again?

  OPAC: The report identified two key risks: supply chain delays
  affecting product delivery timelines, and currency fluctuation
  impacting APAC revenue projections.
```

---

## Coming next

- **Phase 3** — Voice interface (say "Hey OPAC", hear responses)
- **Phase 4** — Live browser summarisation (current tab)
- **Phase 5** — App launcher ("open VS Code")
- **Phase 6** — Startup service + system tray
