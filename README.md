# Face detector

A simple Streamlit app that detects faces in an uploaded photo using OpenCV's
pre-trained Haar cascade detector. Detected faces are outlined and labeled;
the result can be downloaded as a PNG.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, then upload a JPG, PNG, or WebP photo.
