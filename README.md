# Fasal-AI 🌾

An AI-powered agricultural advisory platform built with CrewAI, OpenAI, and a React frontend.

## Overview

Fasal-AI helps Pakistani farmers with:
- **Disease Detection** — Upload a crop image to identify diseases and get treatment advice
- **Weather Advisories** — Real-time weather data with farming recommendations
- **Agronomy Guidance** — Crop management best practices
- **Export Compliance** — MRL (Maximum Residue Limit) checks against PSQCA standards
- **Multilingual Support** — Advisory delivery in Urdu, Punjabi, Sindhi, and English

## Project Structure

```
fasal-ai/
├── src/
│   ├── agents/             # CrewAI agents
│   │   ├── vision_agent.py
│   │   ├── weather_agent.py
│   │   ├── agronomy_agent.py
│   │   ├── export_compliance_agent.py
│   │   └── language_agent.py
│   ├── tools/              # Agent tools
│   │   ├── weather_tool.py
│   │   ├── plant_id_tool.py
│   │   └── mrl_checker_tool.py
│   └── data/               # Knowledge bases
│       ├── crop_knowledge.json
│       ├── mrl_database.json
│       └── psqca_standards.json
├── main.py                 # Entry point & key validation
├── app.py                  # Streamlit app (optional)
├── .env                    # API keys (do not commit)
├── .env.example            # Key template
└── requirements.txt
```

## Setup

1. **Clone the repo and navigate to the project folder**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   copy .env.example .env
   # Fill in your actual keys in .env
   ```

5. **Verify setup**
   ```bash
   python main.py
   # Output: Setup complete. All keys loaded.
   ```

## Required API Keys

| Key | Source |
|-----|--------|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `PLANT_ID_API_KEY` | [plant.id](https://plant.id) |

## Tech Stack

- **AI Agents**: CrewAI + OpenAI GPT-4o
- **Vision**: Plant.id API + GPT-4 Vision
- **Frontend**: React
- **Backend**: Python (FastAPI/Streamlit)
- **Vector Store**: ChromaDB
- **LLM Orchestration**: LangChain
