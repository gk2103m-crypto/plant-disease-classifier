import os
import sqlite3
import tempfile
import logging
from datetime import datetime
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

from inference import PlantDiseaseClassifier

# Configure application logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "predictions.db"
# IMPORTANT: Provide your actual Gemini API key here
GEMINI_API_KEY = "AIzaSyBjcMW5tMt6zemVZNCw_9gxxrxpLUWTM2I"

# Initialize ML Models and AI Clients
try:
    classifier = PlantDiseaseClassifier()
    genai.configure(api_key=GEMINI_API_KEY)
    generative_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise RuntimeError("Model initialization failed.")

app = FastAPI(title="Agricultural Diagnostic API")

# Pydantic schemas for request/response validation
class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    treatment_plan: str

class HistoryItem(BaseModel):
    disease: str
    confidence: float
    treatment_plan: str
    timestamp: str

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def initialize_database():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnostic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                treatment_plan TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()

initialize_database()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Chakra+Petch', sans-serif; }
        .glass-panel {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(75, 85, 99, 0.4);
            box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.03);
        }
        .futuristic-bg {
            background-color: #0d1117;
            background-image: 
                radial-gradient(#1f2937 1px, transparent 1px),
                radial-gradient(#1f2937 1px, transparent 1px);
            background-size: 40px 40px;
            background-position: 0 0, 20px 20px;
            opacity: 0.8;
        }
        .neon-text-green {
            color: #4ade80;
            text-shadow: 0 0 10px rgba(74, 222, 128, 0.6);
        }
        .neon-border-green {
            border-color: #059669;
            box-shadow: 0 0 15px rgba(5, 150, 105, 0.2);
        }
        .neon-btn {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        }
        .neon-btn:hover {
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.6);
        }
    </style>
</head>
<body class="futuristic-bg min-h-screen text-gray-200">
    <main class="max-w-7xl mx-auto py-12 px-6">
        <header class="mb-10 p-6 glass-panel rounded-xl flex items-center justify-between border border-gray-700/50">
            <div>
                <h1 class="text-4xl font-bold tracking-tight text-white flex items-center gap-3">
                    <span class="neon-text-green text-5xl">⌬</span> Plant Disease Classifier
                </h1>
                <p class="text-emerald-300 mt-2 text-sm font-semibold tracking-widest uppercase opacity-80">AI-Based Diagnostic Engine</p>
            </div>
            <div class="p-3 bg-gray-900/50 rounded-full text-emerald-400 text-sm font-mono border border-gray-700">
                //TELEMETRY::ACTIVE
            </div>
        </header>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <section class="lg:col-span-2 glass-panel p-8 rounded-2xl border border-gray-700/50">
                <div class="flex items-center justify-between mb-8 border-b border-gray-700/50 pb-4">
                    <h2 class="text-xl font-semibold text-white flex items-center gap-3">
                        <span class="neon-text-green">⎋</span> Execute Diagnostic Scan
                    </h2>
                    <span class="text-xs font-mono text-gray-500">SYSTEM READY :: PHASE_01</span>
                </div>
                
                <form id="diagnosticForm" class="grid grid-cols-1 md:grid-cols-5 gap-8">
                    <div class="md:col-span-3 border-2 border-dashed neon-border-green bg-gray-900/50 rounded-xl p-10 text-center hover:bg-gray-900 transition-colors duration-300 cursor-pointer group flex flex-col items-center justify-center">
                        <span class="text-6xl text-emerald-900 group-hover:text-emerald-700 transition-colors mb-4">🍃</span>
                        <label for="imageUpload" class="block text-sm font-semibold text-emerald-300 mb-3 group-hover:text-emerald-200">UPLOAD SPECIMEN TELEMETRY</label>
                        <input type="file" id="imageUpload" accept="image/*" class="w-full text-xs text-gray-400 file:mr-4 file:py-2.5 file:px-5 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-gray-800 file:text-emerald-300 hover:file:bg-gray-700 cursor-pointer transition-all" required>
                    </div>
                    
                    <div class="md:col-span-2 flex items-center justify-center">
                        <button type="submit" id="processBtn" class="w-full h-20 neon-btn text-gray-950 font-bold text-lg rounded-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-emerald-500 transform active:scale-[0.98]">
                            EXECUTE ANALYSIS // ⌬
                        </button>
                    </div>
                </form>
                
                <div id="resultsContainer" class="mt-8 hidden border-t border-gray-700 pt-6">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-900/70 p-6 rounded-xl border border-gray-700">
                        <div class="border-r md:border-emerald-900/50 pr-6">
                            <h3 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Diagnostic Classification</h3>
                            <p id="diseaseOutput" class="text-2xl font-bold neon-text-green"></p>
                            <p id="confidenceOutput" class="text-sm font-medium text-emerald-300 mt-1 font-mono"></p>
                        </div>
                        <div>
                            <h3 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Remediation Strategy // LLM GENERATED</h3>
                            <p id="treatmentOutput" class="text-sm text-gray-300 mt-1 leading-relaxed bg-black/40 p-5 rounded-xl border border-gray-800 font-mono"></p>
                        </div>
                    </div>
                </div>
            </section>

            <section class="glass-panel p-8 rounded-2xl border border-gray-700/50">
                <div class="flex justify-between items-center mb-6 border-b border-gray-700/50 pb-4">
                    <h2 class="text-xl font-semibold text-white flex items-center gap-3">
                        <span class="neon-text-green">📋</span> Diagnostic Logs
                    </h2>
                    <button id="refreshBtn" class="text-xs bg-gray-800 text-emerald-300 hover:bg-gray-700 hover:text-emerald-200 font-mono py-1.5 px-3 rounded-md transition-colors border border-gray-700">REFRESH [F5]</button>
                </div>
                <div class="overflow-x-auto rounded-xl border border-gray-700 bg-gray-900/50">
                    <table class="min-w-full divide-y divide-gray-700">
                        <thead class="bg-gray-800/80">
                            <tr>
                                <th scope="col" class="px-5 py-4 text-left text-xs font-bold text-emerald-300 uppercase tracking-widest">Pathology</th>
                                <th scope="col" class="px-5 py-4 text-left text-xs font-bold text-emerald-300 uppercase tracking-widest">Remediation</th>
                            </tr>
                        </thead>
                        <tbody id="historyTableBody" class="divide-y divide-gray-800 text-sm">
                            </tbody>
                    </table>
                </div>
            </section>
        </div>
    </main>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const form = document.getElementById('diagnosticForm');
            const processBtn = document.getElementById('processBtn');
            const resultsContainer = document.getElementById('resultsContainer');
            const refreshBtn = document.getElementById('refreshBtn');

            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                const fileInput = document.getElementById('imageUpload');
                if (!fileInput.files.length) return;

                const originalBtnText = processBtn.innerText;
                processBtn.innerText = "RUNNING TELEMETRY...";
                processBtn.disabled = true;
                processBtn.classList.add('opacity-75', 'cursor-not-allowed', 'shadow-none');

                const payload = new FormData();
                payload.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/api/predict', { method: 'POST', body: payload });
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    
                    const data = await response.json();
                    
                    document.getElementById('diseaseOutput').textContent = data.disease;
                    document.getElementById('confidenceOutput').textContent = `CONFIDENCE_SCORE::${(data.confidence).toFixed(2)}%`;
                    document.getElementById('treatmentOutput').textContent = data.treatment_plan;
                    resultsContainer.classList.remove('hidden');
                    
                    fetchSystemHistory();
                } catch (error) {
                    alert('An error occurred during processing. Please review console logs.');
                } finally {
                    processBtn.innerText = originalBtnText;
                    processBtn.disabled = false;
                    processBtn.classList.remove('opacity-75', 'cursor-not-allowed', 'shadow-none');
                }
            });

            refreshBtn.addEventListener('click', fetchSystemHistory);

            async function fetchSystemHistory() {
                try {
                    const response = await fetch('/api/history');
                    if (!response.ok) return;
                    const records = await response.json();
                    const tableBody = document.getElementById('historyTableBody');
                    tableBody.innerHTML = '';
                    
                    records.forEach(record => {
                        const row = document.createElement('tr');
                        row.className = "hover:bg-gray-800/50 transition-colors font-mono";
                        row.innerHTML = `<td class="px-5 py-4 whitespace-nowrap text-sm font-semibold text-emerald-300">[ID:${(records.indexOf(record)+1).toString().padStart(2,'0')}] ${record.disease}</td><td class="px-5 py-4 text-xs text-gray-400 leading-relaxed">${record.treatment_plan}</td>`;
                        tableBody.appendChild(row);
                    });
                } catch (error) {}
            }
            fetchSystemHistory();
        });
    </script>
</body>
</html>
"""

@app.get("/", include_in_schema=False)
def serve_frontend():
    # DIRECTLY RETURNS HTML - NO FOLDERS NEEDED!
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/api/predict", response_model=PredictionResponse)
async def process_diagnostic(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid payload. Expected an image file.")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            temp_file_path = tmp.name
        
        raw_disease, confidence = classifier.predict(temp_file_path)
        disease_name = raw_disease.replace('_', ' ')

        if "healthy" in disease_name.lower():
            treatment_plan = "Specimen appears healthy. Maintain standard agricultural practices."
        else:
            prompt = f"Provide a concise, professional, two-sentence agricultural treatment plan for a plant diagnosed with {disease_name}."
            
            # Enterprise Graceful Fallback Mechanism
            try:
                # First priority: Try the latest high-speed flash model
                model = genai.GenerativeModel('gemini-1.5-flash')
                llm_response = model.generate_content(prompt)
                treatment_plan = llm_response.text.strip()
            except Exception as e1:
                logger.warning(f"Flash model failed, falling back to Pro: {str(e1)}")
                try:
                    # Second priority: Try the highly stable pro model
                    model = genai.GenerativeModel('gemini-pro')
                    llm_response = model.generate_content(prompt)
                    treatment_plan = llm_response.text.strip()
                except Exception as e2:
                    logger.error(f"Both LLM models failed: {str(e2)}")
                    # Final safety fallback to prevent server 500 crashes
                    treatment_plan = f"Diagnostic complete for {disease_name}. Standard remediation: Isolate the affected specimen, remove infected leaves, and apply appropriate broad-spectrum agricultural treatment."
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO diagnostic_history (disease, confidence, treatment_plan, timestamp) VALUES (?, ?, ?, ?)",
                (disease_name, float(confidence), treatment_plan, timestamp)
            )
            conn.commit()

        return PredictionResponse(disease=disease_name, confidence=confidence, treatment_plan=treatment_plan)

    except Exception as e:
        logger.error(f"Error processing diagnostic: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during processing.")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/api/history", response_model=list[HistoryItem])
def fetch_diagnostic_history():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT disease, confidence, treatment_plan, timestamp FROM diagnostic_history ORDER BY id DESC LIMIT 10")
            rows = cursor.fetchall()
            
        return [HistoryItem(disease=row[0], confidence=round(row[1], 2), treatment_plan=row[2], timestamp=row[3]) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve history.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)