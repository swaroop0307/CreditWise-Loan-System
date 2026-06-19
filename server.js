const express = require('express');
const cors    = require('cors');
const path    = require('path');
const { spawn } = require('child_process');

const app  = express();
const PORT = 3000;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// Serve index.html, style.css, script.js from the same folder as server.js
app.use(express.static(path.join(__dirname)));

// ── Prediction Route ──────────────────────────────────────────────────────────
app.post('/predict', (req, res) => {
    // Validate that a body was sent
    if (!req.body || Object.keys(req.body).length === 0) {
        return res.status(400).json({ status: 'error', message: 'Empty request body.' });
    }

    const inputData = JSON.stringify(req.body);

    // Spawn Python — use 'python3' on Linux/Mac, 'python' on Windows
    const pythonBin     = process.platform === 'win32' ? 'python' : 'python3';
    const pythonProcess = spawn(pythonBin, [
        path.join(__dirname, 'predict.py'),
        inputData
    ]);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', chunk => { stdout += chunk.toString(); });
    pythonProcess.stderr.on('data', chunk => { stderr += chunk.toString(); });

    pythonProcess.on('close', code => {
        try {
            const result = JSON.parse(stdout.trim());

            if (result.status === 'success') {
                res.json(result);
            } else {
                res.status(400).json(result);
            }
        } catch (parseErr) {
            console.error('Python STDERR:', stderr);
            console.error('Python STDOUT:', stdout);
            res.status(500).json({
                status:  'error',
                message: 'Failed to parse Python output.',
                details: stderr || stdout
            });
        }
    });

    pythonProcess.on('error', err => {
        res.status(500).json({
            status:  'error',
            message: `Could not start Python process: ${err.message}. Is Python installed?`
        });
    });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`\n🏦  SecureTrust Bank — CreditWise Backend`);
    console.log(`   Running at:  http://localhost:${PORT}`);
    console.log(`   Open this URL in your browser to use the app.\n`);
});
