document.getElementById('loanForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const submitBtn  = document.getElementById('submitBtn');
    const btnText    = document.getElementById('btnText');
    const resultCard = document.getElementById('resultCard');

    const keys = [
        'Gender', 'Age', 'Marital_Status', 'Education_Level', 'Dependents',
        'Applicant_Income', 'Coapplicant_Income', 'Employment_Status',
        'Employer_Category', 'Credit_Score', 'Savings', 'Existing_Loans',
        'Loan_Amount', 'Collateral_Value', 'Loan_Term', 'DTI_Ratio',
        'Loan_Purpose', 'Property_Area'
    ];

    const payload = {};
    let missing = [];

    keys.forEach(key => {
        const el  = document.getElementById(key);
        const val = el ? el.value.trim() : '';
        if (!val) missing.push(key.replace(/_/g, ' '));
        payload[key] = val;
    });

    if (missing.length > 0) {
        showResult('error', '⚠️', 'Missing Fields', `Please fill in: ${missing.join(', ')}`);
        return;
    }

    // Loading state
    btnText.textContent = '⏳ Analysing Risk Factors...';
    submitBtn.disabled  = true;
    resultCard.className = 'result-card hidden';

    try {
        const response = await fetch('http://127.0.0.1:3000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || data.status !== 'success') {
            throw new Error(data.message || `Server error ${response.status}`);
        }

        if (data.approved === 1) {
            showResult('approved', '🎉', 'Application Approved',
                `Model Confidence: <strong>${data.confidence}%</strong>`);
        } else {
            showResult('rejected', '❌', 'Application Rejected',
                `Risk Probability: <strong>${data.confidence}%</strong>`);
        }

    } catch (err) {
        console.error(err);
        const msg = err.message.includes('Failed to fetch')
            ? 'Cannot reach backend server. Is <code>node server.js</code> running on port 3000?'
            : err.message;
        showResult('error', '⚠️', 'Error', msg);
    } finally {
        btnText.textContent = '⚡ Evaluate Loan Application';
        submitBtn.disabled  = false;
    }
});

function showResult(type, icon, title, detail) {
    const card = document.getElementById('resultCard');
    const cls  = type === 'approved' ? 'approved'
               : type === 'rejected' ? 'rejected'
               : 'error-state';
    card.className = `result-card ${cls}`;
    card.innerHTML = `
        <div class="result-icon">${icon}</div>
        <div class="result-title">${title}</div>
        <div class="result-conf">${detail}</div>
    `;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
}
