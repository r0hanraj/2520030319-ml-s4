const form = document.getElementById("predictionForm");
const resultCard = document.getElementById("resultCard");
const formError = document.getElementById("formError");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";

  const data = Object.fromEntries(new FormData(form).entries());
  resultCard.innerHTML = `
    <div class="result-placeholder">
      <div class="result-symbol">…</div>
      <h3>Running ensemble</h3>
      <p>Preprocessing the application and collecting model votes.</p>
    </div>`;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });

    const result = await response.json();
    if (!response.ok || !result.success) throw new Error(result.error || "Prediction failed.");

    const approved = result.status === "APPROVED";
    const voteRows = Object.entries(result.votes).map(([name, vote]) => `
      <div class="vote ${vote === "Rejected" ? "reject" : ""}">
        <span>${name}</span><b>${vote}</b>
      </div>
    `).join("");

    resultCard.innerHTML = `
      <div class="result-content">
        <span class="eyebrow">ENSEMBLE RESULT</span>
        <div class="result-status ${approved ? "approved" : "rejected"}">${result.status}</div>
        <div class="result-sub">The hard-voting ensemble has produced a ${approved ? "positive" : "negative"} class prediction.</div>
        <div class="agreement">
          <div class="agreement-line"><span>Base-model agreement</span><strong>${result.agreement}%</strong></div>
          <div class="progress"><i style="width:${result.agreement}%"></i></div>
        </div>
        <div class="vote-list">${voteRows}</div>
        <p class="matrix-note" style="margin-top:18px">${result.note}</p>
      </div>`;
  } catch (error) {
    formError.textContent = error.message;
    resultCard.innerHTML = `
      <div class="result-placeholder">
        <div class="result-symbol">!</div>
        <h3>Prediction unavailable</h3>
        <p>Check the entered values and try again.</p>
      </div>`;
  }
});
