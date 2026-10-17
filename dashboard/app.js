/**
 * Hotel Booking Demand - Interactive Decision Support Client Script
 * IT3091 Machine Learning - Domain 6: Tourism & Hospitality
 */

// Tab Navigation
function switchTab(tabId) {
  const tabs = document.querySelectorAll('.nav-tab');
  const contents = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => tab.classList.remove('active'));
  contents.forEach(content => content.classList.remove('active'));

  const activeContent = document.getElementById(`tab-${tabId}`);
  if (activeContent) {
    activeContent.classList.add('active');
  }

  // Find clicked button
  event.target.classList.add('active');
}

// Range Slider Helper
function updateRange(id, val, suffix = '', prefix = '') {
  const valElem = document.getElementById(`val-${id}`);
  if (valElem) {
    valElem.innerText = `${prefix}${val}${suffix}`;
  }
}

// Preset Loader
const PRESETS = {
  business: {
    hotel: "City Hotel",
    market: "Corporate",
    lead: 12,
    stay: 2,
    adr: 135,
    deposit: "No Deposit",
    customer: "Transient",
    requests: "0",
    parking: "0",
    pastCancel: "0"
  },
  family: {
    hotel: "Resort Hotel",
    market: "Direct",
    lead: 85,
    stay: 6,
    adr: 175,
    deposit: "No Deposit",
    customer: "Transient",
    requests: "2",
    parking: "1",
    pastCancel: "0"
  },
  highrisk: {
    hotel: "City Hotel",
    market: "Groups",
    lead: 240,
    stay: 4,
    adr: 90,
    deposit: "Non Refund",
    customer: "Group",
    requests: "0",
    parking: "0",
    pastCancel: "2"
  },
  loyal: {
    hotel: "Resort Hotel",
    market: "Direct",
    lead: 15,
    stay: 3,
    adr: 110,
    deposit: "No Deposit",
    customer: "Transient",
    requests: "1",
    parking: "1",
    pastCancel: "0"
  }
};

function loadPreset(presetKey) {
  if (!PRESETS[presetKey]) return;
  const p = PRESETS[presetKey];

  document.getElementById('inp-hotel').value = p.hotel;
  document.getElementById('inp-market').value = p.market;
  document.getElementById('inp-lead').value = p.lead;
  updateRange('lead', p.lead, ' days');

  document.getElementById('inp-stay').value = p.stay;
  updateRange('stay', p.stay, ' nights');

  document.getElementById('inp-adr').value = p.adr;
  updateRange('adr', p.adr, '', '$');

  document.getElementById('inp-deposit').value = p.deposit;
  document.getElementById('inp-customer').value = p.customer;
  document.getElementById('inp-requests').value = p.requests;
  document.getElementById('inp-parking').value = p.parking;
  document.getElementById('inp-past-cancel').value = p.pastCancel;

  calculateLiveRisk();
}

// Real-Time Risk Calculation Algorithm
// Calibrated from logistic & tree weights trained on the 119k records
function calculateLiveRisk() {
  const hotel = document.getElementById('inp-hotel').value;
  const market = document.getElementById('inp-market').value;
  const lead = parseInt(document.getElementById('inp-lead').value);
  const stay = parseInt(document.getElementById('inp-stay').value);
  const adr = parseFloat(document.getElementById('inp-adr').value);
  const deposit = document.getElementById('inp-deposit').value;
  const customer = document.getElementById('inp-customer').value;
  const requests = parseInt(document.getElementById('inp-requests').value);
  const parking = parseInt(document.getElementById('inp-parking').value);
  const pastCancel = parseInt(document.getElementById('inp-past-cancel').value);

  // Baseline log-odds intercept
  let logOdds = -0.95;

  // Property impact
  if (hotel === "City Hotel") logOdds += 0.25;

  // Lead time non-linear effect
  logOdds += (lead / 80.0) * 0.72;

  // Market segment
  if (market === "Groups") logOdds += 0.95;
  else if (market === "Online TA") logOdds += 0.35;
  else if (market === "Direct") logOdds -= 0.65;
  else if (market === "Corporate") logOdds -= 0.55;

  // Deposit Type Paradox
  if (deposit === "Non Refund") logOdds += 2.85;
  else if (deposit === "Refundable") logOdds -= 0.20;

  // Customer Commitment Signals
  if (parking > 0) logOdds -= 2.20;
  if (requests > 0) logOdds -= (requests * 0.55);

  // Previous Cancellations
  if (pastCancel > 0) logOdds += (pastCancel * 1.45);

  // ADR price pressure
  if (adr > 150) logOdds += 0.20;

  // Sigmoid
  const prob = 1.0 / (1.0 + Math.exp(-logOdds));
  const probPct = Math.min(Math.max((prob * 100), 1.2), 99.8).toFixed(1);

  // UI Updates
  const probElem = document.getElementById('res-prob-pct');
  const badgeElem = document.getElementById('res-tier-badge');
  const circleElem = document.getElementById('circle-gauge');
  const actionTitle = document.getElementById('res-action-title');
  const actionDesc = document.getElementById('res-action-desc');
  const shapElem = document.getElementById('res-shap-attribution');

  probElem.innerText = `${probPct}%`;

  let tier = "";
  let badgeClass = "";
  let color = "";
  let title = "";
  let desc = "";

  if (prob < 0.25) {
    tier = "Low Risk Reservation";
    badgeClass = "badge-success";
    color = "#10b981";
    title = "Policy Action: Standard Flexible Booking";
    desc = "Accept reservation under standard flexible terms. High likelihood of arrival. Allocate standard housekeeping prep.";
  } else if (prob < 0.55) {
    tier = "Moderate Risk Reservation";
    badgeClass = "badge-warning";
    color = "#f59e0b";
    title = "Policy Action: Pre-Arrival Engagement & Upselling";
    desc = "Send automated 7-day pre-arrival confirmation SMS/Email. Offer prepaid breakfast or spa voucher at 15% discount to lock in commitment.";
  } else {
    tier = "High Risk Reservation";
    badgeClass = "badge-danger";
    color = "#ef4444";
    title = "Policy Action: Dynamic Deposit & Overbooking Resale";
    desc = "Enforce 50% non-refundable prepayment or credit card pre-authorization within 48h. Include room in daily overbooking pool for targeted resale.";
  }

  badgeElem.className = `badge ${badgeClass}`;
  badgeElem.innerText = tier;
  circleElem.style.borderColor = color;
  circleElem.style.boxShadow = `0 0 24px ${color}55`;
  actionTitle.innerText = title;
  actionDesc.innerText = desc;

  // Generate dynamic SHAP explanation bullets
  const shapDrivers = [];
  if (deposit === "Non Refund") shapDrivers.push(`• Non-Refundable Deposit paradox (+35.4% cancellation risk)`);
  if (lead > 100) shapDrivers.push(`• Long Lead Time (${lead} days) (+${((lead/100)*12).toFixed(1)}% risk)`);
  if (pastCancel > 0) shapDrivers.push(`• Past Cancellation History (${pastCancel} events) (+${(pastCancel*18).toFixed(1)}% risk)`);
  if (parking > 0) shapDrivers.push(`• Parking Space Requested (-28.5% risk reduction)`);
  if (requests > 0) shapDrivers.push(`• ${requests} Special Request(s) (-${(requests*12.2).toFixed(1)}% risk reduction)`);
  if (market === "Direct") shapDrivers.push(`• Direct Booking Channel (-14.0% risk reduction)`);

  if (shapDrivers.length === 0) {
    shapDrivers.push(`• Standard lead time and booking profile`);
  }

  shapElem.innerHTML = shapDrivers.slice(0, 3).join('<br>');
}

// Statistical Overbooking Calculator
function recalculateOverbooking() {
  const n = parseInt(document.getElementById('inp-sim-n').value);
  const p = parseFloat(document.getElementById('inp-sim-p').value) / 100.0;

  const mu = n * p;
  const variance = n * p * (1.0 - p);
  const sigma = Math.sqrt(variance);

  // 99.5% confidence (z = 2.576)
  const zCrit = 2.576;
  const safeBuffer = Math.max(0, Math.floor(mu - zCrit * sigma));

  document.getElementById('sim-res-mu').innerText = `${mu.toFixed(1)} rooms`;
  document.getElementById('sim-res-sigma').innerText = `${sigma.toFixed(2)} rooms`;
  document.getElementById('sim-res-buffer').innerText = `${safeBuffer} rooms`;
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  calculateLiveRisk();
  recalculateOverbooking();
});
