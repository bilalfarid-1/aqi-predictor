// Pearls AQI • AeroSense Precision Client Logic
let forecastChartInstance = null;
let radarChartInstance = null;
let cachedShapData = null;

const citySelect = document.getElementById('city-select');
const modelSelect = document.getElementById('model-select');
const headerCityTitle = document.getElementById('header-city-title');

// Nav controllers
const navDashboard = document.getElementById('nav-btn-dashboard');
const navShap = document.getElementById('nav-btn-shap');
const navTelemetry = document.getElementById('nav-btn-telemetry');

const mobNavDash = document.getElementById('mob-nav-dash');
const mobNavShap = document.getElementById('mob-nav-shap');
const mobNavMet = document.getElementById('mob-nav-met');

const viewDashboard = document.getElementById('view-dashboard');
const viewShap = document.getElementById('view-shap');
const viewTelemetry = document.getElementById('view-telemetry');

function setView(activeView) {
    viewDashboard.classList.add('hidden');
    viewShap.classList.add('hidden');
    viewTelemetry.classList.add('hidden');

    [navDashboard, navShap, navTelemetry].forEach(btn => {
        if (btn) {
            btn.classList.remove('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
            btn.classList.add('text-on-surface-variant');
        }
    });

    if (activeView === 'dashboard') {
        viewDashboard.classList.remove('hidden');
        if (navDashboard) navDashboard.classList.add('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
    } else if (activeView === 'shap') {
        viewShap.classList.remove('hidden');
        if (navShap) navShap.classList.add('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
        renderShapBreakdown();
    } else if (activeView === 'telemetry') {
        viewTelemetry.classList.remove('hidden');
        if (navTelemetry) navTelemetry.classList.add('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
    }
}

if (navDashboard) navDashboard.addEventListener('click', () => setView('dashboard'));
if (navShap) navShap.addEventListener('click', () => setView('shap'));
if (navTelemetry) navTelemetry.addEventListener('click', () => setView('telemetry'));

if (mobNavDash) mobNavDash.addEventListener('click', () => setView('dashboard'));
if (mobNavShap) mobNavShap.addEventListener('click', () => setView('shap'));
if (mobNavMet) mobNavMet.addEventListener('click', () => setView('telemetry'));

citySelect.addEventListener('change', () => loadAllData());
modelSelect.addEventListener('change', () => loadAllData());

async function loadAllData() {
    const city = citySelect.value;
    const model = modelSelect.value;

    headerCityTitle.textContent = `${city} — Executive Telemetry`;

    try {
        const [telRes, fcRes, shapRes] = await Promise.all([
            fetch(`/api/telemetry/${city}`),
            fetch(`/api/forecast/${city}?model=${model}`),
            fetch(`/api/shap/${city}?model=${model}`)
        ]);

        const telemetry = await telRes.json();
        const forecast = await fcRes.json();
        cachedShapData = await shapRes.json();

        updateHeroCard(telemetry);
        updateDeltas(forecast);
        updateTelemetryGrid(telemetry);
        renderForecastSpline(forecast);
        renderRadarVector(telemetry.pollutants);
        updateMeteorology(telemetry.weather);

        if (!viewShap.classList.contains('hidden')) {
            renderShapBreakdown();
        }
    } catch (err) {
        console.error("Telemetry ingest error:", err);
    }
}

function updateHeroCard(tel) {
    const aqi = tel.current_aqi;
    const cat = tel.category;

    const heroCard = document.getElementById('hero-card');
    const heroAqi = document.getElementById('hero-aqi-value');
    const heroDesc = document.getElementById('hero-desc');
    const badgeLabel = document.getElementById('hero-badge-label');
    const badgeIcon = document.getElementById('hero-badge-icon');

    heroAqi.textContent = Math.round(aqi);
    heroDesc.textContent = cat.desc;
    badgeLabel.textContent = cat.label.toUpperCase();

    heroCard.className = "lg:col-span-8 flex flex-col justify-between p-8 md:p-12 relative overflow-hidden ";
    
    if (aqi <= 50) {
        heroCard.classList.add('clay-card-emerald');
        badgeIcon.textContent = "check_circle";
    } else if (aqi <= 100) {
        heroCard.classList.add('clay-card-amber');
        badgeIcon.textContent = "info";
    } else if (aqi <= 150) {
        heroCard.classList.add('clay-card-orange');
        badgeIcon.textContent = "warning";
    } else if (aqi <= 200) {
        heroCard.classList.add('clay-card-rose');
        badgeIcon.textContent = "report_problem";
    } else {
        heroCard.classList.add('clay-card-purple');
        badgeIcon.textContent = "crisis_alert";
    }
}

function updateDeltas(fc) {
    const f = fc.forecast;
    
    document.getElementById('delta-val-24').textContent = Math.round(f.t24.value);
    applyDeltaTrend('delta-icon-24', f.t24.delta);

    document.getElementById('delta-val-48').textContent = Math.round(f.t48.value);
    applyDeltaTrend('delta-icon-48', f.t48.delta);

    document.getElementById('delta-val-72').textContent = Math.round(f.t72.value);
    applyDeltaTrend('delta-icon-72', f.t72.delta);

    const banner = document.getElementById('hazard-banner');
    if (fc.is_hazardous) {
        banner.classList.remove('hidden');
        document.getElementById('hazard-message').textContent = fc.hazard_message;
    } else {
        banner.classList.add('hidden');
    }
}

function applyDeltaTrend(iconId, delta) {
    const icon = document.getElementById(iconId);
    if (!icon) return;
    if (delta > 2) {
        icon.textContent = "trending_up";
        icon.className = "material-symbols-outlined text-aqi-error";
    } else if (delta < -2) {
        icon.textContent = "trending_down";
        icon.className = "material-symbols-outlined text-aqi-good";
    } else {
        icon.textContent = "trending_flat";
        icon.className = "material-symbols-outlined text-outline";
    }
}

function updateTelemetryGrid(tel) {
    const p = tel.pollutants;
    document.getElementById('val-pm25').textContent = p.pm2_5.toFixed(1);
    document.getElementById('val-no2').textContent = p.no2.toFixed(1);
    document.getElementById('val-o3').textContent = p.o3.toFixed(1);
    document.getElementById('val-co').textContent = p.co.toFixed(1);
}

function updateMeteorology(w) {
    document.getElementById('w-temp').textContent = `${w.temperature.toFixed(1)} °C`;
    document.getElementById('w-wind').textContent = `${w.wind_speed.toFixed(1)} km/h`;
    document.getElementById('w-hum').textContent = `${Math.round(w.humidity)} %`;
    document.getElementById('w-pres').textContent = `${Math.round(w.pressure)} hPa`;
}

function renderForecastSpline(fc) {
    const f = fc.forecast;
    const labels = ["T-0 (Now)", "+24h", "+48h", "+72h"];
    const values = [f.t0.value, f.t24.value, f.t48.value, f.t72.value];

    const canvas = document.getElementById('forecastChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (forecastChartInstance) forecastChartInstance.destroy();

    const gradient = ctx.createLinearGradient(0, 0, 0, 260);
    gradient.addColorStop(0, 'rgba(0, 108, 73, 0.20)');
    gradient.addColorStop(1, 'rgba(0, 108, 73, 0.00)');

    forecastChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                borderColor: '#006c49',
                borderWidth: 3,
                tension: 0.35,
                fill: true,
                backgroundColor: gradient,
                pointBackgroundColor: '#006c49',
                pointBorderColor: '#FFFFFF',
                pointBorderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#191c1e',
                    titleFont: { family: 'Inter', size: 12, weight: '700' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    padding: 10,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `Projected AQI: ${Math.round(ctx.raw)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { family: 'JetBrains Mono', size: 11 }, color: '#6f7a72' }
                },
                y: {
                    min: 0,
                    max: Math.max(...values, 200) + 30,
                    grid: { color: 'rgba(187, 202, 191, 0.25)' },
                    ticks: { font: { family: 'JetBrains Mono', size: 11 }, color: '#6f7a72' }
                }
            }
        }
    });
}

function renderRadarVector(pollutants) {
    const canvas = document.getElementById('radarChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (radarChartInstance) radarChartInstance.destroy();

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
            datasets: [{
                data: [
                    pollutants.pm2_5,
                    pollutants.pm10,
                    pollutants.no2,
                    pollutants.so2,
                    pollutants.co,
                    pollutants.o3
                ],
                backgroundColor: 'rgba(0, 108, 73, 0.15)',
                borderColor: '#006c49',
                borderWidth: 2,
                pointBackgroundColor: '#006c49',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                r: {
                    grid: { color: '#bbcabf' },
                    ticks: { display: false },
                    pointLabels: { font: { family: 'Inter', size: 11, weight: '700' }, color: '#3f4943' }
                }
            }
        }
    });
}

function renderShapBreakdown() {
    const container = document.getElementById('shap-container');
    if (!container) return;

    if (!cachedShapData || !cachedShapData.top_drivers) {
        container.innerHTML = '<div class="text-xs text-on-surface-variant font-data-tabular">Loading SHAP explanations...</div>';
        return;
    }

    const drivers = cachedShapData.top_drivers;
    if (drivers.length === 0) {
        container.innerHTML = '<div class="text-xs text-on-surface-variant font-data-tabular">No SHAP attribution records available.</div>';
        return;
    }

    const maxAbs = Math.max(...drivers.map(d => Math.abs(d.shap_value)), 1.0);

    let rowsHtml = drivers.map(d => {
        const isPos = d.shap_value > 0;
        const pct = Math.min(Math.round((Math.abs(d.shap_value) / maxAbs) * 100), 100);
        const colorClass = isPos ? 'bg-aqi-error' : 'bg-aqi-good';
        const sign = isPos ? '+' : '';

        return `
            <div class="flex items-center justify-between py-2 border-b border-outline-variant/30 text-xs font-data-tabular">
                <span class="w-1/3 font-bold text-on-surface truncate">${d.feature}</span>
                <div class="w-1/3 px-2 flex items-center">
                    <div class="w-full bg-slate-200 h-2 rounded-full overflow-hidden flex">
                        <div class="${colorClass} h-full rounded-full" style="width: ${pct}%"></div>
                    </div>
                </div>
                <div class="w-1/3 text-right">
                    <span class="font-bold ${isPos ? 'text-aqi-error' : 'text-aqi-good'}">${sign}${d.shap_value.toFixed(2)} pts</span>
                    <span class="text-on-surface-variant text-[10px] ml-1">(${d.raw_value.toFixed(1)})</span>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div class="w-full h-full flex flex-col justify-between overflow-y-auto pr-2">
            <div class="flex justify-between pb-2 border-b border-outline-variant font-label-caps text-[11px] text-on-surface-variant uppercase">
                <span class="w-1/3">Feature Name</span>
                <span class="w-1/3 text-center">Relative Weight</span>
                <span class="w-1/3 text-right">SHAP Impact (Raw)</span>
            </div>
            <div class="flex-1 flex flex-col justify-around py-1">
                ${rowsHtml}
            </div>
        </div>
    `;
}

document.addEventListener('DOMContentLoaded', () => loadAllData());
