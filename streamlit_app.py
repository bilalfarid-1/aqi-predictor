import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path

from src.config import CITIES, DEFAULT_CITY, get_aqi_category
from src.data_ingestion import fetch_combined_data
from src.feature_engineering import create_features
from src.models.train import MultiHorizonForecaster
from src.explainability import compute_shap_explanations

st.set_page_config(
    page_title="Pearls AQI • AeroSense Precision",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Seamless iframe styling
st.markdown("""
<style>
    #MainMenu, header, footer { visibility: hidden; height: 0; }
    .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
        max-width: 100% !important;
    }
    iframe {
        width: 100% !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar Selection
c1, c2, c3 = st.columns([6, 3, 3])
with c1:
    st.markdown("<h3 style='font-family: Inter, sans-serif; font-weight: 700; color: #006c49; margin: 0; padding-top: 5px;'>Pearls AQI • AeroSense Precision Platform</h3>", unsafe_allow_html=True)
with c2:
    selected_city = st.selectbox("Monitoring Station", list(CITIES.keys()), index=0, label_visibility="collapsed")
with c3:
    model_choice = st.selectbox("Inference Model", ["XGBoost (V2)", "Random Forest", "LightGBM", "Ridge Baseline"], index=0, label_visibility="collapsed")

model_type_map = {
    "XGBoost (V2)": "xgboost",
    "Random Forest": "randomforest",
    "LightGBM": "lightgbm",
    "Ridge Baseline": "ridge"
}
model_type = model_type_map.get(model_choice, "xgboost")

@st.cache_data(ttl=600)
def get_data_payload(city, model_key):
    raw_df = fetch_combined_data(city=city, past_days=7, forecast_days=4)
    feature_df = create_features(raw_df, is_training=False)
    latest_row = feature_df.iloc[[-1]]

    current_aqi = float(latest_row['us_aqi'].values[0]) if 'us_aqi' in latest_row else 100.0
    category = get_aqi_category(current_aqi)

    try:
        forecaster = MultiHorizonForecaster.load()
    except Exception:
        forecaster = MultiHorizonForecaster(model_type=model_key)
        sample_df = fetch_combined_data(city=DEFAULT_CITY, past_days=7, forecast_days=4)
        sample_feats = create_features(sample_df, is_training=True)
        forecaster.train(sample_feats)

    predictions = forecaster.predict(latest_row)
    fc_24 = float(predictions[24][0])
    fc_48 = float(predictions[48][0])
    fc_72 = float(predictions[72][0])

    def get_val(col, default=0.0):
        return float(latest_row[col].values[0]) if col in latest_row else default

    pollutants = {
        "pm2_5": get_val("pm2_5", 40.0),
        "pm10": get_val("pm10", 45.0),
        "no2": get_val("no2", 30.0),
        "so2": get_val("so2", 10.0),
        "co": get_val("co", 50.0),
        "o3": get_val("o3", 80.0),
    }

    weather = {
        "temperature": get_val("temperature_2m", 28.0),
        "humidity": get_val("relative_humidity_2m", 65.0),
        "wind_speed": get_val("wind_speed_10m", 12.0),
        "pressure": get_val("surface_pressure", 1008.0),
    }

    shap_info = compute_shap_explanations(forecaster, latest_row, horizon=24)
    drivers = shap_info.get("top_drivers", []) if shap_info else []

    return {
        "city": city,
        "model": model_key,
        "current_aqi": round(current_aqi),
        "category": category,
        "forecast": {
            "t0": round(current_aqi),
            "t24": round(fc_24),
            "t48": round(fc_48),
            "t72": round(fc_72),
            "delta_24": round(fc_24 - current_aqi),
            "delta_48": round(fc_48 - current_aqi),
            "delta_72": round(fc_72 - current_aqi)
        },
        "pollutants": pollutants,
        "weather": weather,
        "drivers": drivers
    }

payload = get_data_payload(selected_city, model_type)

html_path = Path(__file__).resolve().parent / "app" / "static" / "index.html"
html_content = html_path.read_text(encoding="utf-8")

html_content = html_content.replace('<option value="Lahore">Lahore</option>', f'<option value="{selected_city}" selected>{selected_city}</option>')
html_content = html_content.replace('<option value="xgboost">XGBoost (V2)</option>', f'<option value="{model_type}" selected>{model_choice}</option>')

injection_script = f"""
<script>
    window.__PRELOADED_PAYLOAD__ = {json.dumps(payload)};
</script>
<script>
document.addEventListener('DOMContentLoaded', () => {{
    const p = window.__PRELOADED_PAYLOAD__;
    if (!p) return;

    const title = document.getElementById('header-city-title');
    if (title) title.textContent = p.city + ' — Executive Telemetry';

    const heroCard = document.getElementById('hero-card');
    const heroAqi = document.getElementById('hero-aqi-value');
    const heroDesc = document.getElementById('hero-desc');
    const badgeLabel = document.getElementById('hero-badge-label');
    const badgeIcon = document.getElementById('hero-badge-icon');

    if (heroAqi) heroAqi.textContent = p.current_aqi;
    if (heroDesc) heroDesc.textContent = p.category.desc;
    if (badgeLabel) badgeLabel.textContent = p.category.label.toUpperCase();

    if (heroCard) {{
        heroCard.className = "lg:col-span-8 flex flex-col justify-between p-8 md:p-12 relative overflow-hidden ";
        if (p.current_aqi <= 50) {{
            heroCard.classList.add('clay-card-emerald');
            badgeIcon.textContent = "check_circle";
        }} else if (p.current_aqi <= 100) {{
            heroCard.classList.add('clay-card-amber');
            badgeIcon.textContent = "info";
        }} else if (p.current_aqi <= 150) {{
            heroCard.classList.add('clay-card-orange');
            badgeIcon.textContent = "warning";
        }} else if (p.current_aqi <= 200) {{
            heroCard.classList.add('clay-card-rose');
            badgeIcon.textContent = "report_problem";
        }} else {{
            heroCard.classList.add('clay-card-purple');
            badgeIcon.textContent = "crisis_alert";
        }}
    }}

    const d24 = document.getElementById('delta-val-24');
    const d48 = document.getElementById('delta-val-48');
    const d72 = document.getElementById('delta-val-72');
    if (d24) d24.textContent = p.forecast.t24;
    if (d48) d48.textContent = p.forecast.t48;
    if (d72) d72.textContent = p.forecast.t72;

    const applyIcon = (id, delta) => {{
        const el = document.getElementById(id);
        if (!el) return;
        if (delta > 2) {{
            el.textContent = "trending_up";
            el.className = "material-symbols-outlined text-aqi-error";
        }} else if (delta < -2) {{
            el.textContent = "trending_down";
            el.className = "material-symbols-outlined text-aqi-good";
        }} else {{
            el.textContent = "trending_flat";
            el.className = "material-symbols-outlined text-outline";
        }}
    }};
    applyIcon('delta-icon-24', p.forecast.delta_24);
    applyIcon('delta-icon-48', p.forecast.delta_48);
    applyIcon('delta-icon-72', p.forecast.delta_72);

    if (document.getElementById('val-pm25')) document.getElementById('val-pm25').textContent = p.pollutants.pm2_5.toFixed(1);
    if (document.getElementById('val-no2')) document.getElementById('val-no2').textContent = p.pollutants.no2.toFixed(1);
    if (document.getElementById('val-o3')) document.getElementById('val-o3').textContent = p.pollutants.o3.toFixed(1);
    if (document.getElementById('val-co')) document.getElementById('val-co').textContent = p.pollutants.co.toFixed(1);

    if (document.getElementById('w-temp')) document.getElementById('w-temp').textContent = p.weather.temperature.toFixed(1) + ' °C';
    if (document.getElementById('w-wind')) document.getElementById('w-wind').textContent = p.weather.wind_speed.toFixed(1) + ' km/h';
    if (document.getElementById('w-hum')) document.getElementById('w-hum').textContent = Math.round(p.weather.humidity) + ' %';
    if (document.getElementById('w-pres')) document.getElementById('w-pres').textContent = Math.round(p.weather.pressure) + ' hPa';

    const cSpline = document.getElementById('forecastChart');
    if (cSpline) {{
        const ctx = cSpline.getContext('2d');
        const grad = ctx.createLinearGradient(0, 0, 0, 260);
        grad.addColorStop(0, 'rgba(0, 108, 73, 0.20)');
        grad.addColorStop(1, 'rgba(0, 108, 73, 0.00)');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: ['T-0 (Now)', '+24h', '+48h', '+72h'],
                datasets: [{{
                    data: [p.forecast.t0, p.forecast.t24, p.forecast.t48, p.forecast.t72],
                    borderColor: '#006c49',
                    borderWidth: 3,
                    tension: 0.35,
                    fill: true,
                    backgroundColor: grad,
                    pointBackgroundColor: '#006c49',
                    pointBorderColor: '#FFFFFF',
                    pointBorderWidth: 3,
                    pointRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});
    }}

    const cRadar = document.getElementById('radarChart');
    if (cRadar) {{
        new Chart(cRadar.getContext('2d'), {{
            type: 'radar',
            data: {{
                labels: ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
                datasets: [{{
                    data: [p.pollutants.pm2_5, p.pollutants.pm10, p.pollutants.no2, p.pollutants.so2, p.pollutants.co, p.pollutants.o3],
                    backgroundColor: 'rgba(0, 108, 73, 0.15)',
                    borderColor: '#006c49',
                    borderWidth: 2,
                    pointBackgroundColor: '#006c49',
                    pointRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ r: {{ grid: {{ color: '#bbcabf' }}, ticks: {{ display: false }} }} }}
            }}
        }});
    }}

    const shapBox = document.getElementById('shap-container');
    if (shapBox && p.drivers && p.drivers.length > 0) {{
        const maxAbs = Math.max(...p.drivers.map(d => Math.abs(d.shap_value)), 1.0);
        let rowsHtml = p.drivers.map(d => {{
            const isPos = d.shap_value > 0;
            const pct = Math.min(Math.round((Math.abs(d.shap_value) / maxAbs) * 100), 100);
            const colorClass = isPos ? 'bg-aqi-error' : 'bg-aqi-good';
            const sign = isPos ? '+' : '';
            return `
                <div class="flex items-center justify-between py-2 border-b border-outline-variant/30 text-xs font-data-tabular">
                    <span class="w-1/3 font-bold text-on-surface truncate">${{d.feature}}</span>
                    <div class="w-1/3 px-2 flex items-center">
                        <div class="w-full bg-slate-200 h-2 rounded-full overflow-hidden flex">
                            <div class="${{colorClass}} h-full rounded-full" style="width: ${{pct}}%"></div>
                        </div>
                    </div>
                    <div class="w-1/3 text-right">
                        <span class="font-bold ${{isPos ? 'text-aqi-error' : 'text-aqi-good'}}">${{sign}}${{d.shap_value.toFixed(2)}} pts</span>
                        <span class="text-on-surface-variant text-[10px] ml-1">(${{d.raw_value.toFixed(1)}})</span>
                    </div>
                </div>
            `;
        }}).join('');

        shapBox.innerHTML = `
            <div class="w-full h-full flex flex-col justify-between overflow-y-auto pr-2">
                <div class="flex justify-between pb-2 border-b border-outline-variant font-label-caps text-[11px] text-on-surface-variant uppercase">
                    <span class="w-1/3">Feature Name</span>
                    <span class="w-1/3 text-center">Relative Weight</span>
                    <span class="w-1/3 text-right">SHAP Impact (Raw)</span>
                </div>
                <div class="flex-1 flex flex-col justify-around py-1">${{rowsHtml}}</div>
            </div>
        `;
    }}

    const vDash = document.getElementById('view-dashboard');
    const vShap = document.getElementById('view-shap');
    const vMet = document.getElementById('view-telemetry');
    const btnDash = document.getElementById('nav-btn-dashboard');
    const btnShap = document.getElementById('nav-btn-shap');
    const btnMet = document.getElementById('nav-btn-telemetry');

    const switchTab = (tab) => {{
        if (vDash) vDash.classList.add('hidden');
        if (vShap) vShap.classList.add('hidden');
        if (vMet) vMet.classList.add('hidden');
        [btnDash, btnShap, btnMet].forEach(b => {{
            if (b) {{
                b.classList.remove('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
                b.classList.add('text-on-surface-variant');
            }}
        }});

        if (tab === 'dashboard') {{
            if (vDash) vDash.classList.remove('hidden');
            if (btnDash) btnDash.classList.add('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
        }} else if (tab === 'shap') {{
            if (vShap) vShap.classList.remove('hidden');
            if (btnShap) btnShap.classList.add('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
        }} else if (tab === 'telemetry') {{
            if (vMet) vMet.classList.remove('hidden');
            if (btnMet) btnMet.classList.add('bg-surface-variant', 'text-primary', 'font-bold', 'translate-x-1');
        }}
    }};

    if (btnDash) btnDash.addEventListener('click', () => switchTab('dashboard'));
    if (btnShap) btnShap.addEventListener('click', () => switchTab('shap'));
    if (btnMet) btnMet.addEventListener('click', () => switchTab('telemetry'));
}});
</script>
"""

html_content = html_content.replace('</body>', injection_script + '</body>')

components.html(html_content, height=1050, scrolling=True)
