import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Simulasi Interferensi Selaput Tipis",
    page_icon="🔬",
    layout="wide"
)

# ============================================================================
# CSS CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    .header-container {
        background-color: #006994;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .header-container h1 {
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .header-container h6 {
        color: white;
        text-align: center;
        margin: 0.0rem 0;
    }
    .card-container {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #006994;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .footer-container {
        background-color: #006994;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 0rem;
        text-align: center;
        color: white;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header-container">
    <h1>Simulasi Interferensi Selaput Tipis</h1>
    <h6>Dikembangkan oleh Felix Marcellino Henrikus, S.Si.</h6>
    <h6>Program Studi Magister Sains Data, UKSW Salatiga</h6>
    <h6>Untuk digunakan dalam pembelajaran Optika Gelombang di S1 Fisika, UKSW Salatiga</h6>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# INFORMASI TAMBAHAN
# ============================================================================
with st.expander("📚 Panduan Penggunaan"):
    st.markdown("""
    ### Cara Menggunakan Simulasi Ini:
    
    1. **Pilih Jumlah Layer**: Tentukan jumlah medium (3-8 layer)
    2. **Susun Medium**: Pilih medium untuk setiap layer dari database yang tersedia
    3. **Atur Ketebalan**: Masukkan ketebalan setiap layer dalam satuan mm
    4. **Pilih Panjang Gelombang**: Dari UV (200 nm) hingga IR (1100 nm)
    5. **Atur Sudut Datang**: 0° (normal) hingga 90° (grazing)
    6. **Pilih Mode**: Manual atau Optimal Thickness untuk anti-reflection coating
    
    ### Informasi Fisika:
    
    - **Pembalikan Fase**: Terjadi ketika cahaya merambat dari medium dengan indeks bias lebih rendah ke lebih tinggi
    - **Ketebalan Optimal**: d = λ/(4n) untuk minimum reflection
    - **Transmitansi**: T + R + A = 1 (konservasi energi)
    
    ### Referensi:
    
    - Hecht, E. (2017). Optics (5th ed.). Pearson.
    - Born, M., & Wolf, E. (1999). Principles of Optics. Cambridge University Press.
    """)

with st.expander("📋 Database Medium"):
    st.markdown("""
    | Medium | Indeks Bias (n) |
    |--------|-----------------|
    | Udara | 1.00 |
    | Etanol | 1.36 |
    | Air | 1.33 |
    | MgF₂ | 1.38 |
    | Minyak Goreng | 1.45 |
    | SiO₂ | 1.46 |
    | Gliserin | 1.47 |
    | Crown Glass | 1.52 |
    """)

# ============================================================================
# DATABASE MEDIUM
# ============================================================================
MEDIUM_DATABASE = {
    "Udara": 1.00,
    "Etanol": 1.36,
    "Air": 1.33,
    "MgF₂": 1.38,
    "Minyak Goreng": 1.45,
    "SiO₂": 1.46,
    "Gliserin": 1.47,
    "Crown Glass": 1.52
}

# ============================================================================
# FUNGSI FISIS
# ============================================================================

def calculate_phase_change(n1, n2):
    """
    Menghitung apakah terjadi pembalikan fase pada interface.
    Returns: True jika terjadi pembalikan fase (π), False jika sefase
    """
    if n2 > n1:
        return True  # Pembalikan fase terjadi
    else:
        return False  # Tidak ada pembalikan fase

def calculate_optical_path_difference(n, d, theta_incident, wavelength):
    """
    Menghitung beda lintasan optik untuk interferensi
    """
    # Hukum Snellius
    theta_refracted = np.arcsin(np.sin(np.radians(theta_incident)) / n)
    
    # Beda lintasan optik
    opd = 2 * n * d * np.cos(theta_refracted)
    
    return opd, np.degrees(theta_refracted)

def calculate_reflection_coefficient(n1, n2, theta_incident, polarization='s'):
    """
    Menghitung koefisien refleksi menggunakan persamaan Fresnel
    """
    theta_i = np.radians(theta_incident)
    
    try:
        theta_t = np.arcsin(n1 * np.sin(theta_i) / n2)
    except:
        return 1.0  # Total internal reflection
    
    if polarization == 's':
        r = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / \
            (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
    else:  # p-polarization
        r = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / \
            (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
    
    return r ** 2

def calculate_transmittance_reflectance_multilayer(layers, thicknesses, wavelength_nm, theta_incident=0):
    """
    Menghitung transmitansi dan reflektansi untuk multilayer thin film
    menggunakan metode Transfer Matrix yang benar dengan konservasi energi
    
    Returns: T, R (dengan T + R <= 1)
    """
    
    # Konversi satuan
    wavelength_m = wavelength_nm * 1e-9  # nm → meter
    theta_0 = np.radians(theta_incident)  # derajat → radian
    
    n_0 = layers[0]  # Medium incident
    n_s = layers[-1]  # Substrat
    
    # Hitung sudut di setiap layer (Snell's law)
    thetas = []
    for j, n_j in enumerate(layers):
        sin_theta_j = n_0 * np.sin(theta_0) / n_j
        if abs(sin_theta_j) <= 1:
            theta_j = np.arcsin(sin_theta_j)
        else:
            theta_j = np.pi / 2  # Total internal reflection
        thetas.append(theta_j)
    
    # === BUILD TRANSFER MATRIX ===
    M = np.array([[1, 0], [0, 1]], dtype=complex)
    
    for j in range(len(layers) - 1):
        n_j = layers[j]
        n_j1 = layers[j + 1]
        theta_j = thetas[j]
        theta_j1 = thetas[j + 1]
        
        # Dapatkan ketebalan layer j (dalam meter)
        if j < len(thicknesses):
            d_j_m = thicknesses[j] * 1e-3  # mm → meter
        else:
            d_j_m = 0
        
        # === Characteristic matrix untuk layer j+1 ===
        # δ = (2π/λ) * n * d * cos(θ)
        if d_j_m > 0:
            delta_j = (2 * np.pi / wavelength_m) * n_j1 * d_j_m * np.cos(theta_j1)
            
            # Matriks karakteristik layer
            cos_delta = np.cos(delta_j)
            sin_delta = np.sin(delta_j)
            
            # Admittance untuk s-polarization
            eta_j1 = n_j1 * np.cos(theta_j1)
            
            m_11 = cos_delta
            m_12 = 1j * sin_delta / eta_j1
            m_21 = 1j * eta_j1 * sin_delta
            m_22 = cos_delta
            
            M_layer = np.array([[m_11, m_12], [m_21, m_22]], dtype=complex)
            M = M @ M_layer
    
    # === HITUNG R DAN T DARI MATRIX TOTAL ===
    # Admittance medium incident dan exit
    eta_0 = n_0 * np.cos(theta_0)
    eta_s = n_s * np.cos(thetas[-1])
    
    # Koefisien refleksi
    numerator = (M[0, 0] + M[0, 1] * eta_s) * eta_0 - (M[1, 0] + M[1, 1] * eta_s)
    denominator = (M[0, 0] + M[0, 1] * eta_s) * eta_0 + (M[1, 0] + M[1, 1] * eta_s)
    
    if abs(denominator) < 1e-12:
        r = 0
    else:
        r = numerator / denominator
    
    # Koefisien transmitansi
    if abs(denominator) < 1e-12:
        t = 0
    else:
        t = 2 * eta_0 / denominator
    
    # Reflektansi
    R = np.abs(r) ** 2
    
    # Transmitansi (dengan koreksi impedansi untuk konservasi energi)
    T = np.abs(t) ** 2 * (eta_s.real / eta_0.real) if eta_0.real > 0 else 0
    
    # === NORMALISASI UNTUK KONSERVASI ENERGI ===
    # Pastikan T + R <= 1
    total = T + R
    
    if total > 1.0:
        # Normalisasi proporsional
        scale = 1.0 / total
        T = T * scale
        R = R * scale
    
    # Batasi nilai antara 0 dan 1
    T = max(0.0, min(1.0, T.real))
    R = max(0.0, min(1.0, R.real))
    
    return T, R

def calculate_optimal_thickness(n_film, n_substrate, wavelength, theta_incident=0):
    """
    Menghitung ketebalan optimal untuk minimizing reflection (anti-reflection coating)
    Returns: ketebalan dalam satuan mm
    """
    theta_rad = np.radians(theta_incident)
    
    # Ketebalan optimal untuk destructive interference
    # d = λ/(4n) - hasil dalam nm karena wavelength dalam nm
    d_optimal_nm = wavelength / (4 * n_film * np.cos(theta_rad))
    
    # Konversi nm ke mm: 1 nm = 10^-6 mm
    d_optimal_mm = d_optimal_nm * 1e-6
    
    return d_optimal_mm

def calculate_absorbance(transmittance, reflectance):
    """
    Menghitung absorbansi dari transmitansi dan reflektansi
    A = 1 - T - R
    """
    return max(0, 1 - transmittance - reflectance)

# ============================================================================
# SIDEBAR - INPUT PARAMETERS
# ============================================================================
st.sidebar.markdown("""
<div class="card-container">
    <h3>⚙️ Parameter Simulasi</h3>
</div>
""", unsafe_allow_html=True)

# Jumlah layer
num_layers = st.sidebar.slider(
    "Jumlah Medium (Layer)",
    min_value=3,
    max_value=8,
    value=4,
    help="Pilih jumlah lapisan medium (minimal 3, maksimal 8)"
)

# Pilihan medium untuk setiap layer
st.sidebar.markdown("### 📦 Susunan Medium")
layers = []
for i in range(num_layers):
    medium = st.sidebar.selectbox(
        f"Medium Layer {i+1}",
        options=list(MEDIUM_DATABASE.keys()),
        index=min(i, len(MEDIUM_DATABASE)-1),
        key=f"medium_{i}"
    )
    layers.append(MEDIUM_DATABASE[medium])

# Ketebalan setiap layer
st.sidebar.markdown("### 📏 Ketebalan Layer (mm)")
thicknesses = []
for i in range(num_layers - 1):  # Layer terakhir adalah substrate, tidak perlu ketebalan
    thickness = st.sidebar.number_input(
        f"Ketebalan Layer {i+1}",
        min_value=0.0,  # ✅ Bisa 0 mm (terbuka bebas)
        max_value=10.0,
        value=0.001,
        step=0.0001,
        format="%.4f",
        key=f"thickness_{i}",
        help="0 mm = terbuka bebas di alam"
    )
    thicknesses.append(thickness)

# Panjang gelombang
st.sidebar.markdown("### 🌈 Panjang Gelombang")
wavelength = st.sidebar.slider(
    "Panjang Gelombang (nm)",
    min_value=200,
    max_value=1100,
    value=550,
    step=10,
    help="UV (200-400 nm), Visible (400-700 nm), IR (700-1100 nm)"
)

# Sudut datang
st.sidebar.markdown("### 📐 Sudut Datang")
theta_incident = st.sidebar.slider(
    "Sudut Datang (derajat)",
    min_value=0,
    max_value=90,
    value=0,
    step=1,
    help="0° = normal, 90° = grazing incidence"
)

# Mode kalkulasi
st.sidebar.markdown("### 🔍 Mode Kalkulasi")
calculation_mode = st.sidebar.radio(
    "Pilih Mode",
    options=["Manual", "Optimal Thickness"],
    help="Manual: gunakan ketebalan yang ditentukan\nOptimal: hitung ketebalan untuk minimizing reflection (hanya 3 layer)"
)

# ✅ REVISI: Jika Optimal Thickness, paksa 3 layer
if calculation_mode == "Optimal Thickness":
    st.sidebar.warning("⚠️ Mode Optimal Thickness hanya tersedia untuk **3 layer** (Medium 1 - Film - Substrat)")
    num_layers = 3  # Paksa 3 layer

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Card Container untuk Informasi Fase
st.markdown("""
<div class="card-container">
    <h3>📊 Informasi Pembalikan Fase</h3>
</div>
""", unsafe_allow_html=True)

# Tabel informasi fase
phase_info_data = []
for i in range(len(layers) - 1):
    n1 = layers[i]
    n2 = layers[i + 1]
    phase_change = calculate_phase_change(n1, n2)
    phase_info_data.append({
        "Interface": f"Layer {i+1} → Layer {i+2}",
        "n₁": n1,
        "n₂": n2,
        "Δn": n2 - n1,
        "Status": "🔄 Pembalikan Fase (π)" if phase_change else "✓ Sefase (0)"
    })

# Display table
phase_df = pd.DataFrame(phase_info_data)
st.dataframe(phase_df, use_container_width=True, hide_index=True)

# Card Container untuk Perhitungan
st.markdown("""
<div class="card-container">
    <h3>🧮 Perhitungan Fisika</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Persamaan Dasar")
    st.latex(r"""
    \text{Beda Lintasan Optik: } \Delta = 2nd\cos\theta
    """)
    st.latex(r"""
    \text{Interferensi Konstruktif: } \Delta = m\lambda
    """)
    st.latex(r"""
    \text{Interferensi Destruktif: } \Delta = (m+\frac{1}{2})\lambda
    """)
    st.latex(r"""
    \text{Hukum Snellius: } n_1\sin\theta_1 = n_2\sin\theta_2
    """)

with col2:
    st.markdown("### Koefisien Fresnel")
    st.latex(r"""
    R_s = \left|\frac{n_1\cos\theta_i - n_2\cos\theta_t}{n_1\cos\theta_i + n_2\cos\theta_t}\right|^2
    """)
    st.latex(r"""
    R_p = \left|\frac{n_2\cos\theta_i - n_1\cos\theta_t}{n_2\cos\theta_i + n_1\cos\theta_t}\right|^2
    """)
    st.latex(r"""
    T = 1 - R - A
    """)

# ============================================================================
# KETEBALAN OPTIMAL (Revisi untuk 3 Layer)
# ============================================================================
if calculation_mode == "Optimal Thickness":
    st.markdown("""
    <div class="card-container">
        <h3>🎯 Ketebalan Optimal (Anti-Reflection Coating)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ Pastikan konfigurasi 3 layer: Medium 1 - Film - Substrat
    if len(layers) == 3:
        n_air = layers[0]      # Medium pertama (biasanya udara)
        n_film = layers[1]     # Layer film (coating)
        n_substrate = layers[2] # Substrat
        
        # Hitung ketebalan optimal untuk panjang gelombang yang dipilih
        d_optimal = calculate_optimal_thickness(n_film, n_substrate, wavelength, theta_incident)
        
        # ✅ FIX: Update thicknesses dengan benar
        # Layer 1 (udara) = 0 mm (terbuka)
        # Layer 2 (film) = d_optimal
        thicknesses_optimal = [0.0, d_optimal]
        
        # Konversi ke nm untuk display yang lebih mudah dibaca
        d_optimal_nm = d_optimal * 1e6  # mm ke nm
        
        st.markdown(f"""
        ### Konfigurasi 3 Layer:
        
        | Layer | Medium | Indeks Bias (n) | Ketebalan |
        |-------|--------|-----------------|-----------|
        | 1 | {list(MEDIUM_DATABASE.keys())[list(MEDIUM_DATABASE.values()).index(n_air)]} | {n_air:.2f} | 0.0000 mm (terbuka) |
        | 2 | {list(MEDIUM_DATABASE.keys())[list(MEDIUM_DATABASE.values()).index(n_film)]} | {n_film:.2f} | {d_optimal:.6f} mm ({d_optimal_nm:.2f} nm) |
        | 3 | {list(MEDIUM_DATABASE.keys())[list(MEDIUM_DATABASE.values()).index(n_substrate)]} | {n_substrate:.2f} | ∞ (substrat) |
        
        ### Perhitungan Ketebalan Optimal:
        
        $$d_{{\\text{{optimal}}}} = \\frac{{\\lambda}}{{4n_{{\\text{{film}}}}}} = \\frac{{{wavelength} \\text{{ nm}}}}{{4 \\times {n_film:.2f}}} = {d_optimal_nm:.2f} \\text{{ nm}} = {d_optimal:.6f} \\text{{ mm}}$$
        
        ### Kondisi Ideal Anti-Reflection Coating:
        
        $$n_{{\\text{{film}}}} = \\sqrt{{n_{{\\text{{air}}}} \\times n_{{\\text{{substrate}}}}}} = \\sqrt{{{n_air:.2f} \\times {n_substrate:.2f}}} = {np.sqrt(n_air * n_substrate):.4f}$$
        
        **Indeks bias film saat ini:** {n_film:.2f}  
        **Indeks bias ideal:** {np.sqrt(n_air * n_substrate):.4f}  
        **Status:** {'✅ Optimal' if abs(n_film - np.sqrt(n_air * n_substrate)) < 0.05 else '⚠️ Tidak Ideal (pilih MgF₂ untuk hasil terbaik)'}
        """)
        
        # ✅ Gunakan thicknesses_optimal untuk kalkulasi
        thicknesses_to_use = thicknesses_optimal
    else:
        st.error("❌ Mode Optimal Thickness memerlukan tepat **3 layer**!")
        st.info("Silakan ubah jumlah layer menjadi 3 di sidebar.")
        thicknesses_to_use = thicknesses

# ============================================================================
# PLOT HASIL SIMULASI (Revisi untuk Optimal Thickness)
# ============================================================================
st.markdown("""
<div class="card-container">
    <h3>📈 Hasil Simulasi</h3>
</div>
""", unsafe_allow_html=True)

# Tentukan thicknesses yang akan digunakan
if calculation_mode == "Optimal Thickness" and len(layers) == 3:
    n_film = layers[1]
    n_substrate = layers[2]
    d_optimal = calculate_optimal_thickness(n_film, n_substrate, wavelength, theta_incident)
    thicknesses_to_use = [0.0, d_optimal]
else:
    thicknesses_to_use = thicknesses

# Kalkulasi untuk seluruh range panjang gelombang
wavelength_range = np.linspace(200, 1100, 300)
transmittance_values = []
reflectance_values = []
absorbance_values = []

for wl in wavelength_range:
    T, R = calculate_transmittance_reflectance_multilayer(layers, thicknesses_to_use, wl, theta_incident)
    A = calculate_absorbance(T, R)
    
    transmittance_values.append(T)
    reflectance_values.append(R)
    absorbance_values.append(A)

if calculation_mode == "Manual":
    # Diagram Batang untuk Mode Manual
    fig = go.Figure()
    
    idx = int(np.argmin(np.abs(wavelength_range - wavelength)))
    T_val = transmittance_values[idx]
    R_val = reflectance_values[idx]
    A_val = absorbance_values[idx]
    
    fig.add_trace(go.Bar(
        x=['Transmitansi (T)', 'Reflektansi (R)', 'Absorbansi (A)'],
        y=[T_val, R_val, A_val],
        marker_color=['#2ecc71', '#e74c3c', '#3498db'],
        text=[f'{T_val:.4f}', f'{R_val:.4f}', f'{A_val:.4f}'],
        textposition='outside'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Parameter",
        yaxis_title="Intensitas Relatif",
        yaxis_range=[0, 1.1],
        template='plotly_white',
        title=f"Intensitas pada λ = {wavelength} nm",
        showlegend=False
    )
else:
    # ✅ Line Chart untuk Mode Optimal - Spektrum Penuh UV-Vis-IR
    fig = go.Figure()
    
    # Plot Transmitansi
    fig.add_trace(
        go.Scatter(
            x=wavelength_range, 
            y=transmittance_values, 
            name='Transmitansi (T)',
            line=dict(color='#2ecc71', width=3, shape='spline'),
            mode='lines',
            fill=None
        )
    )
    
    # Plot Reflektansi
    fig.add_trace(
        go.Scatter(
            x=wavelength_range, 
            y=reflectance_values, 
            name='Reflektansi (R)',
            line=dict(color='#e74c3c', width=3, shape='spline'),
            mode='lines',
            fill=None
        )
    )
    
    # Plot Absorbansi
    fig.add_trace(
        go.Scatter(
            x=wavelength_range, 
            y=absorbance_values, 
            name='Absorbansi (A)',
            line=dict(color='#3498db', width=3, shape='spline'),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        )
    )
    
    # Garis vertikal pada wavelength yang dipilih
    fig.add_vline(
        x=wavelength, 
        line_dash="dash", 
        line_color="black", 
        line_width=2,
        annotation_text=f"λ = {wavelength} nm", 
        annotation_position="top"
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Panjang Gelombang (nm)",
        yaxis_title="Intensitas Relatif",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        title="Spektrum Absorbansi, Transmitansi, dan Reflektansi (UV - Visible - IR)",
        xaxis=dict(
            range=[200, 1100],
            tickmode='array',
            tickvals=[200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
            ticktext=['200', '300', '400', '500', '600', '700', '800', '900', '1000', '1100'],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            range=[0, 1.05],
            tickformat='.2f',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Tambahkan region shading untuk UV, Visible, IR
    fig.add_vrect(
        x0=200, x1=400, 
        fillcolor="purple", 
        opacity=0.1, 
        layer="below",
        annotation_text="UV", 
        annotation_position="top",
        line_width=0
    )
    fig.add_vrect(
        x0=400, x1=700, 
        fillcolor="yellow", 
        opacity=0.1, 
        layer="below",
        annotation_text="Visible", 
        annotation_position="top",
        line_width=0
    )
    fig.add_vrect(
        x0=700, x1=1100, 
        fillcolor="orange", 
        opacity=0.1, 
        layer="below",
        annotation_text="IR", 
        annotation_position="top",
        line_width=0
    )

st.plotly_chart(fig, use_container_width=True)

# Informasi pada panjang gelombang yang dipilih
st.markdown("### 📍 Hasil pada Panjang Gelombang Terpilih")

T_current, R_current = calculate_transmittance_reflectance_multilayer(
    layers, thicknesses_to_use, wavelength, theta_incident
)
A_current = calculate_absorbance(T_current, R_current)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Transmitansi (T)", f"{T_current:.4f}", f"{T_current*100:.2f}%")

with col2:
    st.metric("Reflektansi (R)", f"{R_current:.4f}", f"{R_current*100:.2f}%")

with col3:
    st.metric("Absorbansi (A)", f"{A_current:.4f}", f"{A_current*100:.2f}%")

# Visualisasi lapisan
st.markdown("""
<div class="card-container">
    <h3>🔬 Visualisasi Susunan Lapisan</h3>
</div>
""", unsafe_allow_html=True)

# Buat diagram lapisan
layer_names = list(MEDIUM_DATABASE.keys())
layer_indices = [list(MEDIUM_DATABASE.values()).index(n) for n in layers]

fig_layers = go.Figure()

for i, (n, name_idx) in enumerate(zip(layers, layer_indices)):
    fig_layers.add_trace(
        go.Bar(
            x=[f"Layer {i+1}"],
            y=[n],
            name=f"{list(MEDIUM_DATABASE.keys())[name_idx]} (n={n:.2f})",
            marker_color=f'rgb({50 + i*30}, {100 + i*20}, {150 + i*10})',
            text=[f"n = {n:.2f}"],
            textposition='outside'
        )
    )

fig_layers.update_layout(
    height=400,
    xaxis_title="Layer",
    yaxis_title="Indeks Bias (n)",
    barmode='group',
    showlegend=True,
    template='plotly_white'
)

st.plotly_chart(fig_layers, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer-container">
    <p>© 2026 - Felix Marcellino Henrikus, S.Si. - UKSW Salatiga</p>
</div>
""", unsafe_allow_html=True)


