# Quick Reference: Real Data Sources for Seawall Optimization

**Last Updated**: November 2024

---

## Top 5 Essential Datasets

### 1. NOAA Tide Gauge Data (Water Levels)
- **URL**: https://api.tidesandcurrents.noaa.gov/api/prod/
- **What**: Historical hourly water levels at 200+ coastal stations
- **Format**: JSON, CSV, XML
- **Best for**: Extracting annual maxima, fitting GEV distributions
- **Example API call**:
  ```
  https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?
  product=hourly_height&station=8761724&begin_date=20140101&
  end_date=20231231&datum=MLLW&units=english&format=json
  ```

### 2. NOAA HURDAT2 (Hurricane Tracks)
- **URL**: https://www.nhc.noaa.gov/data/
- **What**: All Atlantic hurricanes 1851-2024 with tracks and intensity
- **Format**: Comma-delimited text
- **File size**: ~7 MB
- **Best for**: Historical storm frequency and intensity distributions

### 3. NOAA SLOSH Model Output (Storm Surge)
- **URL**: https://www.nhc.noaa.gov/nationalsurge/
- **What**: Probabilistic surge heights from 100,000 simulated storms
- **Format**: GeoTIFF
- **Best for**: Establishing baseline surge distributions by location
- **Products**: MEOWs (by category), MOMs (maximum of all)

### 4. CoNED Coastal Elevation Data
- **URL**: https://coast.noaa.gov/dataviewer/
- **What**: High-resolution topobathymetric DEMs for US coasts
- **Format**: GeoTIFF
- **Resolution**: 1-10m
- **Best for**: Ground elevations, coastal slopes, infrastructure heights

### 5. GEBCO Global Bathymetry
- **URL**: https://download.gebco.net/
- **What**: Global ocean depth and land elevation
- **Format**: NetCDF, GeoTIFF
- **Resolution**: 15 arc-seconds (~450m)
- **Best for**: Baseline seafloor topography

---

## Recommended Study Region

### **Gulf Coast: Louisiana** (Best Data Availability)

**Specific Segment**: Grand Isle to Shell Beach, LA (~100 miles)

**Why This Region**:
- ✓ Highest storm frequency (best statistical power)
- ✓ 10+ tide gauges with 30+ years data
- ✓ Extensive SLOSH coverage
- ✓ Real-world case studies (New Orleans levees)
- ✓ Documented construction costs ($3.7B-$14.5B projects)
- ✓ CoNED high-resolution elevation data

**Key Tide Gauge Stations**:
- 8761724: Grand Isle, LA
- 8764227: Shell Beach, LA
- 8760922: Pilots Station East, LA
- 8768094: Calcasieu Pass, LA

**Historical Hurricanes**:
- Katrina (2005): 25+ ft surge
- Rita (2005): 15+ ft surge
- Gustav (2008): 12+ ft surge
- Isaac (2012): 11+ ft surge
- Laura (2020): 17+ ft surge
- Ida (2021): 15+ ft surge

---

## Essential Research Papers

### Must-Read (Storm Surge Modeling):
1. **Jafarinejad et al. (2023)** - "Storm surge hazard estimation along the US Gulf Coast: A Bayesian hierarchical approach"
   - *Coastal Engineering*
   - Provides spatial surge estimates with uncertainty

2. **Rohmer et al. (2024)** - "Projecting U.S. coastal storm surge risks with deep learning"
   - arXiv:2506.13963
   - 900,000 synthetic TCs for probabilistic analysis

### Must-Read (Seawall Optimization):
3. **Yoe et al. (2018)** - "Cost-Benefit Analysis of Building Levees: Miami Case Study"
   - *Water*, 10(2), 169
   - Real-world CBA methodology

4. **Yadav et al. (2021)** - "Optimization of Coastal Protections in the Presence of Climate Change"
   - *Frontiers in Climate*, 3, 613293
   - Multi-location spatial optimization

### Must-Read (Extreme Value Analysis):
5. **Tebaldi et al. (2020)** - "Probabilistic reanalysis of storm surge extremes in Europe"
   - *PNAS*, 117(4), 1877-1883
   - Gold standard for GEV spatial analysis

---

## Data Processing Pipeline

### Step 1: Data Collection (2-3 weeks)
```
1. Select 20 coastal points along 5-mile segment
2. Download 30+ years hourly water levels (NOAA API)
3. Download SLOSH MEOWs/MOMs for region
4. Download CoNED elevation data
5. Extract HURDAT2 storms affecting region
```

### Step 2: Extreme Value Analysis (1-2 weeks)
```python
# For each location:
1. Extract annual maximum water levels
2. Fit GEV distribution: stats.genextreme.fit(annual_max)
3. Calculate return levels: 10yr, 50yr, 100yr
4. Validate with Q-Q plots and K-S test
5. Compare with SLOSH model outputs
```

### Step 3: Spatial Correlation (1 week)
```python
# Across locations:
1. Build empirical correlation matrix
2. Fit exponential decay: ρ(d) = ρ₀^(d/d₀)
3. Typical values: ρ₀ ≈ 0.6-0.8, d₀ ≈ 1-5 miles
```

### Step 4: Generate Events (1 week)
```python
# Synthetic correlated events:
1. Generate correlated normal samples
2. Transform to uniform via Φ(z)
3. Transform to GEV via F^(-1)(u)
4. Create 5,000-10,000 events
5. Validate: check marginals and correlations
```

### Step 5: Cost Calibration (1 week)
```python
# From literature and case studies:
- Base unit cost: $150-600 per linear foot per foot height
- Quadratic model: C(h) = a + b*h + c*h²
- Typical values:
  - b ≈ $100,000-200,000 per sector
  - c ≈ $50,000-100,000 per sector
- Spatial variation: ±30% based on access/conditions
```

---

## Construction Cost Reference

### Cost per Linear Foot (2024 dollars)
- Vinyl seawall: $200-600/ft
- Steel seawall: $250-700/ft
- Concrete seawall: $200-800/ft
- Commercial/high-erosion: $700-2,000/ft

### Height Factors
- 5 ft wall: baseline
- 8-10 ft wall: ~2x cost
- 15+ ft wall: ~3-4x cost

### Large-Scale Projects
- Louisiana West Shore: $3.7B / 17.5 miles = **$211M per mile**
- New Orleans system: $14.5B / 130 miles = **$111M per mile**
- Typical range: **$50M-$250M per mile**

### For 0.25-mile sectors:
- **Base cost**: $12.5M - $60M per sector
- **Unit cost (b)**: $100,000 - $200,000 per ft height
- **Quadratic (c)**: $50,000 - $100,000 per ft²

---

## Python Code Snippets

### Fetch NOAA Tide Data
```python
import requests
import pandas as pd

url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
params = {
    'product': 'hourly_height',
    'station': '8761724',  # Grand Isle, LA
    'begin_date': '20140101',
    'end_date': '20231231',
    'datum': 'MLLW',
    'units': 'english',
    'format': 'json'
}

response = requests.get(url, params=params)
data = response.json()
df = pd.DataFrame(data['data'])
df['water_level'] = pd.to_numeric(df['v'], errors='coerce')
```

### Fit GEV Distribution
```python
from scipy import stats
import numpy as np

# Extract annual maxima
df['year'] = pd.to_datetime(df['t']).dt.year
annual_max = df.groupby('year')['water_level'].max()

# Fit GEV
shape, loc, scale = stats.genextreme.fit(annual_max)

# Calculate 100-year return level
p_100yr = 1 - 1/100
z_100yr = stats.genextreme.ppf(p_100yr, shape, loc, scale)
print(f"100-year surge: {z_100yr:.2f} ft")
```

### Generate Correlated Events
```python
# Copula approach
n_sectors = 20
n_events = 5000

# Correlation matrix (exponential decay)
rho = 0.7
corr_matrix = np.array([[rho**abs(i-j) for j in range(n_sectors)]
                        for i in range(n_sectors)])

# Generate correlated normal
normal_samples = np.random.multivariate_normal(
    mean=np.zeros(n_sectors),
    cov=corr_matrix,
    size=n_events
)

# Transform to uniform
uniform_samples = stats.norm.cdf(normal_samples)

# Transform to GEV for each sector
surge_events = np.zeros((n_events, n_sectors))
for i in range(n_sectors):
    surge_events[:, i] = stats.genextreme.ppf(
        uniform_samples[:, i],
        shape[i], loc[i], scale[i]
    )
```

---

## Validation Checklist

### Statistical Validation
- [ ] GEV fit p-value > 0.05 (K-S test)
- [ ] Q-Q plot shows good alignment
- [ ] Return levels match SLOSH MOMs (within 20%)
- [ ] Spatial correlation follows exponential decay

### Physical Validation
- [ ] 100-year surge matches historical max events
- [ ] Optimal heights exceed historical max surge
- [ ] Spatial patterns make physical sense (higher in exposed areas)

### Economic Validation
- [ ] Total cost within order of magnitude of real projects
- [ ] Cost per mile: $50M-$250M (Gulf Coast)
- [ ] Optimal heights in feasible range: 10-20 ft
- [ ] Higher protection in high-value areas

---

## Common Gulf Coast Tide Gauge Stations

| Station | Name | Lat | Lon | Years | Key Storms |
|---------|------|-----|-----|-------|------------|
| 8761724 | Grand Isle, LA | 29.26 | -89.96 | 1947+ | All major Gulf storms |
| 8764227 | Shell Beach, LA | 29.87 | -89.67 | 1997+ | Katrina, Isaac, Ida |
| 8760922 | Pilots Station East | 28.93 | -89.41 | 1982+ | Katrina, Gustav |
| 8768094 | Calcasieu Pass, LA | 29.77 | -93.34 | 1936+ | Rita, Laura |
| 8771450 | Galveston Pier 21 | 29.31 | -94.79 | 1904+ | Ike, Harvey |
| 8770570 | Sabine Pass North | 29.73 | -93.87 | 1958+ | Rita, Ike |

---

## Key Contacts

- **NOAA Tide Data**: tide.predictions@noaa.gov
- **NOAA Digital Coast**: coastal.info@noaa.gov
- **USGS Coastal**: https://www.usgs.gov/coastal

---

## Installation Requirements

```bash
# Essential packages
pip install numpy scipy pandas requests

# Optional for GIS
pip install geopandas rasterio netCDF4

# NOAA data access
pip install py_noaa

# Advanced extreme value analysis
pip install pyextremes
```

---

## File Structure for Real Data

```
data/
├── raw/
│   ├── tide_gauges/
│   │   └── station_XXXXXXX.csv
│   ├── slosh/
│   │   └── meow_cat1-5.tif
│   ├── elevation/
│   │   └── coned_region.tif
│   └── hurdat2/
│       └── hurdat2_1851_2024.txt
│
├── processed/
│   ├── surge_distributions/
│   │   └── sector_XX_gev_params.json
│   ├── spatial_correlation_matrix.npy
│   └── cost_parameters.json
│
└── synthetic/
    ├── real_data_floods_5000events.npz
    └── real_data_params.json
```

---

## Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Data Collection | 2-3 weeks | Raw datasets |
| GEV Analysis | 2 weeks | Parameters for 20 locations |
| Spatial Correlation | 1 week | Correlation matrix |
| Event Generation | 1 week | 5,000 synthetic events |
| Cost Calibration | 1 week | Cost function params |
| Validation | 1 week | Comparison with reality |
| **TOTAL** | **8-11 weeks** | **Real-data optimization** |

---

## Next Steps

1. **Review full documentation**: `REAL_DATA_SOURCES.md`
2. **Run code examples**: `python code/real_data_processor.py`
3. **Select study region**: Louisiana Coast recommended
4. **Begin data collection**: Start with tide gauge API
5. **Perform GEV analysis**: Fit distributions to annual maxima
6. **Generate events**: Create spatially-correlated flood scenarios
7. **Run optimization**: Use with existing `seawall_model.py`

---

**For detailed methodology and references, see**: `data/REAL_DATA_SOURCES.md`
