# Metro Bike Share Demand Intelligence

![Top Models](https://img.shields.io/badge/Top%20Models-XGBoost%20%7C%20LightGBM%20%7C%20SARIMAX-6A5ACD?style=flat-square)
![Scope](https://img.shields.io/badge/Scope-System%20%2B%20Station%20Level-2F80C9?style=flat-square)
![Forecasting](https://img.shields.io/badge/Forecasting-Multi--Horizon%20Forecasting-2F80C9?style=flat-square)
![Benchmarks](https://img.shields.io/badge/Benchmarks-Naive%20%7C%20Seasonal%20Naive%20%7C%20ETS-E88A45?style=flat-square)
![Validation](https://img.shields.io/badge/Validation-MASE%20Backtesting%20at%207%20and%2030%20Days-2F80C9?style=flat-square)
![Output](https://img.shields.io/badge/Output-Artifacts%20%2B%20Dashboard-E88A45?style=flat-square)

Metro Bike Share does not have one demand problem.

It has a **network problem**: how much total demand is coming?  
And it has a **station problem**: where will that demand actually show up?

This repository was revised to answer both.

It keeps the original SQL work visible, adds a cleaner warehouse and forecasting structure around it, and turns the project into a usable decision system: diagnose the signal, compare models honestly, generate forecasts with uncertainty, and review everything in one dashboard.


> **What this system is really for**  
> Better demand visibility before demand becomes an operations problem.

---

## Architecture

This project is still worth continuing because many data-oriented initiatives across companies are now being revisited and modernized. In this case, the earlier work was built on a strong foundation, especially in the extraction of business logic and the development of an enhanced data model. 

![Data model placeholder](sql/data_model/datamodel.webp)

That foundation should be preserved because it still provides real value. At the same time, renewal is necessary: forecasting requires more than legacy SQL logic. It needs a clearer and more modern path from raw data to diagnosis, from diagnosis to modeling, and from modeling to production-ready pipelines that a team can review, trust, and use.

That is why the project is structured as an end-to-end workflow that spans data modeling and standardization through to dashboards and monitoring.:

| Layer | Purpose & Value |
|------|----------------|
| Legacy SQL | Preserves trusted historical logic and provides business rules with cleaned foundations |
| Warehouse Layer | Creates a stronger engineering bridge with reusable structure, contracts, and utilities |
| Diagnosis | Helps understand the signal before modeling, leading to better assumptions and cleaner framing |
| Forecasting | Enables time-aware model comparison and produces actionable predictions at multiple levels |
| Dashboard | Makes results easier to read and share, supporting faster review and clearer communication |


---

## Two forecasting views

<p align="center">
  <img src="dashboard/Station-Level Forecasting.png" width="45%" style="display:inline-block; margin-right:10px;" />
  <img src="dashboard/System-Level Forecasting.png" width="45%" style="display:inline-block;" />
</p>

This project uses two views because planning and operations need different kinds of visibility.

### System view
The system-level side treats the network as one demand series. It is the planning view: cleaner, smoother, and easier to interpret for direction, seasonality, horizon design, and overall forecast quality.

### Station view
The station-level side keeps the local picture. It is the operations view: noisier, more heterogeneous, and much closer to the actual places where demand decisions matter. It is where maturity, sparsity, categories, and clusters become important.


The point is not to choose one over the other. The point is to let them work together:
>- the **system view** gives confidence in overall demand structure
>- the **station view** shows where the average breaks down

---

## Forecasting Methodology

This is not a repo built around one favorite model or one flattering score.

Instead, the project is built around a more credible evaluation pattern:

- compare multiple model families, not just one
- evaluate by horizon, not one pooled average
- evaluate by slice where station behavior differs
- keep uncertainty visible through forecast intervals
- separate diagnosis from forecasting so model choices have context

That is what makes the work stronger as a product, not just cleaner as a repository.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="forecasts/station_level/figures/station_level_model_comparison.png" alt="Station-level model comparison" width="95%">
      <br>
      <sub><b>Station level</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="forecasts/system_level/figures/system_level_model_comparison.png" alt="System-level model comparison" width="95%">
      <br>
      <sub><b>System level</b></sub>
    </td>
  </tr>
</table>

## The four pipeline paths

The easiest way to understand the system is to read it as four connected pipelines.

| Pipeline | Role | Results |
|---|---|---|
| System-Level Diagnosis | Understand aggregate signal quality, seasonality, and structure | [System-Level Diagnosis](diagnosis/system_level/) |
| Station-Level Diagnosis | Understand heterogeneity, sparsity, maturity, and behavior slices | [Station-Level Diagnosis](diagnosis/station_level/) |
| System-Level Forecasting | Build the network-wide planning baseline | [System-Level Forecasting](forecasts/system_level) |
| Station-Level Forecasting | Build the local station-day operational view | [Station-Level Forecasting](forecasts/station_level) |


## Dashboard Demo

You can preview the dashboard through the key sections below:

<p align="center">
  <img src="dashboard/Station-Level Diagnosis.png" width="45%" />
  <img src="dashboard/Station-Level Forecasting.png" width="45%" /><br/>
  <img src="dashboard/System-Level Diagnosis.png" width="45%" />
  <img src="dashboard/System-Level Forecasting.png" width="45%" />
</p>

Each image highlights a different section of the dashboard for quick exploration.

[Open the full dashboard demo video](dashboard/dashboard-demo.mov)
---

<details>
<summary><strong>Setup</strong></summary>

Use a non-system Python `>=3.10` such as Homebrew Python on macOS.

```bash
# Clone repo
git clone https://github.com/sastmo/Metro-Bike-Share.git
cd Metro-Bike-Share

# Create and activate virtual environment
# Windows:
py -3 -m venv .venv
.venv\Scripts\activate        # CMD
.venv\Scripts\Activate.ps1    # PowerShell

# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Set `POSTGRES_URL` only if you want PostgreSQL persistence.

</details>

<details>
<summary><strong>Common commands</strong></summary>

| Task | Command | Outputs |
|---|---|---|
| System-level diagnosis | `python -m system_level.cli diagnose --level system <dataset.csv> --target-col trip_count --time-col bucket_start --segment-type system_total --segment-id all` | `diagnosis/system_level/outputs/` |
| Station-level diagnosis | `python -m system_level.cli diagnose --level station --input <station_daily.csv> --date-col date --station-col station_id --target-col target` | `diagnosis/station_level/outputs/` |
| System-level forecasting | `python -m system_level.cli forecast --level system --config configs/system_level/config.yaml --verbose` | `forecasts/system_level/` |
| Station-level forecasting | `python -m system_level.cli forecast --level station --config configs/station_level/config.yaml --verbose` | `forecasts/station_level/` |
| Dashboard | `python -m streamlit run scripts/dashboard/run_dashboard.py` | `http://localhost:8501` |

Optional runtime check before forecasting:

```bash
python -m system_level.cli doctor --level station --config configs/station_level/config.yaml
```

</details>

