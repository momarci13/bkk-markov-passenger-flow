# Manually exported tables

`scripts/tdk_demand_data.py` reads optional tables from this folder. They are used
when present, and the script falls back to the documented default when one is missing.

| file | columns | source |
|---|---|---|
| `bkk_modal_split.csv` | mode, route_type, share | BKK trips by mode (KSH STADAT 24.1.1.21 / BKK report) |
| `district_population.csv` | district, population | KSH Census 2022, by district (e.g. "IV. kerület") |
| `district_cars.csv` | district, cars_per_1000 | KSH TIMEA / BP-STAR, passenger cars per 1,000 residents |
| `district_pt_share.csv` | district, pt_share | Census 2022 commuters by mode: public-transport share |
| `commuting_od.csv` | origin, destination, commuters | Census 2022 district-to-district public-transport commuters |

District names follow the geoBoundaries labels used in `scripts/budapest_basemap.py`.
