"""Project-wide constants."""

# European Standard Population weights (2013 revision) for age standardization.
# Age groups: 0-4, 5-9, ..., 85+
# Source: Eurostat
EUROPEAN_STANDARD_POPULATION = {
    "0-4": 5000,
    "5-9": 5500,
    "10-14": 5500,
    "15-19": 5500,
    "20-24": 6000,
    "25-29": 6000,
    "30-34": 6500,
    "35-39": 7000,
    "40-44": 7000,
    "45-49": 7000,
    "50-54": 7000,
    "55-59": 6500,
    "60-64": 6000,
    "65-69": 5500,
    "70-74": 5000,
    "75-79": 4000,
    "80-84": 2500,
    "85+": 2500,
}

# Summer months (June–September)
SUMMER_MONTHS = [6, 7, 8, 9]

# Italian NUTS-2 region codes (2021 classification)
NUTS2_CODES = [
    "ITC1",  # Piemonte
    "ITC2",  # Valle d'Aosta
    "ITC3",  # Liguria
    "ITC4",  # Lombardia
    "ITH1",  # Trentino-Alto Adige
    "ITH2",  # Veneto
    "ITH3",  # Friuli-Venezia Giulia
    "ITH4",  # Emilia-Romagna
    "ITI1",  # Toscana
    "ITI2",  # Umbria
    "ITI3",  # Marche
    "ITI4",  # Lazio
    "ITF1",  # Abruzzo
    "ITF2",  # Molise
    "ITF3",  # Campania
    "ITF4",  # Puglia
    "ITF5",  # Basilicata
    "ITF6",  # Calabria
    "ITG1",  # Sicilia
    "ITG2",  # Sardegna
]

# Kelvin to Celsius offset
KELVIN_OFFSET = 273.15

# CRS for Italian geographic data
CRS_WGS84 = "EPSG:4326"
CRS_ITALY = "EPSG:32632"  # UTM zone 32N (covers most of Italy)
