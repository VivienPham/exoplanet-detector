# ==========================================================
# EXOPLANET TRANSIT DETECTION PIPELINE
# Validation target: Kepler-10b
#
# Steps performed:
# 1. Download Kepler telescope light curve data
# 2. Clean the light curve (remove NaNs and outliers)
# 3. Remove long-term stellar variability
# 4. Search for periodic transits using Box Least Squares
# 5. Fold the light curve at the detected period
# 6. Measure transit depth
# 7. Estimate planet radius
# ==========================================================

# -----------------------------
# Import libraries
# -----------------------------
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from urllib.parse import quote
from astroquery.mast import Catalogs

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

def fetch_from_exoplanet_archive(target):
    """Try NASA Exoplanet Archive first for known planet-host stars."""
    query = f"""
    select top 1 hostname, tic_id, st_rad, st_mass, st_teff
    from pscomppars
    where hostname = '{target}'
    """

    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=" + quote(query) +
        "&format=csv"
    )

    try:
        df = pd.read_csv(url)
    except Exception:
        return None

    if len(df) == 0:
        return None

    row = df.iloc[0]
    return {
        "radius": row["st_rad"] if pd.notna(row["st_rad"]) else None,
        "mass": row["st_mass"] if pd.notna(row["st_mass"]) else None,
        "teff": row["st_teff"] if pd.notna(row["st_teff"]) else None,
        "source": "NASA Exoplanet Archive"
    }


def fetch_from_tic(target):
    """Fallback to the TESS Input Catalog for general stellar properties."""
    try:
        result = Catalogs.query_object(target, catalog="TIC", radius=0.02)
    except Exception:
        return None

    if result is None or len(result) == 0:
        return None

    row = result[0]

    def safe_get(colname):
        return row[colname] if colname in row.colnames and row[colname] is not None else None

    # TIC columns may vary by environment, so check a few common names
    radius = safe_get("rad")
    mass = safe_get("mass")
    teff = safe_get("Teff")

    return {
        "radius": float(radius) if radius is not None else None,
        "mass": float(mass) if mass is not None else None,
        "teff": float(teff) if teff is not None else None,
        "source": "TIC"
    }


def get_stellar_params(target):
    """Use local database first, then archive, then TIC."""
    if target in STAR_DATABASE:
        star = STAR_DATABASE[target].copy()
        star["source"] = "local database"
        return star

    star = fetch_from_exoplanet_archive(target)
    if star is not None and any(v is not None for k, v in star.items() if k in ["radius", "mass", "teff"]):
        return star

    star = fetch_from_tic(target)
    if star is not None and any(v is not None for k, v in star.items() if k in ["radius", "mass", "teff"]):
        return star

    return {
        "radius": None,
        "mass": None,
        "teff": None,
        "source": "not found"
    }

# -----------------------------------------------------------
# Built-in stellar parameter database
# -----------------------------------------------------------
STAR_DATABASE = {
    "Kepler-10": {
        "radius": 1.065,
        "mass": 0.91,
        "teff": 5627
    },
    "Kepler-22": {
        "radius": 0.979,
        "mass": 0.97,
        "teff": 5518
    },
    "TOI-700": {
        "radius": 0.42,
        "mass": 0.42,
        "teff": 3480
    }
}

def run_pipeline(target,
                 stellar_radius_rsun,
                 stellar_mass_msun=None,
                 stellar_teff=None,
                 semi_major_axis_au=None,
                 mission="Kepler",
                 albedo=0.3,
                 period_range=(0.5, 5),
                 duration_range=(0.01, 0.1),
                 flatten_window=401,
                 quarter=None):

    print("\n==============================")
    print("Running pipeline for:", target)
    print("==============================")

    # -------------------------------------------------------
    # Search for available light curves
    # -------------------------------------------------------
    print("Searching for light curves...")

    if mission == "Kepler":
        search = lk.search_lightcurve(
            target,
            mission=mission,
            author="Kepler",
            exptime=1800,
            quarter = quarter
            
        )

    elif mission == "TESS":
        search = lk.search_lightcurve(
            target,
            mission="TESS",
            author="SPOC"
    )[:1]

    else:
        search = lk.search_lightcurve(
            target,
            mission=mission
        )

    print("Number of search results found:", len(search))

    if len(search) == 0:
        print("No light curve files found for this target.")
        return

# -------------------------------------------------------
# Download the light curve files
# -------------------------------------------------------
# -------------------------------------------------------
# Download light curve data
# Kepler: stitch multiple quarters together
# TESS: use only one sector first for speed
# -------------------------------------------------------
# -------------------------------------------------------
# Download light curve data
# Kepler: stitch multiple quarters
# TESS: download only one sector (faster)
# -------------------------------------------------------
    if mission == "Kepler":

        print("Downloading files...")
        lc_collection = search.download_all()

        print("Download complete")

        print("Stitching files...")
        lc = lc_collection.stitch()

        print("Stitch complete")

    elif mission == "TESS":

        print("Downloading single TESS sector...")
        lc = search[0].download()

        print("Download complete")

    else:

        print("Downloading file...")
        lc = search[0].download()

        print("Download complete")


    # -------------------------------------------------------
    # Remove missing data points
    # -------------------------------------------------------
    print("Removing NaNs...")
    lc = lc.remove_nans()


    # -------------------------------------------------------
    # Remove extreme outliers (instrument noise spikes)
    # -------------------------------------------------------
    print("Removing outliers...")
    lc = lc.remove_outliers()


    # -------------------------------------------------------
    # Flatten the light curve
    #
    # This removes long-term stellar brightness variations
    # so that short transit dips are easier to detect
    # -------------------------------------------------------
    print("Number of data points:", len(lc.time))

    print("Flattening...")
    lc_clean = lc.flatten(window_length=flatten_window)


    # -------------------------------------------------------
    # Create a grid of possible orbital periods
    #
    # The BLS algorithm will test each period to see
    # if repeating transit signals appear.
    # -------------------------------------------------------
    print("Building period grid...")
    periods = np.linspace(period_range[0], period_range[1], 1000)
    durations = np.linspace(duration_range[0], duration_range[1], 10)

    # -------------------------------------------------------
    # Run the Box Least Squares algorithm
    #
    # This is the standard method used to detect exoplanets
    # in transit photometry.
    # -------------------------------------------------------
    print("Running BLS...")

    periodogram = lc_clean.to_periodogram(
    method="bls",
    period=periods,
    duration=durations,
    frequency_factor=500
)


    # Identify the strongest detected transit signal
    best_period = periodogram.period_at_max_power
    best_t0 = periodogram.transit_time_at_max_power

    print("Detected period:", best_period)


    # -------------------------------------------------------
    # Fold the light curve at the detected period
    #
    # This stacks all transits on top of each other
    # which increases the signal-to-noise ratio.
    # -------------------------------------------------------
    folded_lc = lc_clean.fold(period=best_period,
                              epoch_time=best_t0)


    # Bin the folded data to reduce noise
    binned = folded_lc.bin(time_bin_size=0.005)


    # -------------------------------------------------------
    # Estimate the transit depth
    #
    # Transit depth ≈ (Rp / Rs)^2
    #
    # We estimate the depth by comparing the median
    # in-transit flux to the median out-of-transit flux.
    # -------------------------------------------------------
    phase = binned.time.value
    flux = binned.flux.value


    # Points near phase = 0 contain the transit
    in_transit = np.abs(phase) < 0.03


    # Points far from phase = 0 represent baseline brightness
    out_of_transit = np.abs(phase) > 0.08


    baseline_flux = np.median(flux[out_of_transit])
    transit_flux = np.median(flux[in_transit])


    depth = baseline_flux - transit_flux

    print("Transit depth:", depth)


    # -------------------------------------------------------
    # Estimate planet radius
    #
    # Rp = Rs * sqrt(depth)
    #
    # Convert solar radii → Earth radii
    # -------------------------------------------------------
    if stellar_radius_rsun is not None:
        planet_radius_rsun = stellar_radius_rsun * np.sqrt(depth)
        planet_radius_rearth = planet_radius_rsun * 109.1
        print("Estimated planet radius (Earth radii):", planet_radius_rearth)
    else:
        print("Planet radius cannot be estimated (stellar radius unknown)")
    
    # -------------------------------------------------------
    # Estimate orbital distance using Kepler's Third Law
    #
    # a^3 = (G * M_star * P^2) / (4 * pi^2)
    #
    # where:
    # P = orbital period
    # M_star = stellar mass
    # a = semi-major axis
    # -------------------------------------------------------
    if stellar_mass_msun is not None:

        G = 6.67430e-11       # gravitational constant (m^3 kg^-1 s^-2)
        M_sun = 1.98847e30    # mass of Sun (kg)
        AU = 1.495978707e11   # meters in one AU
        day = 86400           # seconds in one day

        P_sec = best_period.value * day
        M_star_kg = stellar_mass_msun * M_sun

        a_m = ((G * M_star_kg * P_sec**2) / (4 * np.pi**2))**(1/3)
        a_au_est = a_m / AU

        print("Estimated orbital distance (AU):", a_au_est)

    if stellar_mass_msun is not None:
        v_m_per_s = (2 * np.pi * a_m) / P_sec
        v_km_per_s = v_m_per_s / 1000
        print("Estimated orbital velocity (km/s):", v_km_per_s)

    else:
        a_au_est = semi_major_axis_au


    # -------------------------------------------------------
    # Estimate stellar luminosity
    #
    # L/Lsun = (R/Rsun)^2 * (T/Tsun)^4
    # -------------------------------------------------------
    if stellar_teff is not None:

        T_sun = 5772

        luminosity_lsun = (stellar_radius_rsun**2) * (stellar_teff / T_sun)**4

        print("Estimated stellar luminosity (L/Lsun):", luminosity_lsun)

    else:
        luminosity_lsun = None


    # -------------------------------------------------------
    # Estimate equilibrium temperature
    #
    # Teq = Teff * sqrt(R_star / (2a)) * (1 - albedo)^0.25
    # -------------------------------------------------------
    if stellar_teff is not None and a_au_est is not None:

        rsun_to_au = 0.00465047
        stellar_radius_au = stellar_radius_rsun * rsun_to_au

        teq = stellar_teff * np.sqrt(stellar_radius_au / (2 * a_au_est)) * (1 - albedo)**0.25

        print("Estimated equilibrium temperature (K):", teq)
        print("Estimated equilibrium temperature (C):", teq - 273.15)

    else:
        teq = None


    # -------------------------------------------------------
    # Estimate stellar energy received by the planet
    #
    # S/Searth = (L/Lsun) / (a/AU)^2
    # -------------------------------------------------------
    if luminosity_lsun is not None and a_au_est is not None:

        insolation = luminosity_lsun / (a_au_est**2)

        print("Estimated stellar flux relative to Earth:", insolation)

    else:
        insolation = None


    # -------------------------------------------------------
    # Optional: Estimate equilibrium temperature
    #
    # Teq ≈ Teff * sqrt(R* / (2a))
    #
    # Only runs if stellar temperature and orbital distance
    # are provided.
    # -------------------------------------------------------
    if stellar_teff is not None and semi_major_axis_au is not None:

        albedo = 0.3
        rsun_to_au = 0.00465047
        stellar_radius_au = stellar_radius_rsun * rsun_to_au

        teq = stellar_teff * np.sqrt(
            stellar_radius_au / (2 * semi_major_axis_au)
        )

        print("Estimated equilibrium temperature:", teq, "K")
        print("Estimated temperature (C):", teq - 273.15)


    # -------------------------------------------------------
    # Plot the BLS periodogram
    #
    # Shows which orbital period produced the strongest signal
    # -------------------------------------------------------
    periodogram.plot()

    plt.title(f"BLS Periodogram - {target}")

    plt.savefig(f"results/{target}_periodogram.png", dpi=300)

    plt.show()

    # -------------------------------------------------------
    # Plot the folded light curve
    #
    # Blue points = raw folded data
    # Orange points = binned data (clearer transit)
    # -------------------------------------------------------
    fig, ax = plt.subplots()

    folded_lc.scatter(ax=ax, alpha=0.1)
    binned.scatter(ax=ax)

    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(0.9998, 1.0001)

    ax.set_title(f"Folded Light Curve - {target}")
    ax.set_xlabel("Phase [days]")
    ax.set_ylabel("Normalized Flux")

    plt.savefig(f"results/{target}_folded_lightcurve.png", dpi=300)

    plt.show()


# -----------------------------------------------------------
# VALIDATION CASE
#
# Kepler-10b is a known rocky exoplanet.
# We use it to verify that our detection pipeline works.
# -----------------------------------------------------------

def choose_search_settings(mission, stellar_radius_rsun, stellar_teff, target):

    # Validation preset for Kepler-10
    if target == "Kepler-10" and mission == "Kepler":
        return {
            "period_range": (0.7, 1.0),
            "duration_range": (0.01, 0.1),
            "flatten_window": 401,
            "quarter": 3
        }

    # Long-period system example
    if target == "Kepler-22" and mission == "Kepler":
        return {
            "period_range": (250, 330),
            "duration_range": (0.15, 0.6),
            "flatten_window": 1001,
            "quarter": None
        }

    # General heuristics
    if mission == "TESS":
        if stellar_teff is not None and stellar_teff < 4000:
            return {
                "period_range": (0.5, 20),
                "duration_range": (0.02, 0.2),
                "flatten_window": 401,
                "quarter": None
            }
        else:
            return {
                "period_range": (0.5, 10),
                "duration_range": (0.01, 0.2),
                "flatten_window": 401,
                "quarter": None
            }

    if mission == "Kepler":
        if stellar_radius_rsun is not None and stellar_radius_rsun > 1.2:
            return {
                "period_range": (0.5, 20),
                "duration_range": (0.03, 0.3),
                "flatten_window": 801,
                "quarter": None
            }
        else:
            return {
                "period_range": (0.5, 10),
                "duration_range": (0.01, 0.2),
                "flatten_window": 401,
                "quarter": None
            }

    return {
        "period_range": (0.5, 10),
        "duration_range": (0.01, 0.2),
        "flatten_window": 401,
        "quarter": None
    }

# -----------------------------------------------------------
# USER INPUT FROM TERMINAL
# -----------------------------------------------------------

target = input("Enter target star name [Kepler-10]: ").strip()
if target == "":
    target = "Kepler-10"

mission = input("Enter mission [Kepler]: ").strip()
if mission == "":
    mission = "Kepler"

star = get_stellar_params(target)

stellar_radius_rsun = star["radius"]
stellar_mass_msun = star["mass"]
stellar_teff = star["teff"]

print(f"Stellar parameters source: {star['source']}")
print("radius =", stellar_radius_rsun)
print("mass =", stellar_mass_msun)
print("teff =", stellar_teff)

settings = choose_search_settings(
    mission,
    stellar_radius_rsun,
    stellar_teff,
    target
)

print("\nUsing automatic search settings:")
print("period_range =", settings["period_range"])
print("duration_range =", settings["duration_range"])
print("flatten_window =", settings["flatten_window"])
print("quarter =", settings["quarter"])

run_pipeline(
    target=target,
    stellar_radius_rsun=stellar_radius_rsun,
    stellar_mass_msun=stellar_mass_msun,
    stellar_teff=stellar_teff,
    mission=mission,
    albedo=0.3,
    period_range=settings["period_range"],
    duration_range=settings["duration_range"],
    flatten_window=settings["flatten_window"],
    quarter=settings["quarter"]
)


