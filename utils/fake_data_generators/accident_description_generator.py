import csv
from pathlib import Path

import random

from utils.fake_data_generators.accident_text_constants import (
    CASE_CONFIG,
    MAJOR_SEVERITIES,
    POLICE_INFO,
    SPEED_DESCRIPTIONS,
    STYLE_OPTIONS,
    TIME_CONDITIONS,
    TOW_INFO,
    WEATHER_CONDITIONS,
    WITNESS_INFO,
)


VALID_CASE_TYPES = {
    "Front Collision",
    "Parked Car",
    "Rear Collision",
    "Side Collision",
    "Vehicle Theft",
}

def resolve_case_type(row: dict) -> str:
    incident_type = str(row.get("incident_type", "")).strip()
    collision_type = str(row.get("collision_type", "")).strip()

    for value in (collision_type, incident_type):
        if value in VALID_CASE_TYPES:
            return value

    return "Side Collision"

def generate_accident_description(row: dict) -> str:

    vehicle = str(row.get('auto_make_model', '')).strip()
    if vehicle:
        make, _, model = vehicle.partition(' ')
        model = model.strip()
    else:
        make = str(row.get('auto_make', '')).strip()
        model = str(row.get('auto_model', '')).strip()
    year = row['auto_year']
    severity = row['incident_severity']
    case_type = resolve_case_type(row)

    config = CASE_CONFIG[case_type]
    severity_bucket = "major" if severity in MAJOR_SEVERITIES else "minor"

    damage_cfg = config["damage"][severity_bucket]
    damage_count = random.randint(*damage_cfg["count"])
    damage = ", ".join(
        random.sample(damage_cfg["pool"], min(damage_count, len(damage_cfg["pool"])))
    )

    airbags = random.choice(config["airbags"][severity_bucket])
    incident_phrase = random.choice(config["incident_phrases"])

    weather = random.choice(WEATHER_CONDITIONS)
    time_of_day = random.choice(TIME_CONDITIONS)
    time_of_day_cap = time_of_day.capitalize()
    speed = random.choice(SPEED_DESCRIPTIONS)

    witness = random.choice(WITNESS_INFO)
    police = random.choice(POLICE_INFO)
    tow = random.choice(TOW_INFO)

    style = random.choice(STYLE_OPTIONS)
    templates = config["templates"]

    text = random.choice(templates[style]).format(
        make=make,
        model=model,
        year=year,
        incident_phrase=incident_phrase,
        damage=damage,
        airbags=airbags,
        weather=weather,
        time_of_day=time_of_day,
        time_of_day_cap=time_of_day_cap,
        speed=speed,
        witness=witness,
        police=police,
        tow=tow,
    )

    return " ".join(text.split())


def add_generated_text_column(input_path: Path, output_path: Path, column_name: str) -> int:
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if column_name not in fieldnames:
        fieldnames.append(column_name)

    for row in rows:
        row[column_name] = generate_accident_description(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)