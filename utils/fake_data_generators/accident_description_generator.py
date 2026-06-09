import csv
from pathlib import Path

import random

from utils.accident_text_constants import (
    DETAILS_MAJOR,
    DETAILS_MINOR,
    FRONT_COLLISION_TEMPLATES,
    MAJOR_AIRBAGS,
    MINOR_AIRBAGS,
    PARKED_AIRBAGS,
    PARKED_CAR_TEMPLATES,
    POLICE_INFO,
    REAR_COLLISION_TEMPLATES,
    SIDE_COLLISION_TEMPLATES,
    SPEED_DESCRIPTIONS,
    STYLE_OPTIONS,
    TEMPLATES_BY_STYLE_COLLISION,
    TEMPLATES_BY_STYLE_PARKED,
    TEMPLATES_BY_STYLE_THEFT,
    THEFT_AIRBAGS,
    THEFT_DETAILS,
    TIME_CONDITIONS,
    TOW_INFO,
    VEHICLE_THEFT_TEMPLATES,
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

    if case_type == "Vehicle Theft":
        damage_count = random.randint(2, 3)
        damage = ", ".join(
            random.sample(
                THEFT_DETAILS,
                min(damage_count, len(THEFT_DETAILS))
            )
        )

        airbags = random.choice(THEFT_AIRBAGS)

    elif case_type == "Parked Car":

        damage_count = random.randint(2, 4)
        damage = ", ".join(
            random.sample(
                DETAILS_MINOR,
                min(damage_count, len(DETAILS_MINOR))
            )
        )

        airbags = random.choice(PARKED_AIRBAGS)

    elif severity in ['Major Damage', 'Total Loss']:

        damage_count = random.randint(3, 5)

        damage = ", ".join(
            random.sample(
                DETAILS_MAJOR + DETAILS_MINOR,
                min(damage_count, len(DETAILS_MAJOR + DETAILS_MINOR))
            )
        )

        airbags = random.choice(MAJOR_AIRBAGS)

    else:

        damage_count = random.randint(2, 4)

        damage = ", ".join(
            random.sample(
                DETAILS_MINOR,
                min(damage_count, len(DETAILS_MINOR))
            )
        )

        airbags = random.choice(MINOR_AIRBAGS)

    # Auswahl der Unfallart

    if case_type == "Vehicle Theft":
        accident_type = random.choice(VEHICLE_THEFT_TEMPLATES)

    elif case_type == "Parked Car":
        accident_type = random.choice(PARKED_CAR_TEMPLATES)

    elif case_type == "Front Collision":
        accident_type = random.choice(FRONT_COLLISION_TEMPLATES)

    elif case_type == "Rear Collision":
        accident_type = random.choice(REAR_COLLISION_TEMPLATES)

    else:
        accident_type = random.choice(SIDE_COLLISION_TEMPLATES)

    # Zufallsparameter

    weather = random.choice(WEATHER_CONDITIONS)
    time_of_day = random.choice(TIME_CONDITIONS)
    time_of_day_cap = time_of_day.capitalize()
    speed = random.choice(SPEED_DESCRIPTIONS)

    witness = random.choice(WITNESS_INFO)
    police = random.choice(POLICE_INFO)
    tow = random.choice(TOW_INFO)

    style = random.choice(STYLE_OPTIONS)

    if case_type == "Vehicle Theft":
        templates = TEMPLATES_BY_STYLE_THEFT[style]
    elif case_type == "Parked Car":
        templates = TEMPLATES_BY_STYLE_PARKED[style]
    else:
        templates = TEMPLATES_BY_STYLE_COLLISION[style]

    text = random.choice(templates).format(
        make=make,
        model=model,
        year=year,
        accident_type=accident_type,
        damage=damage,
        airbags=airbags,
        weather=weather,
        time_of_day=time_of_day,
        time_of_day_cap=time_of_day_cap,
        speed=speed,
        witness=witness,
        police=police,
        tow=tow
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

