import random

from utils.fake_data_generators.accident_text_constants import (
    CASE_CONFIG,
    DAMAGE_INTRO_TEMPLATES,
    DETAIL_LEVELS,
    DETAIL_LEVEL_WEIGHTS,
    EVENT_STANDALONE_TEMPLATES,
    MAJOR_SEVERITIES,
    POLICE_INFO,
    TIME_CONDITIONS,
    TOW_INFO,
    VEHICLE_IN_EVENT_TEMPLATES,
    VEHICLE_INTRO_TEMPLATES,
    VEHICLE_REF_WITH_YEAR,
    VEHICLE_REF_WITHOUT_YEAR,
    WEATHER_CONDITIONS,
    WITNESS_INFO,
    YEAR_STANDALONE_TEMPLATES,
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

def _build_vehicle_ref(make: str, model: str, year, *, with_year: bool) -> str:
    if with_year and year:
        return random.choice(VEHICLE_REF_WITH_YEAR).format(make=make, model=model, year=year)
    return random.choice(VEHICLE_REF_WITHOUT_YEAR).format(make=make, model=model)


def _build_context_sentence(time_of_day: str, time_of_day_cap: str, weather: str) -> str:
    if time_of_day and weather:
        return random.choice([
            f"{time_of_day_cap} {weather}.",
            f"Das Ereignis trat {time_of_day} {weather} ein.",
            f"Der Vorfall ereignete sich {time_of_day} {weather}.",
            f"Bedingungen: {time_of_day}, {weather}.",
        ])
    if time_of_day:
        return random.choice([
            f"Zeitpunkt: {time_of_day}.",
            f"Das Ereignis trat {time_of_day} ein.",
            f"{time_of_day_cap} kam es dazu.",
        ])
    if weather:
        return random.choice([
            f"Wetterlage: {weather}.",
            f"Wetterbedingungen: {weather}.",
        ])
    return ""


def _find_spans(text: str, value: str, label: str) -> list[tuple[int, int, str]]:
    """Return all non-overlapping character-level spans of *value* in *text*."""
    if not value:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    while True:
        pos = text.find(value, start)
        if pos == -1:
            break
        spans.append((pos, pos + len(value), label))
        start = pos + len(value)
    return spans


def generate_accident_description(row: dict) -> tuple[str, dict]:
    vehicle = str(row.get("auto_make_model", "")).strip()
    if vehicle:
        make, _, model = vehicle.partition(" ")
        model = model.strip()
    else:
        make = str(row.get("auto_make", "")).strip()
        model = str(row.get("auto_model", "")).strip()

    year = row["auto_year"]
    severity = row["incident_severity"]
    case_type = resolve_case_type(row)

    config = CASE_CONFIG[case_type]
    severity_bucket = "major" if severity in MAJOR_SEVERITIES else "minor"
    detail_level = random.choices(DETAIL_LEVELS, weights=DETAIL_LEVEL_WEIGHTS, k=1)[0]

    # ── damage ────────────────────────────────────────────────────────────────
    damage_cfg = config["damage"][severity_bucket]
    damage_count = random.randint(*damage_cfg["count"])
    damage = ", ".join(
        random.sample(damage_cfg["pool"], min(damage_count, len(damage_cfg["pool"])))
    )

    incident_phrase = random.choice(config["incident_phrases"])

    # ── contextual fields ─────────────────────────────────────────────────────
    if detail_level == "minimal":
        airbags = time_of_day = time_of_day_cap = weather = witness = police = tow = ""

    elif detail_level == "normal":
        airbags = random.choice(config["airbags"][severity_bucket])
        time_of_day = time_of_day_cap = weather = witness = police = tow = ""

    elif detail_level == "detailed":
        airbags = random.choice(config["airbags"][severity_bucket])
        weather = random.choice(WEATHER_CONDITIONS)
        time_of_day = random.choice(TIME_CONDITIONS)
        time_of_day_cap = time_of_day.capitalize()
        witness = random.choice(WITNESS_INFO)
        police = random.choice(POLICE_INFO)
        tow = random.choice(TOW_INFO)

    else:  # noisy
        airbags = random.choice(config["airbags"][severity_bucket]) if random.random() < 0.65 else ""
        weather = random.choice(WEATHER_CONDITIONS) if random.random() < 0.60 else ""
        time_of_day = random.choice(TIME_CONDITIONS) if random.random() < 0.60 else ""
        time_of_day_cap = time_of_day.capitalize() if time_of_day else ""
        witness = ""
        police = random.choice(POLICE_INFO) if random.random() < 0.35 else ""
        tow = random.choice(TOW_INFO) if random.random() < 0.25 else ""

    has_vehicle = bool(make or model)

    # ── pick assembly strategy ────────────────────────────────────────────────
    # distant          – year in its own sentence, separated from make/model
    # vehicle_in_event – vehicle ref embedded inside the event sentence
    # vehicle_then_event – vehicle → event → damage
    # event_then_vehicle – event → vehicle (maybe) → damage
    # no_vehicle       – omit vehicle info entirely
    if not has_vehicle or (detail_level == "noisy" and random.random() < 0.40):
        strategy = "no_vehicle"
    else:
        strategy = random.choices(
            ["vehicle_in_event", "vehicle_then_event", "event_then_vehicle", "distant"],
            weights=[0.28, 0.27, 0.25, 0.20],
            k=1,
        )[0]

    # ── build core sentence components ────────────────────────────────────────
    damage_sentence = random.choice(DAMAGE_INTRO_TEMPLATES).format(damage=damage)
    event_sentence = random.choice(EVENT_STANDALONE_TEMPLATES).format(incident_phrase=incident_phrase)

    if strategy == "vehicle_in_event":
        vehicle_ref = _build_vehicle_ref(make, model, year, with_year=random.random() > 0.25)
        combined = random.choice(VEHICLE_IN_EVENT_TEMPLATES).format(
            vehicle_ref=vehicle_ref, incident_phrase=incident_phrase
        )
        core_parts = [combined, damage_sentence]

    elif strategy == "vehicle_then_event":
        vehicle_ref = _build_vehicle_ref(make, model, year, with_year=random.random() > 0.25)
        vehicle_sentence = random.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=vehicle_ref)
        core_parts = [vehicle_sentence, event_sentence, damage_sentence]

    elif strategy == "event_then_vehicle":
        vehicle_ref = _build_vehicle_ref(make, model, year, with_year=random.random() > 0.25)
        vehicle_sentence = random.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=vehicle_ref)
        # Occasionally put damage before the vehicle mention
        if random.random() < 0.35:
            core_parts = [event_sentence, damage_sentence, vehicle_sentence]
        else:
            core_parts = [event_sentence, vehicle_sentence, damage_sentence]

    elif strategy == "distant":
        # Make/model and year appear in separate sentences, forcing
        # the model to learn distant entity dependencies
        make_model_ref = _build_vehicle_ref(make, model, None, with_year=False)
        vehicle_sentence = random.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=make_model_ref)
        year_sentence = random.choice(YEAR_STANDALONE_TEMPLATES).format(year=year)
        core_parts = [vehicle_sentence, year_sentence, event_sentence, damage_sentence]
        random.shuffle(core_parts)

    else:  # no_vehicle
        core_parts = [event_sentence, damage_sentence]
        if random.random() < 0.5:
            random.shuffle(core_parts)

    # ── inject context at a random position ──────────────────────────────────
    if time_of_day or weather:
        ctx = _build_context_sentence(time_of_day, time_of_day_cap, weather)
        if ctx:
            insert_pos = random.randint(0, len(core_parts))
            core_parts.insert(insert_pos, ctx)

    # ── append trailing info ──────────────────────────────────────────────────
    for extra in filter(None, [airbags, witness, police, tow]):
        core_parts.append(extra)

    text = " ".join(" ".join(core_parts).split())

    # ── build entity spans via post-hoc matching ──────────────────────────────
    entities: list[tuple[int, int, str]] = []
    # Search longest tokens first so shorter substrings don't shadow full matches
    entity_lookup = sorted(
        [(str(year), "YEAR"), (make, "MAKE"), (model, "MODEL")],
        key=lambda t: len(t[0]),
        reverse=True,
    )
    occupied: set[int] = set()
    for value, label in entity_lookup:
        if not value:
            continue
        for span in _find_spans(text, value, label):
            span_range = set(range(span[0], span[1]))
            if not span_range & occupied:
                entities.append(span)
                occupied |= span_range

    entities.sort(key=lambda s: s[0])
    return text, {"entities": entities}
