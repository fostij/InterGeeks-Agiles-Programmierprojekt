import random
import re

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

def generate_accident_description(row: dict) -> str:
    make, model = row.get("auto_make", "").strip(), row.get("auto_model", "").strip()
    year, severity = row["auto_year"], row["incident_severity"]
    
    case_type = _resolve_case_type(row)
    config = CASE_CONFIG[case_type]
    severity_bucket = "major" if severity in MAJOR_SEVERITIES else "minor"
    
    detail_level = random.choices(DETAIL_LEVELS, weights=DETAIL_LEVEL_WEIGHTS, k=1)[0]
    context_vars = _generate_context_fields(config, severity_bucket, detail_level)
    
    damage = _generate_damage_string(config["damage"][severity_bucket])
    incident_phrase = random.choice(config["incident_phrases"])
    
    strategy = _determine_strategy(make, model, detail_level)
    core_parts = _build_core_sentences(strategy, make, model, year, incident_phrase, damage)
    
    core_parts = _inject_context_and_extras(core_parts, context_vars)
    
    text = " ".join(" ".join(core_parts).split())
    
    return text

def get_labels(target_fields: list[str], row: dict) -> dict:
    return {field: row.get(field, "") for field in target_fields}

def _resolve_case_type(row: dict) -> str:
    incident_type = str(row.get("incident_type", "")).strip()
    collision_type = str(row.get("collision_type", "")).strip()

    for value in (collision_type, incident_type):
        if value in VALID_CASE_TYPES:
            return value

    return "Side Collision" #TODO: maybe add some randomness here instead of defaulting to "Side Collision"?

def _generate_context_fields(config: dict, severity_bucket: str, detail_level: str) -> dict:
    ctx = {k: "" for k in ["airbags", "weather", "time_of_day", "time_of_day_cap", "witness", "police", "tow"]}
    
    if detail_level == "minimal":
        return ctx
        
    if detail_level == "normal":
        ctx["airbags"] = random.choice(config["airbags"][severity_bucket])
        return ctx
        
    if detail_level == "detailed":
        ctx["airbags"] = random.choice(config["airbags"][severity_bucket])
        ctx["weather"] = random.choice(WEATHER_CONDITIONS)
        ctx["time_of_day"] = random.choice(TIME_CONDITIONS)
        ctx["time_of_day_cap"] = ctx["time_of_day"].capitalize()
        ctx["witness"] = random.choice(WITNESS_INFO)
        ctx["police"] = random.choice(POLICE_INFO)
        ctx["tow"] = random.choice(TOW_INFO)
        return ctx
        
    if detail_level == "noisy":
        if random.random() < 0.65: ctx["airbags"] = random.choice(config["airbags"][severity_bucket])
        if random.random() < 0.60: ctx["weather"] = random.choice(WEATHER_CONDITIONS)
        if random.random() < 0.60: 
            ctx["time_of_day"] = random.choice(TIME_CONDITIONS)
            ctx["time_of_day_cap"] = ctx["time_of_day"].capitalize()
        if random.random() < 0.35: ctx["police"] = random.choice(POLICE_INFO)
        if random.random() < 0.25: ctx["tow"] = random.choice(TOW_INFO)
        return ctx
    return ctx

def _generate_damage_string(damage_cfg: dict) -> str:
    damage_count = random.randint(*damage_cfg["count"])
    sampled = random.sample(damage_cfg["pool"], min(damage_count, len(damage_cfg["pool"])))
    return ", ".join(sampled)

def _determine_strategy(make: str, model: str, detail_level: str) -> str:
    if not (make or model) or (detail_level == "noisy" and random.random() < 0.40):
        return "no_vehicle"
        
    return random.choices(
        ["vehicle_in_event", "vehicle_then_event", "event_then_vehicle", "distant"],
        weights=[0.28, 0.27, 0.25, 0.20],
        k=1,
    )[0]


def _build_core_sentences(strategy: str, make: str, model: str, year: int, incident_phrase: str, damage: str) -> list[str]:
    damage_sentence = random.choice(DAMAGE_INTRO_TEMPLATES).format(damage=damage)
    event_sentence = random.choice(EVENT_STANDALONE_TEMPLATES).format(incident_phrase=incident_phrase)
    
    if strategy == "no_vehicle":
        core_parts = [event_sentence, damage_sentence]
        if random.random() < 0.5:
            random.shuffle(core_parts)
        return core_parts

    with_year = random.random() > 0.25

    if strategy == "vehicle_in_event":
        vehicle_ref = _build_vehicle_ref(make, model, year, with_year=with_year)
        combined = random.choice(VEHICLE_IN_EVENT_TEMPLATES).format(
            vehicle_ref=vehicle_ref, incident_phrase=incident_phrase
        )
        return [combined, damage_sentence]

    if strategy == "vehicle_then_event":
        vehicle_ref = _build_vehicle_ref(make, model, year, with_year=with_year)
        vehicle_sentence = random.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=vehicle_ref)
        return [vehicle_sentence, event_sentence, damage_sentence]

    if strategy == "event_then_vehicle":
        vehicle_ref = _build_vehicle_ref(make, model, year, with_year=with_year)
        vehicle_sentence = random.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=vehicle_ref)
        if random.random() < 0.35:
            return [event_sentence, damage_sentence, vehicle_sentence]
        return [event_sentence, vehicle_sentence, damage_sentence]

    if strategy == "distant":
        make_model_ref = _build_vehicle_ref(make, model, None, with_year=False)
        vehicle_sentence = random.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=make_model_ref)
        year_sentence = random.choice(YEAR_STANDALONE_TEMPLATES).format(year=year)
        core_parts = [vehicle_sentence, year_sentence, event_sentence, damage_sentence]
        random.shuffle(core_parts)
        return core_parts
        
    return [event_sentence, damage_sentence]

def _build_vehicle_ref(make: str, model: str, year, *, with_year: bool) -> str:
    if with_year and year:
        return random.choice(VEHICLE_REF_WITH_YEAR).format(make=make, model=model, year=year)
    return random.choice(VEHICLE_REF_WITHOUT_YEAR).format(make=make, model=model)

def _inject_context_and_extras(core_parts: list[str], ctx: dict) -> list[str]:
    if ctx["time_of_day"] or ctx["weather"]:
        context_sentence = _build_context_sentence(ctx["time_of_day"], ctx["time_of_day_cap"], ctx["weather"])
        if context_sentence:
            insert_pos = random.randint(0, len(core_parts))
            core_parts.insert(insert_pos, context_sentence)
            
    for extra in [ctx["airbags"], ctx["witness"], ctx["police"], ctx["tow"]]:
        if extra:
            core_parts.append(extra)
            
    return core_parts

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