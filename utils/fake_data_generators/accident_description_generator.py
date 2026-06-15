import random
import pandas as pd
from dataclasses import dataclass
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

@dataclass
class GenerationContext:
    make: str
    model: str
    year: int
    case_type: str
    severity_bucket: str
    detail_level: str
    strategy: str
    with_year: bool
    context_vars: dict
    incident_phrase: str
    damage_string: str

class AccidentDataGenerator:
    def __init__(self, target_fields: list[str]):
        self.target_fields = target_fields

    def generate_text(self, row: dict, seed: int = None) -> str:
        text, _ = self._generate(row, seed)
        return text

    def generate_with_labels(self, row: dict, seed: int = None) -> tuple[str, dict]:
        return self._generate(row, seed)
    
    def _generate(self, row: dict, seed: int = None) -> tuple[str, dict]:
        rng = random.Random(seed) if seed is not None else random.Random()

        ctx = self._create_generation_context(row, rng)

        core_parts = self._build_core_sentences(ctx, rng)
        core_parts = self._inject_context_and_extras(core_parts, ctx.context_vars, rng)
        
        labels = self._get_labels(ctx, row)

        text = " ".join(" ".join(core_parts).split())
    
        return text, labels

    def _create_generation_context(self, row: dict, rng: random.Random) -> GenerationContext:
        make = row.get("auto_make", "").strip()
        model = row.get("auto_model", "").strip()
        year = row["auto_year"]
        
        case_type = self._resolve_case_type(row)
        severity = row["incident_severity"]
        severity_bucket = "major" if severity in MAJOR_SEVERITIES else "minor"
        
        detail_level = rng.choices(DETAIL_LEVELS, weights=DETAIL_LEVEL_WEIGHTS, k=1)[0]
        
        config = CASE_CONFIG[case_type]
        context_vars = self._generate_context_fields(config, severity_bucket, detail_level, rng)
        
        incident_phrase = rng.choice(config["incident_phrases"])
        damage_string = self._generate_damage_string(config["damage"][severity_bucket])
        
        strategy = self._determine_strategy(make, model, detail_level, rng)
        with_year = strategy != "no_vehicle" and rng.random() > 0.25
        
        return GenerationContext(
            make=make,
            model=model,
            year=year,
            case_type=case_type,
            severity_bucket=severity_bucket,
            detail_level=detail_level,
            strategy=strategy,
            with_year=with_year,
            context_vars=context_vars,
            incident_phrase=incident_phrase,
            damage_string=damage_string
        )
            
    def _get_labels(self, ctx: GenerationContext, row: dict) -> dict:
        labels = {}
        
        vehicle_present = (ctx.strategy != "no_vehicle")
        year_present = vehicle_present and ctx.with_year
        
        for field in self.target_fields:
            val = row.get(field, "")
            
            if val == "" or pd.isna(val) or val == "?":
                labels[field] = -100 if field in ["auto_year", "number_of_vehicles_involved", "witnesses"] else None
                continue

            if field in ["auto_make", "auto_model"] and not vehicle_present:
                labels[field] = None
            elif field == "auto_year" and not year_present:
                labels[field] = -100
            elif field == "weather" and ctx.context_vars.get("weather") == "":
                labels[field] = None
            elif field == "witnesses" and ctx.context_vars.get("witness") == "":
                labels[field] = -100
            elif field == "authorities_contacted" and ctx.context_vars.get("police") == "":
                labels[field] = None
            else:
                labels[field] = val
                
        return labels

    def _resolve_case_type(self, row: dict) -> str:
        incident_type = str(row.get("incident_type", "")).strip()
        collision_type = str(row.get("collision_type", "")).strip()

        for value in (collision_type, incident_type):
            if value in VALID_CASE_TYPES:
                return value

        return "Side Collision" #TODO: maybe add some randomness here instead of defaulting to "Side Collision"?

    def _generate_context_fields(self, config: dict, severity_bucket: str, detail_level: str, rng: random.Random) -> dict:
        ctx = {k: "" for k in ["airbags", "weather", "time_of_day", "time_of_day_cap", "witness", "police", "tow"]}
        
        if detail_level == "minimal":
            return ctx
            
        if detail_level == "normal":
            ctx["airbags"] = rng.choice(config["airbags"][severity_bucket])
            return ctx
            
        if detail_level == "detailed":
            ctx["airbags"] = rng.choice(config["airbags"][severity_bucket])
            ctx["weather"] = rng.choice(WEATHER_CONDITIONS)
            ctx["time_of_day"] = rng.choice(TIME_CONDITIONS)
            ctx["time_of_day_cap"] = ctx["time_of_day"].capitalize()
            ctx["witness"] = rng.choice(WITNESS_INFO)
            ctx["police"] = rng.choice(POLICE_INFO)
            ctx["tow"] = rng.choice(TOW_INFO)
            return ctx
            
        if detail_level == "noisy":
            if rng.random() < 0.65: ctx["airbags"] = rng.choice(config["airbags"][severity_bucket])
            if rng.random() < 0.60: ctx["weather"] = rng.choice(WEATHER_CONDITIONS)
            if rng.random() < 0.60: 
                ctx["time_of_day"] = rng.choice(TIME_CONDITIONS)
                ctx["time_of_day_cap"] = ctx["time_of_day"].capitalize()
            if rng.random() < 0.35: ctx["police"] = rng.choice(POLICE_INFO)
            if rng.random() < 0.25: ctx["tow"] = rng.choice(TOW_INFO)
            return ctx
        return ctx

    def _generate_damage_string(self, damage_cfg: dict, rng: random.Random) -> str:
        damage_count = rng.randint(*damage_cfg["count"])
        sampled = rng.sample(damage_cfg["pool"], min(damage_count, len(damage_cfg["pool"])))
        return ", ".join(sampled)

    def _determine_strategy(self, make: str, model: str, detail_level: str, rng: random.Random) -> str:
        if not (make or model) or (detail_level == "noisy" and rng.random() < 0.40):
            return "no_vehicle"
            
        return rng.choices(
            ["vehicle_in_event", "vehicle_then_event", "event_then_vehicle", "distant"],
            weights=[0.28, 0.27, 0.25, 0.20],
            k=1,
        )[0]

    def _build_core_sentences(self, ctx: GenerationContext, rng: random.Random) -> list[str]:
        damage_sentence = rng.choice(DAMAGE_INTRO_TEMPLATES).format(damage=ctx.damage_string)
        event_sentence = rng.choice(EVENT_STANDALONE_TEMPLATES).format(incident_phrase=ctx.incident_phrase)
        
        if ctx.strategy == "no_vehicle":
            core_parts = [event_sentence, damage_sentence]
            if rng.random() < 0.5:
                rng.shuffle(core_parts)
            return core_parts

        if ctx.strategy == "vehicle_in_event":
            vehicle_ref = self._build_vehicle_ref(ctx, rng)
            combined = rng.choice(VEHICLE_IN_EVENT_TEMPLATES).format(
                vehicle_ref=vehicle_ref, incident_phrase=ctx.incident_phrase
            )
            return [combined, damage_sentence]

        if ctx.strategy == "vehicle_then_event":
            vehicle_ref = self._build_vehicle_ref(ctx, rng)
            vehicle_sentence = rng.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=vehicle_ref)
            return [vehicle_sentence, event_sentence, damage_sentence]

        if ctx.strategy == "event_then_vehicle":
            vehicle_ref = self._build_vehicle_ref(ctx, rng)
            vehicle_sentence = rng.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=vehicle_ref)
            if rng.random() < 0.35:
                return [event_sentence, damage_sentence, vehicle_sentence]
            return [event_sentence, vehicle_sentence, damage_sentence]

        if ctx.strategy == "distant":
            vehicle_ref = rng.choice(VEHICLE_REF_WITHOUT_YEAR).format(make=ctx.make, model=ctx.model)
            vehicle_sentence = rng.choice(VEHICLE_INTRO_TEMPLATES).format(vehicle_ref=vehicle_ref)

            core_parts = [vehicle_sentence, event_sentence, damage_sentence]

            if ctx.with_year and ctx.year:
                year_sentence = rng.choice(YEAR_STANDALONE_TEMPLATES).format(year=ctx.year)
                core_parts.append(year_sentence)
                
            rng.shuffle(core_parts)
            return core_parts
            
        return [event_sentence, damage_sentence]

    def _build_vehicle_ref(self, ctx: GenerationContext, rng: random.Random) -> str:
        if ctx.with_year and ctx.year:
            return rng.choice(VEHICLE_REF_WITH_YEAR).format(make=ctx.make, model=ctx.model, year=ctx.year)
        return rng.choice(VEHICLE_REF_WITHOUT_YEAR).format(make=ctx.make, model=ctx.model)

    def _inject_context_and_extras(self, core_parts: list[str], context_vars: dict, rng: random.Random) -> list[str]:
        if context_vars["time_of_day"] or context_vars["weather"]:
            context_sentence = self._build_context_sentence(context_vars["time_of_day"], context_vars["time_of_day_cap"], context_vars["weather"], rng)
            if context_sentence:
                insert_pos = rng.randint(0, len(core_parts))
                core_parts.insert(insert_pos, context_sentence)
                
        for extra in [context_vars["airbags"], context_vars["witness"], context_vars["police"], context_vars["tow"]]:
            if extra:
                core_parts.append(extra)
                
        return core_parts

    def _build_context_sentence(self, time_of_day: str, time_of_day_cap: str, weather: str, rng: random.Random) -> str:
        if time_of_day and weather:
            return rng.choice([
                f"{time_of_day_cap} {weather}.",
                f"Das Ereignis trat {time_of_day} {weather} ein.",
                f"Der Vorfall ereignete sich {time_of_day} {weather}.",
                f"Bedingungen: {time_of_day}, {weather}.",
            ])
        if time_of_day:
            return rng.choice([
                f"Zeitpunkt: {time_of_day}.",
                f"Das Ereignis trat {time_of_day} ein.",
                f"{time_of_day_cap} kam es dazu.",
            ])
        if weather:
            return rng.choice([
                f"Wetterlage: {weather}.",
                f"Wetterbedingungen: {weather}.",
            ])
        return ""
