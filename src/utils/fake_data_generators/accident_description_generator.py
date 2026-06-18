import random
import pandas as pd
from dataclasses import dataclass, field
from src.ml_config import VALID_CASE_TYPES
from src.utils.fake_data_generators.accident_text_constants import (
    BODILY_INJURIES_INFO,
    CASE_CONFIG,
    DAMAGE_INTRO_TEMPLATES,
    DETAIL_LEVELS,
    DETAIL_LEVEL_WEIGHTS,
    EVENT_STANDALONE_TEMPLATES,
    POLICE_REPORT_INFO,
    PROPERTY_DAMAGE_INFO,
    SEVERITY_BUCKET_MAP,
    TIME_CONDITIONS,
    TOW_INFO,
    VEHICLE_IN_EVENT_TEMPLATES,
    VEHICLE_INTRO_TEMPLATES,
    VEHICLE_REF_WITH_YEAR,
    VEHICLE_REF_WITHOUT_YEAR,
    WEATHER_CONDITIONS,
    WITNESS_INFO_BY_COUNT,
    YEAR_STANDALONE_TEMPLATES,
)


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
        rng = random.Random(seed) if seed is not None else random.Random()
        text, _ = self._generate(row, rng)
        return text

    def generate_with_labels(self, row: dict, seed: int = None) -> tuple[str, dict]:
        rng = random.Random(seed) if seed is not None else random.Random()
        return self._generate(row, rng)

    def generate_with_labels_and_vehicles(
        self, row: dict, vehicle_ref: tuple[str, str, int], seed: int = None
    ):
        rng = random.Random(seed) if seed is not None else random.Random()
        ctx = self._create_generation_context(row, rng, vehicle=vehicle_ref)
        return self._generate(row, rng, ctx)

    def _generate(self, row: dict, rng: random.Random, ctx=None) -> tuple[str, dict]:
        if ctx is None:
            ctx = self._create_generation_context(row, rng)

        core_parts = self._build_core_sentences(ctx, rng)
        core_parts = self._inject_context_and_extras(core_parts, ctx, row, rng)

        labels = self._get_labels(ctx, row)
        text = " ".join(" ".join(core_parts).split())
        return text, labels

    def _create_generation_context(
        self,
        row: dict,
        rng: random.Random,
        vehicle: tuple[str, str, int] | None = None,
    ) -> GenerationContext:
        if vehicle:
            make, model, year = vehicle
        else:
            make = row.get("auto_make", "").strip()
            model = row.get("auto_model", "").strip()
            year = row["auto_year"]

        case_type = self._resolve_case_type(row)
        severity = str(row.get("incident_severity", ""))
        severity_bucket = SEVERITY_BUCKET_MAP.get(severity, "minor")

        # Vehicle Theft only has major/minor buckets
        if case_type == "Vehicle Theft":
            severity_bucket = "major" if severity_bucket in ("total_loss", "major") else "minor"

        detail_level = rng.choices(DETAIL_LEVELS, weights=DETAIL_LEVEL_WEIGHTS, k=1)[0]

        config = CASE_CONFIG[case_type]
        context_vars = self._generate_context_fields(config, severity_bucket, detail_level, rng)

        incident_phrase = rng.choice(config["incident_phrases"])
        damage_string = self._generate_damage_string(config["damage"][severity_bucket], rng)

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
            damage_string=damage_string,
        )

    def _generate_context_fields(
        self, config: dict, severity_bucket: str, detail_level: str, rng: random.Random
    ) -> dict:
        ctx = {k: "" for k in ["airbags", "weather", "time_of_day", "time_of_day_cap", "tow"]}

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
            ctx["tow"] = rng.choice(TOW_INFO)
            return ctx

        if detail_level == "noisy":
            if rng.random() < 0.65:
                ctx["airbags"] = rng.choice(config["airbags"][severity_bucket])
            if rng.random() < 0.60:
                ctx["weather"] = rng.choice(WEATHER_CONDITIONS)
            if rng.random() < 0.60:
                ctx["time_of_day"] = rng.choice(TIME_CONDITIONS)
                ctx["time_of_day_cap"] = ctx["time_of_day"].capitalize()
            if rng.random() < 0.25:
                ctx["tow"] = rng.choice(TOW_INFO)
            return ctx

        return ctx

    def _get_labels(self, ctx: GenerationContext, row: dict) -> dict:
        labels = {}
        vehicle_present = ctx.strategy != "no_vehicle"
        year_present = vehicle_present and ctx.with_year

        for field in self.target_fields:
            # Vehicle fields — take from ctx, not from insurance row
            if field == "auto_make":
                labels[field] = ctx.make if vehicle_present else None
                continue

            if field == "auto_year":
                labels[field] = ctx.year if year_present else None
                continue

            val = row.get(field, "")
            if val == "" or (isinstance(val, float) and pd.isna(val)) or val == "?":
                labels[field] = None
                continue

            # Field-specific presence logic
            if field == "witnesses":
                # Present only if detail_level allows it (handled in inject)
                labels[field] = val
            elif field == "bodily_injuries":
                labels[field] = val
            elif field == "property_damage":
                labels[field] = val
            elif field == "police_report_available":
                labels[field] = val
            else:
                labels[field] = val

        return labels

    def _build_core_sentences(self, ctx: GenerationContext, rng: random.Random) -> list[str]:
        damage_sentence = rng.choice(DAMAGE_INTRO_TEMPLATES).format(damage=ctx.damage_string)
        event_sentence = rng.choice(EVENT_STANDALONE_TEMPLATES).format(
            incident_phrase=ctx.incident_phrase
        )

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
            vehicle_sentence = rng.choice(VEHICLE_INTRO_TEMPLATES).format(
                vehicle_ref=vehicle_ref
            )
            return [vehicle_sentence, event_sentence, damage_sentence]

        if ctx.strategy == "event_then_vehicle":
            vehicle_ref = self._build_vehicle_ref(ctx, rng)
            vehicle_sentence = rng.choice(VEHICLE_INTRO_TEMPLATES).format(
                vehicle_ref=vehicle_ref
            )
            if rng.random() < 0.35:
                return [event_sentence, damage_sentence, vehicle_sentence]
            return [event_sentence, vehicle_sentence, damage_sentence]

        if ctx.strategy == "distant":
            vehicle_ref = rng.choice(VEHICLE_REF_WITHOUT_YEAR).format(
                make=ctx.make, model=ctx.model
            )
            vehicle_sentence = rng.choice(VEHICLE_INTRO_TEMPLATES).format(
                vehicle_ref=vehicle_ref
            )
            core_parts = [vehicle_sentence, event_sentence, damage_sentence]
            if ctx.with_year and ctx.year:
                year_sentence = rng.choice(YEAR_STANDALONE_TEMPLATES).format(year=ctx.year)
                core_parts.append(year_sentence)
            rng.shuffle(core_parts)
            return core_parts

        return [event_sentence, damage_sentence]

    def _inject_context_and_extras(
        self,
        core_parts: list[str],
        ctx: GenerationContext,
        row: dict,
        rng: random.Random,
    ) -> list[str]:
        # Make incident scope explicit so the model can learn incident_type and
        # number_of_vehicles_involved from text, not only from weak correlations.
        incident_scope_sentence = self._build_incident_scope_sentence(row, ctx.detail_level, rng)
        if incident_scope_sentence:
            core_parts.insert(rng.randint(0, len(core_parts)), incident_scope_sentence)

        if ctx.context_vars["time_of_day"] or ctx.context_vars["weather"]:
            sentence = self._build_context_sentence(
                ctx.context_vars["time_of_day"],
                ctx.context_vars["time_of_day_cap"],
                ctx.context_vars["weather"],
                rng,
            )
            if sentence:
                core_parts.insert(rng.randint(0, len(core_parts)), sentence)

        if ctx.context_vars["airbags"]:
            core_parts.append(ctx.context_vars["airbags"])

        if ctx.context_vars["tow"]:
            core_parts.append(ctx.context_vars["tow"])

        witnesses_val = row.get("witnesses")
        if witnesses_val is not None and ctx.detail_level != "minimal":
            try:
                count = int(witnesses_val)
                options = WITNESS_INFO_BY_COUNT.get(count, [])
                # noisy: 50% chance; others: always include
                if options and (ctx.detail_level != "noisy" or rng.random() < 0.60):
                    core_parts.append(rng.choice(options))
            except (ValueError, TypeError):
                pass

        injuries_val = row.get("bodily_injuries")
        if injuries_val is not None and ctx.detail_level in ("detailed", "noisy", "normal"):
            try:
                count = int(injuries_val)
                options = BODILY_INJURIES_INFO.get(count, [])
                if options and (ctx.detail_level != "noisy" or rng.random() < 0.70):
                    core_parts.append(rng.choice(options))
            except (ValueError, TypeError):
                pass

        prop_val = row.get("property_damage")
        if prop_val is not None and str(prop_val) in PROPERTY_DAMAGE_INFO:
            if ctx.detail_level != "minimal" and (
                ctx.detail_level != "noisy" or rng.random() < 0.55
            ):
                core_parts.append(rng.choice(PROPERTY_DAMAGE_INFO[str(prop_val)]))

        police_val = row.get("police_report_available")
        if police_val is not None and str(police_val) in POLICE_REPORT_INFO:
            if ctx.detail_level != "minimal" and (
                ctx.detail_level != "noisy" or rng.random() < 0.50
            ):
                core_parts.append(rng.choice(POLICE_REPORT_INFO[str(police_val)]))

        return core_parts

    def _resolve_case_type(self, row: dict) -> str:
        incident_type = str(row.get("incident_type", "")).strip()
        collision_type = str(row.get("collision_type", "")).strip()

        for value in (collision_type, incident_type):
            if value in VALID_CASE_TYPES:
                return value

        return "Side Collision"

    def _generate_damage_string(self, damage_cfg: dict, rng: random.Random) -> str:
        damage_count = rng.randint(*damage_cfg["count"])
        sampled = rng.sample(damage_cfg["pool"], min(damage_count, len(damage_cfg["pool"])))
        return ", ".join(sampled)

    def _determine_strategy(
        self, make: str, model: str, detail_level: str, rng: random.Random
    ) -> str:
        if not (make or model) or (detail_level == "noisy" and rng.random() < 0.40):
            return "no_vehicle"

        return rng.choices(
            ["vehicle_in_event", "vehicle_then_event", "event_then_vehicle", "distant"],
            weights=[0.28, 0.27, 0.25, 0.20],
            k=1,
        )[0]

    def _build_vehicle_ref(self, ctx: GenerationContext, rng: random.Random) -> str:
        if ctx.with_year and ctx.year:
            return rng.choice(VEHICLE_REF_WITH_YEAR).format(
                make=ctx.make, model=ctx.model, year=ctx.year
            )
        return rng.choice(VEHICLE_REF_WITHOUT_YEAR).format(make=ctx.make, model=ctx.model)

    def _build_context_sentence(
        self,
        time_of_day: str,
        time_of_day_cap: str,
        weather: str,
        rng: random.Random,
    ) -> str:
        if time_of_day and weather:
            return rng.choice(
                [
                    f"{time_of_day_cap} {weather}.",
                    f"Das Ereignis trat {time_of_day} {weather} ein.",
                    f"Der Vorfall ereignete sich {time_of_day} {weather}.",
                    f"Bedingungen: {time_of_day}, {weather}.",
                ]
            )
        if time_of_day:
            return rng.choice(
                [
                    f"Zeitpunkt: {time_of_day}.",
                    f"Das Ereignis trat {time_of_day} ein.",
                    f"{time_of_day_cap} kam es dazu.",
                ]
            )
        if weather:
            return rng.choice(
                [
                    f"Wetterlage: {weather}.",
                    f"Wetterbedingungen: {weather}.",
                ]
            )
        return ""

    def _build_incident_scope_sentence(
        self,
        row: dict,
        detail_level: str,
        rng: random.Random,
    ) -> str:
        if detail_level == "minimal":
            return ""

        incident_type = str(row.get("incident_type", "")).strip()
        count_raw = row.get("number_of_vehicles_involved")

        count = None
        try:
            if count_raw is not None:
                count = int(count_raw)
        except (TypeError, ValueError):
            count = None

        if incident_type == "Single Vehicle Collision":
            options = [
                "Es war nur ein Fahrzeug beteiligt.",
                "Der Vorfall betrifft ein einzelnes Fahrzeug.",
                "Es handelte sich um einen Einzelfahrzeugunfall.",
            ]
            return rng.choice(options)

        if incident_type == "Multi-vehicle Collision":
            if count is not None and count > 1:
                options = [
                    f"Insgesamt waren {count} Fahrzeuge beteiligt.",
                    f"Der Unfall umfasste {count} beteiligte Fahrzeuge.",
                    f"Es waren {count} Fahrzeuge in den Vorfall involviert.",
                ]
            else:
                options = [
                    "Es waren mehrere Fahrzeuge beteiligt.",
                    "Der Vorfall betraf mehrere beteiligte Fahrzeuge.",
                    "Es handelte sich um eine Mehrfahrzeugkollision.",
                ]
            return rng.choice(options)

        # Fallback: if a valid count exists, still expose it explicitly.
        if count is not None and count >= 1:
            options = [
                f"An dem Vorfall war {count} Fahrzeug beteiligt."
                if count == 1
                else f"An dem Vorfall waren {count} Fahrzeuge beteiligt.",
                f"Beteiligte Fahrzeuge: {count}.",
            ]
            return rng.choice(options)

        return ""