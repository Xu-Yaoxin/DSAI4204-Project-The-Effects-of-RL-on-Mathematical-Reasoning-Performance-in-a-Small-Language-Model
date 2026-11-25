"""Build synthetic controlled arithmetic families requiring manual review."""

import argparse
import hashlib
import json
import random
import re
import tempfile
import unicodedata
from fractions import Fraction
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--families", type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()
code_root = Path(__file__).resolve().parents[1]
config_path = Path(args.config).expanduser()
if not config_path.is_absolute():
    config_path = code_root / config_path
config = json.loads(config_path.read_text(encoding="utf-8"))
data_config = config["data"]
base_seed = int(data_config.get("seed", config.get("seeds", {}).get("data", config.get("seed", 42))))
seed = args.seed if args.seed is not None else int(data_config.get("structural_seed", base_seed + 1))
family_count = args.families if args.families is not None else int(data_config.get("structural_family_count", 40))
if family_count < 1:
    raise SystemExit("--families must be positive.")
output_dir = Path(data_config["output_dir"]).expanduser()
if not output_dir.is_absolute():
    output_dir = code_root / output_dir
rng = random.Random(seed)
records = []
seen_questions = {}
split_manifest_path = output_dir / "manifest.json"
if not split_manifest_path.is_file():
    raise SystemExit(f"Missing {split_manifest_path}. Run prepare_data.py before building structural variants.")
split_manifest_bytes = split_manifest_path.read_bytes()
split_manifest = json.loads(split_manifest_bytes)
for split in ["train", "dev", "test"]:
    split_path = output_dir / (split + ".jsonl")
    if not split_path.is_file():
        raise SystemExit(f"Missing {split_path}. Run prepare_data.py first so structural overlap can be checked.")
    split_bytes = split_path.read_bytes()
    expected_hash = split_manifest.get("files", {}).get(split + ".jsonl", {}).get("sha256")
    if hashlib.sha256(split_bytes).hexdigest() != expected_hash:
        raise SystemExit(f"Split hash mismatch: {split_path}. Restore the immutable prepared data before building variants.")
    for line in split_bytes.decode("utf-8").split("\n"):
        if line.strip():
            existing = json.loads(line)
            normalized = re.sub(r"\W+", " ", unicodedata.normalize("NFKC", existing["question"]).casefold()).strip()
            seen_questions[normalized] = existing["id"]

for family_index in range(family_count):
    pair_id = f"controlled-pair-{family_index:04d}"
    template_index = family_index % 4
    accepted = False
    attempts = 0
    while not accepted:
        attempts += 1
        if attempts > 10000:
            raise SystemExit("Could not generate unique families; reduce --families or change --seed.")
        n = rng.randint(3, 12)
        p = rng.randint(12, 36)
        d = rng.randint(2, 7)
        n2, p2, d2 = n + rng.randint(1, 3), p + rng.randint(1, 5), d + rng.randint(1, 3)
        family_variants = []
        if template_index == 0:
            template_name = "item_discount"
            family_variants = [
                ("original", f"A shopper buys {n} notebooks. Each notebook costs {p} dollars before a discount of {d} dollars on each notebook. What is the total cost in dollars?", Fraction(n * (p - d)), f"{n} * ({p} - {d})", ["subtraction", "multiplication"], {"items": n, "price": p, "discount": d}),
                ("numerical", f"A shopper buys {n2} notebooks. Each notebook costs {p2} dollars before a discount of {d2} dollars on each notebook. What is the total cost in dollars?", Fraction(n2 * (p2 - d2)), f"{n2} * ({p2} - {d2})", ["subtraction", "multiplication"], {"items": n2, "price": p2, "discount": d2}),
                ("relational", f"A shopper buys {n} notebooks. Each notebook costs {p} dollars. A single coupon deducts {d} dollars from the entire purchase. What is the total cost in dollars?", Fraction(n * p - d), f"{n} * {p} - {d}", ["multiplication", "subtraction"], {"items": n, "price": p, "discount": d}),
                ("recombined", f"A shopper planned to buy {n} notebooks and adds {d} more notebooks. Each notebook costs {p} dollars, with no discount. What is the total cost in dollars?", Fraction((n + d) * p), f"({n} + {d}) * {p}", ["addition", "multiplication"], {"items": n, "price": p, "extra_items": d}),
            ]
        elif template_index == 1:
            template_name = "boxed_stock"
            family_variants = [
                ("original", f"A store has {n} boxes with {p} pencils in each box and {d} loose pencils outside the boxes. How many pencils does the store have?", Fraction(n * p + d), f"{n} * {p} + {d}", ["multiplication", "addition"], {"boxes": n, "per_box": p, "loose": d}),
                ("numerical", f"A store has {n2} boxes with {p2} pencils in each box and {d2} loose pencils outside the boxes. How many pencils does the store have?", Fraction(n2 * p2 + d2), f"{n2} * {p2} + {d2}", ["multiplication", "addition"], {"boxes": n2, "per_box": p2, "loose": d2}),
                ("relational", f"A store has {n} boxes. Each box contains {p} regular pencils and {d} colored pencils. How many pencils does the store have?", Fraction(n * (p + d)), f"{n} * ({p} + {d})", ["addition", "multiplication"], {"boxes": n, "regular_per_box": p, "colored_per_box": d}),
                ("recombined", f"A store has {n} boxes with {p} pencils in each box. It removes {d} damaged pencils from the total stock. How many pencils remain?", Fraction(n * p - d), f"{n} * {p} - {d}", ["multiplication", "subtraction"], {"boxes": n, "per_box": p, "damaged": d}),
            ]
        elif template_index == 2:
            template_name = "shared_fund"
            p, p2 = max(p, n * d + 4), max(p2, n2 * d2 + 4)
            total, fee = n * p, n * d
            total2, fee2 = n2 * p2, n2 * d2
            family_variants = [
                ("original", f"A group has a fund of {total} dollars. A total fee of {fee} dollars is paid from the fund, and the remainder is shared equally among {n} people. How many dollars does each person receive?", Fraction(total - fee, n), f"({total} - {fee}) / {n}", ["subtraction", "division"], {"fund": total, "fee": fee, "people": n}),
                ("numerical", f"A group has a fund of {total2} dollars. A total fee of {fee2} dollars is paid from the fund, and the remainder is shared equally among {n2} people. How many dollars does each person receive?", Fraction(total2 - fee2, n2), f"({total2} - {fee2}) / {n2}", ["subtraction", "division"], {"fund": total2, "fee": fee2, "people": n2}),
                ("relational", f"A fund of {total} dollars is shared equally among {n} people. Each person then pays a personal fee of {fee} dollars from their share. How many dollars does each person keep?", Fraction(total, n) - fee, f"{total} / {n} - {fee}", ["division", "subtraction"], {"fund": total, "personal_fee": fee, "people": n}),
                ("recombined", f"A fund of {total} dollars is shared equally among {n} people. Each person then receives an extra gift of {fee} dollars. How many dollars does each person have?", Fraction(total, n) + fee, f"{total} / {n} + {fee}", ["division", "addition"], {"fund": total, "personal_gift": fee, "people": n}),
            ]
        else:
            template_name = "travel_distance"
            family_variants = [
                ("original", f"A cyclist rides at {p} kilometers per hour for {n} hours, then rides another {d} kilometers in the same direction. How many kilometers does the cyclist travel in total?", Fraction(p * n + d), f"{p} * {n} + {d}", ["multiplication", "addition"], {"speed": p, "hours": n, "extra_km": d}),
                ("numerical", f"A cyclist rides at {p2} kilometers per hour for {n2} hours, then rides another {d2} kilometers in the same direction. How many kilometers does the cyclist travel in total?", Fraction(p2 * n2 + d2), f"{p2} * {n2} + {d2}", ["multiplication", "addition"], {"speed": p2, "hours": n2, "extra_km": d2}),
                ("relational", f"A cyclist rides at {p} kilometers per hour for {n} hours, then continues at the same speed for another {d} hours. How many kilometers does the cyclist travel in total?", Fraction(p * (n + d)), f"{p} * ({n} + {d})", ["addition", "multiplication"], {"speed": p, "hours": n, "extra_hours": d}),
                ("recombined", f"A cyclist rides away from home at {p} kilometers per hour for {n} hours, then rides {d} kilometers back along the same route. How many kilometers from home is the cyclist now?", Fraction(p * n - d), f"{p} * {n} - {d}", ["multiplication", "subtraction"], {"speed": p, "hours": n, "return_km": d}),
            ]
        normalized_variants = [re.sub(r"\W+", " ", unicodedata.normalize("NFKC", variant[1]).casefold()).strip() for variant in family_variants]
        accepted = len(set(normalized_variants)) == 4 and all(value not in seen_questions for value in normalized_variants)
    for (variant_type, question, value, expression, operations, parameters), normalized in zip(family_variants, normalized_variants):
        answer = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        record_id = pair_id + "-" + variant_type
        records.append({"id": record_id, "question": question, "answer": answer, "split": "structural",
                        "pair_id": pair_id, "family_id": template_name, "variant_type": variant_type, "verified": False,
                        "source": {"dataset": "synthetic-controlled-v1", "template": template_name, "seed": seed},
                        "oracle": {"expression": expression, "exact_answer": answer, "operation_count": 2,
                                   "operations": operations, "parameters": parameters},
                        "review_required": True,
                        "scope": "Controlled synthetic variation; no audited claim of unseen operation combinations."})
        seen_questions[normalized] = record_id

payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records).encode("utf-8")
dataset_sha256 = hashlib.sha256(payload).hexdigest()
manifest = {"schema_version": 1, "kind": "synthetic_controlled_variation", "seed": seed,
            "pairs": family_count, "template_families": len({record["family_id"] for record in records}),
            "rows": len(records), "dataset_sha256": dataset_sha256,
            "verified": False, "review_required": True, "all_operation_counts": 2,
            "comparison_note": "Operation count is matched; semantic and numerical difficulty still require manual review.",
            "scope": "Synthetic controlled variants, not GSM-Symbolic and not evidence of unseen compositional generalization.",
            "leakage_check": "Normalized questions checked against train/dev/test and within structural data.",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "split_manifest_sha256": hashlib.sha256(split_manifest_bytes).hexdigest()}
review_template = {"schema_version": 1, "dataset_sha256": dataset_sha256, "approved": False, "reviewer": "",
                   "instructions": "Copy to structural_review.json. Independently solve every question and enter reviewed_answer, then check the gold answer and family relation. Set each verified field only after review. Complete reviewer and approved only when all rows pass. Do not alter the immutable dataset to record review.",
                   "records": [{"id": record["id"], "question_sha256": hashlib.sha256(record["question"].encode("utf-8")).hexdigest(), "reviewed_answer": "", "verified": False, "notes": ""} for record in records]}
output_bytes = {"structural.jsonl": payload,
                "structural_manifest.json": (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
                "structural_review.template.json": (json.dumps(review_template, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")}
for filename, file_bytes in output_bytes.items():
    destination = output_dir / filename
    if destination.exists() and destination.read_bytes() != file_bytes:
        raise SystemExit(f"Immutable structural conflict: {destination}. Use a new output_dir for a revised dataset; existing files were preserved.")
written = 0
for filename, file_bytes in output_bytes.items():
    destination = output_dir / filename
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".variants-", delete=False) as temporary:
            temporary.write(file_bytes)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
        written += 1
print(f"[STRUCTURAL 100%] pairs={family_count} template_families={manifest['template_families']} rows={len(records)} | exact_oracles=generated | manual_review=PENDING")
print(f"[SAVED] {output_dir / 'structural_manifest.json'} | new_files={written} | verified=false")
