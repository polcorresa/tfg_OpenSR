from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    model_family: str
    required_bands: tuple[str, ...]
    model_band_order: tuple[str, ...]
    scale_factor: int
    description: str
    model_url: str | None = None
    model_dir_name: str | None = None


WORKFLOWS: dict[str, WorkflowSpec] = {
    "sen2sr_rgbn_x4": WorkflowSpec(
        name="sen2sr_rgbn_x4",
        model_family="sen2sr",
        required_bands=("B04", "B03", "B02", "B08"),
        model_band_order=("B04", "B03", "B02", "B08"),
        scale_factor=4,
        description="SEN2SRLite NonReference_RGBN_x4",
        model_url="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
        model_dir_name="SEN2SRLite_RGBN",
    ),
    "sen2sr_full_x4": WorkflowSpec(
        name="sen2sr_full_x4",
        model_family="sen2sr",
        required_bands=("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        model_band_order=("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        scale_factor=4,
        description="SEN2SRLite main",
        model_url="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
        model_dir_name="SEN2SRLite_full",
    ),
    "opensr_model_rgbn_x4": WorkflowSpec(
        name="opensr_model_rgbn_x4",
        model_family="opensr_model",
        required_bands=("B04", "B03", "B02", "B08"),
        model_band_order=("B04", "B03", "B02", "B08"),
        scale_factor=4,
        description="LDSR-S2 diffusion model from opensr-model",
    ),
}


def canonicalize_band_name(band_name: str) -> str:
    text = band_name.strip().upper().replace(" ", "")
    match = re.fullmatch(r"B?0?(\d{1,2})(A?)", text)
    if match is None:
        return text

    number = int(match.group(1))
    suffix = match.group(2)

    if suffix == "A":
        return f"B{number}A"

    return f"B{number:02d}"


def canonicalize_band_order(bands: list[str]) -> list[str]:
    return [canonicalize_band_name(band_name) for band_name in bands]


def missing_bands(available_bands: list[str], required_bands: tuple[str, ...]) -> list[str]:
    available = set(available_bands)
    return [band for band in required_bands if band not in available]


def compatibility_report(available_bands: list[str]) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for workflow_name, spec in WORKFLOWS.items():
        missing = missing_bands(available_bands, spec.required_bands)
        report[workflow_name] = {
            "description": spec.description,
            "compatible": len(missing) == 0,
            "missing_bands": missing,
        }
    return report


def resolve_workflow(requested_workflow: str, available_bands: list[str]) -> WorkflowSpec:
    if requested_workflow != "auto":
        try:
            spec = WORKFLOWS[requested_workflow]
        except KeyError as exc:
            raise ValueError(f"Unsupported workflow '{requested_workflow}'.") from exc

        missing = missing_bands(available_bands, spec.required_bands)
        if missing:
            raise ValueError(build_missing_band_message(available_bands, requested_workflow))
        return spec

    for workflow_name in ("sen2sr_full_x4", "sen2sr_rgbn_x4", "opensr_model_rgbn_x4"):
        spec = WORKFLOWS[workflow_name]
        if not missing_bands(available_bands, spec.required_bands):
            return spec

    raise ValueError(build_missing_band_message(available_bands, requested_workflow))


def reorder_indices(input_band_order: list[str], target_band_order: tuple[str, ...]) -> list[int]:
    band_to_index = {band_name: index for index, band_name in enumerate(input_band_order)}
    return [band_to_index[band_name] for band_name in target_band_order]


def build_missing_band_message(available_bands: list[str], requested_workflow: str) -> str:
    lines = [
        "Input bands do not match a supported OpenSR workflow.",
        f"Available bands: {', '.join(available_bands)}",
    ]

    if requested_workflow == "auto":
        for workflow_name, spec in WORKFLOWS.items():
            missing = missing_bands(available_bands, spec.required_bands)
            lines.append(
                f"- {workflow_name} ({spec.description}) is missing: {', '.join(missing)}"
            )
    else:
        spec = WORKFLOWS[requested_workflow]
        missing = missing_bands(available_bands, spec.required_bands)
        lines.append(
            f"- {requested_workflow} ({spec.description}) is missing: {', '.join(missing)}"
        )

    lines.append(
        "To get a 2.5 m output from the current Barcelona cutout, add B08 for the RGBN path, or add B05, B06, B07, and B08 for the full SEN2SR path."
    )
    return "\n".join(lines)
