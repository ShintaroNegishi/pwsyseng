"""ケースファイル (YAML) の読み書き。

ケースはパッケージに同梱してあり、名前だけで引ける。

>>> from gridops import load_case
>>> case = load_case("wscc9")
>>> case.n_bus
9

カレントディレクトリに依存しないのが要点である。安定度の教材 genstab の
``notebooks/src/08_multimachine.py`` は ``Path("cases/wscc9.yaml")`` を
探す実装になっており、notebook をどこから起動したかで動いたり動かなかったり
する。同梱データを :mod:`importlib.resources` で引くことで、この種の
「先生の画面では動くのに」を構造的に無くしている。
"""

from __future__ import annotations

import math
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .case import Branch, Bus, BusType, Case, ReferenceSolution, Unit

#: 同梱ケースが置かれているディレクトリ名（パッケージ gridops の直下）。
CASEDATA_DIR = "casedata"


def _casedata_root():
    """同梱ケースのディレクトリを :mod:`importlib.resources` で引く。"""
    return resources.files("gridops") / CASEDATA_DIR


def list_cases() -> list[str]:
    """同梱されているケースの名前を返す。"""
    names = [
        path.name[: -len(".yaml")]
        for path in _casedata_root().iterdir()
        if path.name.endswith(".yaml")
    ]
    return sorted(names)


def case_path(name: str) -> Path:
    """同梱ケースのファイルパスを返す。

    Raises
    ------
    FileNotFoundError
        その名前のケースが同梱されていないとき。
    """
    candidate = _casedata_root() / f"{name}.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"同梱ケース '{name}' はない。使えるのは {list_cases()}。"
            "自作のファイルを読むときはパスをそのまま load_case に渡すこと。"
        )
    return Path(str(candidate))


def load_case(source: str | Path) -> Case:
    """ケースを読み込む。

    Parameters
    ----------
    source:
        同梱ケースの名前（``"wscc9"`` など）か、YAML ファイルのパス。

    Notes
    -----
    ``solution`` 層は :class:`~gridops.case.ReferenceSolution` に入り、
    潮流計算の **入力には使われない**。答え合わせと初期値にだけ使う。
    """
    import yaml

    path = Path(source)
    if not path.suffix:
        path = case_path(str(source))
    if not path.is_file():
        raise FileNotFoundError(f"ケースファイルが見つからない: {path}")

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    schema = int(data.get("schema", 1))
    if schema != 2:
        raise ValueError(
            f"{path} のスキーマ版 {schema} は読めない（本パッケージは 2）。"
            "genstab の cases/*.yaml は版 1 で、母線に潮流解 (v, angle_deg) を"
            "入力として持つ形式である。gridops.interop で変換すること。"
        )

    limits = data.get("voltage_limits", {}) or {}
    v_min = float(limits.get("v_min", 0.95))
    v_max = float(limits.get("v_max", 1.05))

    buses = [_parse_bus(item, v_min, v_max) for item in data["buses"]]
    branches = [_parse_branch(item) for item in data.get("branches", [])]
    units = [_parse_unit(item) for item in data.get("units", [])]

    contingencies = [
        (int(item["from"]), int(item["to"]))
        for item in (data.get("contingencies", {}) or {}).get("branches", [])
    ]

    case = Case(
        name=str(data.get("name", path.stem)),
        base_mva=float(data.get("base_mva", 100.0)),
        frequency_hz=float(data.get("frequency_hz", 60.0)),
        buses=buses,
        branches=branches,
        units=units,
        commitment=data.get("commitment", {}) or {},
        reliability=data.get("reliability", {}) or {},
        stability=data.get("stability", {}) or {},
        contingencies=contingencies,
        reference=_parse_solution(data.get("solution"), buses),
        source=str(data.get("source", "")),
        modifications=str(data.get("modifications", "")),
    )
    return case


# ----------------------------------------------------------------------
# 個々の要素
# ----------------------------------------------------------------------
def _parse_bus(item: dict[str, Any], v_min: float, v_max: float) -> Bus:
    return Bus(
        id=int(item["id"]),
        type=BusType(str(item.get("type", "pq")).lower()),
        v_set=float(item.get("v_set", 1.0)),
        pd=float(item.get("pd", 0.0)),
        qd=float(item.get("qd", 0.0)),
        gs=float(item.get("gs", 0.0)),
        bs=float(item.get("bs", 0.0)),
        v_min=float(item.get("v_min", v_min)),
        v_max=float(item.get("v_max", v_max)),
        name=str(item.get("name", "")),
    )


def _parse_branch(item: dict[str, Any]) -> Branch:
    return Branch(
        from_bus=int(item["from"]),
        to_bus=int(item["to"]),
        r=float(item.get("r", 0.0)),
        x=float(item["x"]),
        b=float(item.get("b", 0.0)),
        tap=float(item.get("tap", 1.0)),
        shift_deg=float(item.get("shift_deg", 0.0)),
        rate_a=float(item.get("rate_a", math.inf)),
        rate_b=float(item.get("rate_b", math.inf)),
        name=str(item.get("name", "")),
    )


def _parse_unit(item: dict[str, Any]) -> Unit:
    return Unit(
        name=str(item["name"]),
        bus=int(item["bus"]),
        plant=str(item.get("plant", "")),
        p_max_mw=float(item.get("p_max_mw", 0.0)),
        p_min_mw=float(item.get("p_min_mw", 0.0)),
        q_min=float(item.get("q_min", -math.inf)),
        q_max=float(item.get("q_max", math.inf)),
        var_cost=float(item.get("var_cost", 0.0)),
        quadratic=float(item.get("quadratic", 0.0)),
        noload_cost=float(item.get("noload_cost", 0.0)),
        startup_cost=float(item.get("startup_cost", 0.0)),
        min_up=int(item.get("min_up", 1)),
        min_down=int(item.get("min_down", 1)),
        ramp_up=float(item.get("ramp_up", math.inf)),
        ramp_down=float(item.get("ramp_down", math.inf)),
        startup_ramp=_optional_float(item.get("startup_ramp")),
        shutdown_ramp=_optional_float(item.get("shutdown_ramp")),
        u0=int(item.get("u0", 0)),
        hours_in_state=int(item.get("hours_in_state", 24)),
        forced_outage_rate=float(item.get("forced_outage_rate", 0.0)),
        mttf=_optional_float(item.get("mttf")),
        mttr=_optional_float(item.get("mttr")),
        h=_optional_float(item.get("h")),
        xd_prime=_optional_float(item.get("xd_prime")),
        d=float(item.get("d", 0.0)),
    )


def _parse_solution(
    data: dict[str, Any] | None, buses: list[Bus]
) -> ReferenceSolution | None:
    if not data:
        return None
    by_id = {int(item["id"]): item for item in data["buses"]}
    v = np.array([float(by_id[bus.id]["v"]) for bus in buses])
    angle = np.array([float(by_id[bus.id]["angle_deg"]) for bus in buses])
    generation = {
        int(item["bus"]): (float(item["p"]), float(item.get("q", 0.0)))
        for item in data.get("generation", [])
    }
    return ReferenceSolution(
        v=v,
        angle_deg=angle,
        generation=generation,
        checks=data.get("checks", {}) or {},
        digits=int(data.get("digits", 4)),
        source=str(data.get("source", "")),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
