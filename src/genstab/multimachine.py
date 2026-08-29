"""多機系統の過渡安定性（縮約 Ybus による古典モデル）。

SMIB では見えない現象、すなわち発電機どうしが互いに逆位相で振れる
「機器間動揺モード」を扱うためのモジュールである。

計算の流れ
----------
1. 線路・変圧器から母線アドミタンス行列 Ybus を組む
2. 負荷を定インピーダンスに変換して Ybus の対角成分に加える
3. 各発電機の内部母線（背後リアクタンス x'_d の先）を追加する
4. 発電機内部母線以外を Kron 縮約して、発電機の数だけの行列に縮める
5. 縮約行列から各機の電気出力を求める

.. math::

    P_{e,i} = \\sum_{j} E_i E_j
              \\bigl(G_{ij}\\cos(\\delta_i - \\delta_j)
                   + B_{ij}\\sin(\\delta_i - \\delta_j)\\bigr)

負荷を定インピーダンスとみなすのは、負荷母線を縮約で消すために必要な
仮定である。実際の負荷は電圧や周波数に依存するため、これは近似である
点に注意すること。

安定判定
--------
多機系統では「どれか 1 台の角度」ではなく、機器間の角度差
``max |δ_i - δ_j|`` を見る。系全体が一緒に回っても同期は失われないが、
機どうしの角度が開けば脱調する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .events import FaultSchedule, Stage
from .units import DEFAULT_BASE, SystemBase


@dataclass(frozen=True)
class Branch:
    """線路または変圧器（π 型等価回路）。"""

    from_bus: int
    to_bus: int
    r: float = 0.0       #: 直列抵抗 [p.u.]
    x: float = 0.1       #: 直列リアクタンス [p.u.]
    b: float = 0.0       #: 全充電サセプタンス [p.u.]（両端に半分ずつ配分）

    def key(self) -> tuple[int, int]:
        return (min(self.from_bus, self.to_bus), max(self.from_bus, self.to_bus))


@dataclass(frozen=True)
class Load:
    """定電力負荷（定インピーダンスに変換して扱う）。"""

    bus: int
    P: float = 0.0  #: 有効電力 [p.u.]
    Q: float = 0.0  #: 無効電力 [p.u.]


@dataclass(frozen=True)
class GeneratorData:
    """古典モデルの発電機（多機系統用）。"""

    bus: int
    H: float                #: 慣性定数 [s]
    xd_prime: float         #: 過渡リアクタンス [p.u.]
    P: float                #: 事故前の有効電力出力 [p.u.]
    Q: float = 0.0          #: 事故前の無効電力出力 [p.u.]
    D: float = 0.0          #: 制動係数 [p.u.]
    name: str = ""


@dataclass
class MultiMachineNetwork:
    """母線アドミタンス行列と Kron 縮約。

    Parameters
    ----------
    buses:
        母線番号の一覧。
    branches:
        線路・変圧器。
    loads:
        負荷。
    voltages:
        事故前潮流解における各母線の複素電圧 ``{母線番号: 複素電圧}``。
        負荷の定インピーダンス変換と発電機内部電圧の計算に使う。
    """

    buses: Sequence[int]
    branches: Sequence[Branch]
    loads: Sequence[Load]
    voltages: dict[int, complex]

    def __post_init__(self) -> None:
        self._index = {bus: i for i, bus in enumerate(self.buses)}

    # ------------------------------------------------------------------
    def ybus(
        self,
        removed_branches: Sequence[tuple[int, int]] = (),
        *,
        include_loads: bool = True,
    ) -> np.ndarray:
        """母線アドミタンス行列を組む。

        Parameters
        ----------
        removed_branches:
            開放する枝を ``(母線, 母線)`` の組で指定する。事故除去で
            線路を切り離す場合に使う。
        include_loads:
            負荷の定インピーダンス分を対角成分に加えるか。``False`` に
            すると線路と変圧器だけの行列になり、潮流解との整合性を
            検算するのに使える。
        """
        n = len(self.buses)
        Y = np.zeros((n, n), dtype=complex)
        removed = {(min(a, b), max(a, b)) for a, b in removed_branches}

        for branch in self.branches:
            if branch.key() in removed:
                continue
            i, j = self._index[branch.from_bus], self._index[branch.to_bus]
            impedance = complex(branch.r, branch.x)
            if impedance == 0:
                raise ValueError(
                    f"枝 {branch.from_bus}-{branch.to_bus} のインピーダンスがゼロ。"
                )
            y_series = 1.0 / impedance
            y_shunt = 1j * branch.b / 2.0

            Y[i, i] += y_series + y_shunt
            Y[j, j] += y_series + y_shunt
            Y[i, j] -= y_series
            Y[j, i] -= y_series

        if not include_loads:
            return Y

        # 負荷を定インピーダンスに変換する。Y = conj(S) / |V|^2
        for load in self.loads:
            i = self._index[load.bus]
            v = self.voltages[load.bus]
            Y[i, i] += np.conj(complex(load.P, load.Q)) / (abs(v) ** 2)

        return Y

    # ------------------------------------------------------------------
    def reduced_matrix(
        self,
        generators: Sequence[GeneratorData],
        *,
        removed_branches: Sequence[tuple[int, int]] = (),
        grounded_buses: Sequence[int] = (),
    ) -> np.ndarray:
        """発電機内部母線だけに縮約したアドミタンス行列を返す。

        Parameters
        ----------
        removed_branches:
            開放する枝。
        grounded_buses:
            三相地絡した母線。電圧がゼロになるので縮約から取り除く
            （接地されたのと同じ扱いになる）。
        """
        n_bus = len(self.buses)
        n_gen = len(generators)

        # 元の母線 + 発電機内部母線に拡張する。
        size = n_bus + n_gen
        Y = np.zeros((size, size), dtype=complex)
        Y[:n_bus, :n_bus] = self.ybus(removed_branches)

        for k, generator in enumerate(generators):
            i = self._index[generator.bus]
            internal = n_bus + k
            y = 1.0 / (1j * generator.xd_prime)
            Y[i, i] += y
            Y[internal, internal] += y
            Y[i, internal] -= y
            Y[internal, i] -= y

        # 地絡母線は電圧ゼロなので、縮約対象から除外する。
        grounded = {self._index[bus] for bus in grounded_buses}
        keep_internal = list(range(n_bus, size))
        keep_other = [i for i in range(n_bus) if i not in grounded]

        Y_gg = Y[np.ix_(keep_internal, keep_internal)]
        if not keep_other:
            return Y_gg
        Y_gl = Y[np.ix_(keep_internal, keep_other)]
        Y_lg = Y[np.ix_(keep_other, keep_internal)]
        Y_ll = Y[np.ix_(keep_other, keep_other)]

        # Kron 縮約: Y_red = Y_gg - Y_gl Y_ll^-1 Y_lg
        return Y_gg - Y_gl @ np.linalg.solve(Y_ll, Y_lg)

    # ------------------------------------------------------------------
    def power_injections(
        self, removed_branches: Sequence[tuple[int, int]] = ()
    ) -> dict[int, complex]:
        """潮流解から各母線の複素注入電力を求める（データの検算用）。

        .. math::

            \\bar{S}_i = \\bar{V}_i \\left(\\sum_j Y_{ij}\\bar{V}_j\\right)^*

        発電機母線では発電量、負荷母線では負荷の符号を反転した値に
        一致するはずである。一致しなければ、ケースファイルの潮流解と
        線路データのどちらかが食い違っている。
        """
        Y = self.ybus(removed_branches, include_loads=False)
        v = np.array([self.voltages[bus] for bus in self.buses])
        s = v * np.conj(Y @ v)
        return {bus: complex(s[i]) for i, bus in enumerate(self.buses)}

    def internal_voltages(
        self, generators: Sequence[GeneratorData]
    ) -> np.ndarray:
        """事故前潮流解から各機の内部起電力 ``E∠δ`` を求める。

        .. math::

            \\bar{E}_i = \\bar{V}_i + j x'_{d,i} \\bar{I}_i, \\qquad
            \\bar{I}_i = \\left(\\frac{P_i + jQ_i}{\\bar{V}_i}\\right)^*
        """
        result = np.empty(len(generators), dtype=complex)
        for k, generator in enumerate(generators):
            v = self.voltages[generator.bus]
            current = np.conj(complex(generator.P, generator.Q) / v)
            result[k] = v + 1j * generator.xd_prime * current
        return result


def electrical_power(
    Y_reduced: np.ndarray, magnitudes: np.ndarray, angles: np.ndarray
) -> np.ndarray:
    """縮約行列から各機の電気出力 [p.u.] を求める。"""
    G = Y_reduced.real
    B = Y_reduced.imag
    difference = angles[:, None] - angles[None, :]
    products = magnitudes[:, None] * magnitudes[None, :]
    return np.sum(products * (G * np.cos(difference) + B * np.sin(difference)), axis=1)


@dataclass
class MultiMachineSystem:
    """多機系統（全機が古典モデル）。

    状態ベクトルは ``[δ_1, ω_1, δ_2, ω_2, ...]`` の順に並ぶ。

    Parameters
    ----------
    network:
        母線データ。
    generators:
        発電機データ。
    fault:
        事故スケジュール。
    faulted_buses:
        三相地絡する母線。
    tripped_branches:
        事故除去で開放する枝。
    base:
        系統の基準値。WSCC 9 母線系統は 60 Hz なので注意すること。
    power_flow_tolerance:
        潮流解と発電機の宣言出力の許容差 [p.u.]。構築時に整合性を
        検証し、これを超えたら例外を送出する。
    """

    network: MultiMachineNetwork
    generators: Sequence[GeneratorData]
    fault: FaultSchedule = field(default_factory=FaultSchedule.none)
    faulted_buses: Sequence[int] = ()
    tripped_branches: Sequence[tuple[int, int]] = ()
    base: SystemBase = DEFAULT_BASE
    power_flow_tolerance: float = 5e-3

    def __post_init__(self) -> None:
        self.verify_power_flow(self.power_flow_tolerance)
        self.n_machines = len(self.generators)
        self.n_states = 2 * self.n_machines
        names: list[str] = []
        for k, generator in enumerate(self.generators):
            label = generator.name or f"G{k + 1}"
            names.extend([f"delta_{label}", f"omega_{label}"])
        self.state_names = tuple(names)

        emf = self.network.internal_voltages(self.generators)
        self.emf_magnitude = np.abs(emf)
        self._delta0 = np.angle(emf)

        # 3 つのネットワーク状態それぞれについて縮約行列を先に作っておく。
        self._reduced = {
            Stage.PRE: self.network.reduced_matrix(self.generators),
            Stage.FAULT: self.network.reduced_matrix(
                self.generators, grounded_buses=self.faulted_buses
            ),
            Stage.POST: self.network.reduced_matrix(
                self.generators, removed_branches=self.tripped_branches
            ),
        }

        self._Pm0 = electrical_power(
            self._reduced[Stage.PRE], self.emf_magnitude, self._delta0
        )

    # ------------------------------------------------------------------
    def verify_power_flow(self, tolerance: float = 5e-3) -> dict[int, complex]:
        """潮流解と発電機の宣言出力が整合するか検証する。

        機械入力 ``Pm0`` は縮約ネットワークから計算した電気出力として
        求めるため、``GeneratorData.P`` が誤っていても定常状態の残差は
        必ずゼロになる。つまり「平衡点である」というテストだけでは
        データの誤りを検出できない（実際、G2 の宣言出力を 1.63 から 0.30 に
        書き換えても残差はゼロのままだった）。

        そこで構築時に、母線電圧から独立に計算した各母線の注入電力と、
        宣言された発電量・負荷を突き合わせる。ここが合っていなければ
        以降の計算は「解けているが間違っている」状態になる。

        Returns
        -------
        母線番号から複素注入電力への辞書。

        Raises
        ------
        ValueError
            許容差を超える食い違いがある場合。
        """
        injections = self.network.power_injections()

        declared: dict[int, complex] = {}
        for generator in self.generators:
            declared[generator.bus] = declared.get(generator.bus, 0j) + complex(
                generator.P, generator.Q
            )
        for load in self.network.loads:
            declared[load.bus] = declared.get(load.bus, 0j) - complex(load.P, load.Q)

        problems = []
        for bus in self.network.buses:
            expected = declared.get(bus, 0j)
            actual = injections[bus]
            error = abs(actual - expected)
            if error > tolerance:
                problems.append(
                    f"母線 {bus}: 潮流解からの注入 {actual.real:+.4f}{actual.imag:+.4f}j に対し、"
                    f" 宣言値は {expected.real:+.4f}{expected.imag:+.4f}j (差 {error:.2e})"
                )

        if problems:
            detail = "\n  - ".join(problems)
            raise ValueError(
                "潮流解とネットワーク・発電機・負荷のデータが整合しない:\n  - "
                + detail
                + f"\n許容差は {tolerance:g} p.u.。ケースファイルの電圧・位相・"
                " 線路定数・発電機出力・負荷のいずれかが誤っている。"
            )
        return injections

    def initial_state(self) -> np.ndarray:
        """事故前定常状態を返す。"""
        x = np.zeros(self.n_states)
        x[0::2] = self._delta0
        return x

    def switching_times(self, t_end: float) -> list[float]:
        return self.fault.switching_times(t_end)

    def stage_at(self, t: float) -> Stage:
        return self.fault.stage(t)

    def reduced_matrix(self, stage: Stage) -> np.ndarray:
        """指定したネットワーク状態の縮約アドミタンス行列。"""
        return self._reduced[stage]

    # ------------------------------------------------------------------
    def derivatives(
        self, t: float, x: np.ndarray, stage: Stage | None = None
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_states,):
            raise ValueError(
                f"状態ベクトルの形状 {x.shape} が多機系統の状態数 "
                f"({self.n_states},) と一致しない。"
            )
        if stage is None:
            stage = self.stage_at(t)

        angles = x[0::2]
        speeds = x[1::2]
        power = electrical_power(self._reduced[stage], self.emf_magnitude, angles)

        dx = np.empty_like(x)
        dx[0::2] = self.base.omega_s * speeds
        for k, generator in enumerate(self.generators):
            dx[2 * k + 1] = (
                self._Pm0[k] - power[k] - generator.D * speeds[k]
            ) / (2.0 * generator.H)
        return dx

    # ------------------------------------------------------------------
    def algebraic_outputs(
        self, t: np.ndarray, x: np.ndarray
    ) -> dict[str, np.ndarray]:
        """各機の電気出力と、慣性中心 (COI) 基準の角度を求める。"""
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        outputs: dict[str, np.ndarray] = {}

        power = np.empty((self.n_machines, t.size))
        for i in range(t.size):
            power[:, i] = electrical_power(
                self._reduced[self.stage_at(t[i])],
                self.emf_magnitude,
                x[0::2, i],
            )
        for k, generator in enumerate(self.generators):
            label = generator.name or f"G{k + 1}"
            outputs[f"Pe_{label}"] = power[k]

        inertia = np.array([g.H for g in self.generators])
        coi = (inertia @ x[0::2]) / inertia.sum()
        outputs["delta_coi"] = coi
        for k, generator in enumerate(self.generators):
            label = generator.name or f"G{k + 1}"
            outputs[f"delta_coi_{label}"] = x[2 * k] - coi

        outputs["max_separation"] = x[0::2].max(axis=0) - x[0::2].min(axis=0)
        return outputs

    # ------------------------------------------------------------------
    def assess_stability(self, result, angle_limit: float = math.pi) -> bool:
        """機器間の角度差で過渡安定性を判定する。

        多機系統では 1 台の角度そのものではなく、機どうしの角度差
        ``max |δ_i - δ_j|`` を見る。系全体が一緒にずれても同期は
        保たれるためである。
        """
        if not np.all(np.isfinite(result.x)):
            return False
        separation = result.outputs.get("max_separation")
        if separation is None:
            separation = result.x[0::2].max(axis=0) - result.x[0::2].min(axis=0)
        return bool(np.max(np.abs(separation)) <= angle_limit)

    def describe(self) -> str:
        """構成の要約を返す。"""
        lines = [
            f"MultiMachineSystem  ({self.n_machines} 機, 状態数 {self.n_states})",
            f"  基準周波数 : {self.base.frequency_hz} Hz",
            f"  母線数     : {len(self.network.buses)}, 枝数 {len(self.network.branches)}",
            f"  事故       : 母線 {list(self.faulted_buses)} 地絡, "
            f"t={self.fault.t_fault}〜{self.fault.t_clear} s "
            f"(除去時間 {self.fault.clearing_time:.3f} s)",
            f"  事故除去   : 枝 {[tuple(b) for b in self.tripped_branches]} 開放",
            "  発電機:",
        ]
        for k, generator in enumerate(self.generators):
            label = generator.name or f"G{k + 1}"
            lines.append(
                f"    {label:4s} bus {generator.bus:2d}  H={generator.H:6.2f} s  "
                f"x'd={generator.xd_prime:.4f}  P={generator.P:.3f}  "
                f"E={self.emf_magnitude[k]:.4f}∠{math.degrees(self._delta0[k]):6.2f} deg"
            )
        return "\n".join(lines)


def load_case(path: str | Path) -> MultiMachineSystem:
    """YAML のケースファイルから多機系統を組み立てる。

    ケースファイルには母線・枝・負荷・発電機に加えて、事故前潮流解
    （各母線の電圧の大きさと位相）を含める。潮流計算そのものは
    本パッケージの対象外なので、解を数値として与える形にしている。
    """
    import yaml

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    voltages = {
        int(bus["id"]): bus["v"] * np.exp(1j * math.radians(bus["angle_deg"]))
        for bus in data["buses"]
    }
    network = MultiMachineNetwork(
        buses=[int(bus["id"]) for bus in data["buses"]],
        branches=[
            Branch(
                from_bus=int(item["from"]),
                to_bus=int(item["to"]),
                r=float(item.get("r", 0.0)),
                x=float(item["x"]),
                b=float(item.get("b", 0.0)),
            )
            for item in data["branches"]
        ],
        loads=[
            Load(bus=int(item["bus"]), P=float(item["p"]), Q=float(item["q"]))
            for item in data.get("loads", [])
        ],
        voltages=voltages,
    )
    generators = [
        GeneratorData(
            bus=int(item["bus"]),
            H=float(item["h"]),
            xd_prime=float(item["xd_prime"]),
            P=float(item["p"]),
            Q=float(item.get("q", 0.0)),
            D=float(item.get("d", 0.0)),
            name=str(item.get("name", "")),
        )
        for item in data["generators"]
    ]

    fault_data = data.get("fault", {})
    fault = FaultSchedule(
        t_fault=float(fault_data.get("t_fault", math.inf)),
        t_clear=float(fault_data.get("t_clear", math.inf)),
    )

    return MultiMachineSystem(
        network=network,
        generators=generators,
        fault=fault,
        faulted_buses=[int(b) for b in fault_data.get("buses", [])],
        tripped_branches=[
            (int(item["from"]), int(item["to"]))
            for item in fault_data.get("tripped_branches", [])
        ],
        base=SystemBase(
            frequency_hz=float(data.get("frequency_hz", 50.0)),
            s_base_mva=float(data.get("base_mva", 100.0)),
        ),
    )
