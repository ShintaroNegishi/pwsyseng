"""時間領域シミュレーション。

事故発生・除去の瞬間にネットワーク構成が不連続に変化するため、
:func:`simulate` は不連続点で積分区間を分割し、区間ごとにネットワーク
状態を固定して積分する。1 回の `solve_ivp` で不連続点をまたぐと、
ステップ幅制御が破綻して結果に誤差が乗る（教科書どおりの CCT が
得られなくなる）ので、この分割は精度上の要である。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .events import Stage
from .system import SMIBSystem

#: 脱調とみなす回転子位相角のしきい値 [rad]。
DEFAULT_ANGLE_LIMIT = 3.0 * np.pi


@dataclass
class SimulationResult:
    """シミュレーション結果。

    状態量と代数量のどちらも属性・添字の両方で取り出せる::

        result.delta        # 回転子位相角 [rad]
        result["Pe"]        # 電気出力 [p.u.]
    """

    t: np.ndarray                       #: 時刻 [s] (n_t,)
    x: np.ndarray                       #: 状態 (n_states, n_t)
    state_names: tuple[str, ...]
    outputs: dict[str, np.ndarray]      #: Pe, Vt, Pm, Efd
    system: SMIBSystem = field(repr=False)
    diverged: bool = False              #: 角度しきい値に達して打ち切ったか

    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.outputs:
            return self.outputs[name]
        if name in self.state_names:
            return self.x[self.state_names.index(name)]
        raise KeyError(
            f"'{name}' は存在しない。状態: {self.state_names}, "
            f"出力: {tuple(self.outputs)}"
        )

    def _state(self, name: str) -> np.ndarray:
        if name not in self.state_names:
            raise AttributeError(
                f"この系は状態 '{name}' を持たない（状態: {self.state_names}）。"
            )
        return self.x[self.state_names.index(name)]

    @property
    def has_rotor_angle(self) -> bool:
        """回転子位相角を状態に持つか（孤立系では持たない）。"""
        return "delta" in self.state_names

    @property
    def delta(self) -> np.ndarray:
        """回転子位相角 [rad]。"""
        return self._state("delta")

    @property
    def delta_deg(self) -> np.ndarray:
        """回転子位相角 [deg]。"""
        return np.degrees(self.delta)

    @property
    def omega(self) -> np.ndarray:
        """速度偏差 [p.u.]。"""
        return self._state("omega")

    @property
    def frequency_hz(self) -> np.ndarray:
        """回転子周波数 [Hz]。"""
        return self.system.base.frequency_hz * (1.0 + self.omega)

    @property
    def Pe(self) -> np.ndarray:
        """電気出力 [p.u.]。"""
        return self.outputs["Pe"]

    @property
    def Vt(self) -> np.ndarray:
        """端子電圧 [p.u.]。"""
        return self.outputs["Vt"]

    @property
    def Pm(self) -> np.ndarray:
        """機械入力 [p.u.]。"""
        return self.outputs["Pm"]

    @property
    def Efd(self) -> np.ndarray:
        """界磁電圧 [p.u.]。"""
        return self.outputs["Efd"]

    # ------------------------------------------------------------------
    @property
    def max_angle_deviation(self) -> float:
        """初期位相角からの最大偏差 [rad]。"""
        if not self.has_rotor_angle:
            raise AttributeError(
                "この系は回転子位相角を持たないため、位相角偏差を定義できない。"
            )
        return float(np.max(np.abs(self.delta - self.system.operating_point.delta)))

    def is_stable(self, angle_limit: float = np.pi) -> bool:
        """過渡安定かどうかを判定する。

        初期位相角からの偏差が ``angle_limit`` を超えず、かつ計算が
        打ち切られていなければ安定とみなす。既定のしきい値 180° は
        第 1 波での脱調を検出する実用的な基準である。

        Notes
        -----
        これは工学的な便宜上の判定であり、厳密な安定判別ではない。
        制動が非常に弱い場合、大きく振れてから戻る（安定だが判定は
        不安定になる）ことがあるため、波形も必ず目視で確認すること。
        """
        if self.diverged:
            return False
        assess = getattr(self.system, "assess_stability", None)
        if assess is not None:
            # 多機系統など、系ごとに固有の判定基準を持つ場合はそちらに従う。
            return bool(assess(self, angle_limit))
        if not self.has_rotor_angle:
            # 位相角を持たない系（孤立系の周波数応答など）では、
            # 状態が有界であることをもって安定とみなす。
            return bool(np.all(np.isfinite(self.x)))
        if not np.all(np.isfinite(self.delta)):
            return False
        return self.max_angle_deviation <= angle_limit

    def to_dataframe(self):
        """pandas.DataFrame に変換する（表として眺めたいとき用）。"""
        import pandas as pd

        data = {"t": self.t}
        for i, name in enumerate(self.state_names):
            data[name] = self.x[i]
        data.update(self.outputs)
        return pd.DataFrame(data)


def simulate(
    system: SMIBSystem,
    t_end: float = 10.0,
    dt: float = 0.005,
    *,
    x0: np.ndarray | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    angle_limit: float = DEFAULT_ANGLE_LIMIT,
    max_step: float | None = None,
) -> SimulationResult:
    """時間領域シミュレーションを実行する。

    Parameters
    ----------
    system:
        シミュレーション対象。
    t_end:
        終了時刻 [s]。
    dt:
        出力の時間刻み [s]。積分器の内部ステップとは無関係で、
        結果を書き出す間隔にすぎない。
    x0:
        初期状態。省略すると事故前定常状態を使う。
    method:
        `scipy.integrate.solve_ivp` の積分法。動揺方程式は非スティッフ
        なので既定の ``"RK45"`` で足りるが、時定数の小さい励磁系を
        入れた場合は ``"Radau"`` や ``"LSODA"`` が速いことがある。
    rtol, atol:
        積分の許容誤差。教材で数値誤差が議論を汚さないよう既定値を
        厳しめに取っている。
    angle_limit:
        回転子位相角がこの値を超えたら発散とみなして計算を打ち切る [rad]。
        脱調時の無駄な計算と桁あふれを防ぐ。
    max_step:
        積分器の最大ステップ幅 [s]。

    Returns
    -------
    SimulationResult
    """
    if dt <= 0.0:
        raise ValueError(f"時間刻み dt は正でなければならない (dt={dt})。")
    if t_end <= 0.0:
        raise ValueError(f"終了時刻 t_end は正でなければならない (t_end={t_end})。")

    x_current = system.initial_state() if x0 is None else np.asarray(x0, dtype=float)
    if x_current.size != system.n_states:
        raise ValueError(
            f"初期状態の長さ {x_current.size} が系の状態数 {system.n_states} と一致しない。"
        )

    t_grid = np.arange(0.0, t_end + 0.5 * dt, dt)
    boundaries = [0.0, *system.switching_times(t_end), t_end]

    angle_index = (
        system.state_names.index("delta") if "delta" in system.state_names else None
    )

    def make_angle_event(limit: float):
        """|δ| が limit に達したら積分を止めるイベント関数。

        回転子位相角を持たない系（孤立系）では発散判定の意味がないので
        イベントを設定しない。
        """
        if angle_index is None:
            return None

        def event(t, x):
            return limit - abs(x[angle_index])

        event.terminal = True
        event.direction = -1.0
        return event

    t_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    diverged = False

    for i in range(len(boundaries) - 1):
        t_start, t_stop = boundaries[i], boundaries[i + 1]
        if t_stop <= t_start:
            continue

        # 区間内でネットワーク状態は一定。中点で判定して固定する。
        stage: Stage = system.stage_at(0.5 * (t_start + t_stop))

        is_last = i == len(boundaries) - 2
        if is_last:
            mask = (t_grid >= t_start) & (t_grid <= t_stop)
        else:
            mask = (t_grid >= t_start) & (t_grid < t_stop)
        t_eval = t_grid[mask]

        options = {} if max_step is None else {"max_step": max_step}
        sol = solve_ivp(
            fun=lambda t, x, _stage=stage: system.derivatives(t, x, _stage),
            t_span=(t_start, t_stop),
            y0=x_current,
            t_eval=t_eval if t_eval.size else None,
            method=method,
            rtol=rtol,
            atol=atol,
            events=make_angle_event(angle_limit),
            dense_output=False,
            **options,
        )
        if not sol.success and sol.status != 1:
            raise RuntimeError(f"積分に失敗した（区間 [{t_start}, {t_stop}]）: {sol.message}")

        if sol.t.size:
            t_parts.append(sol.t)
            x_parts.append(sol.y)

        if sol.status == 1:  # 角度しきい値に到達（脱調）
            diverged = True
            if sol.t_events[0].size and sol.y_events[0].size:
                t_parts.append(sol.t_events[0])
                x_parts.append(sol.y_events[0].T)
            break

        x_current = sol.y[:, -1] if sol.t.size else x_current
        # t_eval に区間終端が含まれない場合、次区間の初期値は
        # 積分器が到達した最終状態を使う必要がある。
        if not is_last and (t_eval.size == 0 or t_eval[-1] < t_stop):
            sol_end = solve_ivp(
                fun=lambda t, x, _stage=stage: system.derivatives(t, x, _stage),
                t_span=(t_parts[-1][-1] if t_parts else t_start, t_stop),
                y0=x_current,
                method=method,
                rtol=rtol,
                atol=atol,
                **options,
            )
            if sol_end.success and sol_end.y.size:
                x_current = sol_end.y[:, -1]

    if not t_parts:
        raise RuntimeError("シミュレーション結果が空になった。t_end と dt を確認すること。")

    t_all = np.concatenate(t_parts)
    x_all = np.concatenate(x_parts, axis=1)

    # 区間境界で時刻が重複することがあるため取り除く。
    _, unique_idx = np.unique(np.round(t_all, 12), return_index=True)
    t_all = t_all[unique_idx]
    x_all = x_all[:, unique_idx]

    outputs = system.algebraic_outputs(t_all, x_all)

    return SimulationResult(
        t=t_all,
        x=x_all,
        state_names=system.state_names,
        outputs=outputs,
        system=system,
        diverged=diverged,
    )
