"""発電機・ネットワーク・制御器を束ねた系。

:class:`SMIBSystem` が状態ベクトルの連結を担当する。発電機モデルと
各制御器はそれぞれ自分の状態微分を返すだけでよく、全体の状態は

    x = [ 発電機の状態 | 制御器1の状態 | 制御器2の状態 | ... ]

の順に並ぶ。制御器を渡さなければ機械入力と界磁電圧は定常値に固定され、
系は素の動揺方程式に縮退する。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace as _dataclass_replace

import numpy as np

from .controllers.base import Controller, ControllerKind, Measurement
from .events import FaultSchedule, Stage
from .machines.base import Machine
from .network import ElectricalSolution, SMIBNetwork
from .units import DEFAULT_BASE, SystemBase


@dataclass
class OperatingPoint:
    """事故前定常状態（線形化や初期値設定の基準となる動作点）。"""

    x: np.ndarray       #: 全状態の初期値
    delta: float        #: 回転子位相角 [rad]
    Pe: float           #: 電気出力 [p.u.]
    Vt: float           #: 端子電圧 [p.u.]
    Pm: float           #: 機械入力 [p.u.]
    Efd: float          #: 界磁電圧 [p.u.]


@dataclass
class SMIBSystem:
    """1 機無限大母線系統。

    Parameters
    ----------
    machine:
        発電機モデル。
    network:
        外部系統。
    fault:
        事故スケジュール。省略すると事故なし（定常）。
    controllers:
        接続する制御器のリスト。空なら制御なし。
    Pe0:
        事故前の送電電力 [p.u.]。この値から初期位相角が決まる。
    base:
        系統の基準値。

    Examples
    --------
    制御なし（素の動揺方程式）::

        system = SMIBSystem(machine, network, fault)

    AVR と PSS を追加::

        system = SMIBSystem(machine, network, fault,
                            controllers=[avr, pss])
    """

    machine: Machine
    network: SMIBNetwork
    fault: FaultSchedule = field(default_factory=FaultSchedule.none)
    controllers: list[Controller] = field(default_factory=list)
    Pe0: float = 0.8
    base: SystemBase = DEFAULT_BASE

    def __post_init__(self) -> None:
        # ネットワークは複製してから背後リアクタンスを登録する。
        # attach() は渡されたオブジェクトを書き換えるので、同じ
        # SMIBNetwork を x'd の異なる発電機で使い回すと、先に作った系の
        # 動特性まで後から変わってしまう（実測で残差が 1e-17 から 1.8e-2 へ
        # 悪化することを確認している）。複製しておけば、利用者は同じ
        # ネットワーク定義を安心して使い回せる。
        self.network = _dataclass_replace(self.network)
        self.network.attach(self.machine.x_internal)
        self._check_controller_reuse()
        self._validate_controllers()
        self._build_slices()
        self.operating_point = self._solve_operating_point()

    # ------------------------------------------------------------------
    # 構成の検証と状態スライスの構築
    # ------------------------------------------------------------------
    def _check_controller_reuse(self) -> None:
        """同じ制御器インスタンスの使い回しを検出する。

        制御器は :meth:`Controller.initialize` で基準値（AVR なら v_ref、
        状態フィードバックなら動作点）を自分の中に持つ。同じインスタンスを
        別の系に接続すると、その基準値が上書きされ、先に作った系が
        静かに壊れる。ネットワークと違って制御器は利用者が接続後に
        参照する（``avr.v_ref`` を表示するなど）ため複製はせず、
        警告で知らせる方針をとる。
        """
        for controller in self.controllers:
            owner = getattr(controller, "_bound_to", None)
            if owner is not None and owner is not self:
                warnings.warn(
                    f"{type(controller).__name__} のインスタンスが既に別の系に"
                    " 接続されている。制御器は接続時に基準値を自分の中に持つため、"
                    " 使い回すと先に作った系の動作点が壊れる。"
                    " 系ごとに新しいインスタンスを作ること。",
                    UserWarning,
                    stacklevel=4,
                )
            controller._bound_to = self

    def _validate_controllers(self) -> None:
        exciters = [c for c in self.controllers if c.kind is ControllerKind.EXCITER]
        if len(exciters) > 1:
            raise ValueError(
                f"励磁系 (exciter) は 1 台までしか接続できない（{len(exciters)} 台指定された）。"
            )
        stabilizers = [
            c for c in self.controllers if c.kind is ControllerKind.STABILIZER
        ]
        if stabilizers and not exciters:
            raise ValueError(
                "PSS (stabilizer) は励磁系 (exciter) への補助信号として働くため、"
                "AVR と一緒に接続する必要がある。"
            )
        if exciters and not self.machine.responds_to_excitation:
            warnings.warn(
                f"{type(self.machine).__name__} は界磁回路の状態量を持たないため、"
                "励磁系を接続しても内部起電力は変化しない。AVR の効果を見るには"
                " OneAxisMachine 以上のモデルを使うこと。",
                UserWarning,
                stacklevel=3,
            )

        self.exciter = exciters[0] if exciters else None
        self.stabilizers = stabilizers
        self.governors = [
            c for c in self.controllers if c.kind is ControllerKind.GOVERNOR
        ]

    def _build_slices(self) -> None:
        """状態ベクトル内での各要素の位置を決める。"""
        start = self.machine.n_states
        self._machine_slice = slice(0, start)
        self._controller_slices: list[slice] = []
        for controller in self.controllers:
            stop = start + controller.n_states
            self._controller_slices.append(slice(start, stop))
            start = stop
        self.n_states = start

        names = list(self.machine.state_names)
        for controller in self.controllers:
            prefix = type(controller).__name__
            names.extend(f"{prefix}.{n}" for n in controller.state_names)
        self.state_names = tuple(names)

        self._exciter_slice = (
            self._controller_slices[self.controllers.index(self.exciter)]
            if self.exciter is not None
            else slice(0, 0)
        )

    # ------------------------------------------------------------------
    # 定常状態
    # ------------------------------------------------------------------
    def _solve_operating_point(self) -> OperatingPoint:
        x_machine, Pm0, Efd0 = self.machine.initial_state(self.network, self.Pe0)
        delta0 = self.machine.rotor_angle(x_machine)
        emf0 = self.machine.internal_emf(x_machine)
        solution = self.network.solve(Stage.PRE, emf0, delta0)

        meas0 = Measurement(
            t=0.0,
            delta=delta0,
            omega=0.0,
            Pe=solution.Pe,
            Vt=solution.Vt,
            x_machine=x_machine,
        )

        # 各制御器を「定常状態で出力が定常値に一致する」ように初期化する。
        # これを怠ると開始直後に事故と無関係な過渡応答が出る。
        steady_values = {
            ControllerKind.EXCITER: Efd0,
            ControllerKind.GOVERNOR: 0.0,   # 機械入力の偏差として扱う
            ControllerKind.STABILIZER: 0.0,  # 補助信号は定常でゼロ
        }
        controller_states = [
            np.atleast_1d(
                np.asarray(c.initialize(meas0, steady_values[c.kind]), dtype=float)
            ).ravel()
            for c in self.controllers
        ]
        for controller, xc in zip(self.controllers, controller_states):
            if xc.size != controller.n_states:
                raise ValueError(
                    f"{type(controller).__name__}.initialize() が返した状態の長さ "
                    f"{xc.size} が n_states={controller.n_states} と一致しない。"
                )

        x0 = np.concatenate([x_machine, *controller_states]) if controller_states \
            else np.array(x_machine, dtype=float)

        self._Pm0 = float(Pm0)
        self._Efd0 = float(Efd0)

        return OperatingPoint(
            x=x0,
            delta=delta0,
            Pe=solution.Pe,
            Vt=solution.Vt,
            Pm=float(Pm0),
            Efd=float(Efd0),
        )

    def initial_state(self) -> np.ndarray:
        """事故前定常状態の状態ベクトルを返す。"""
        return self.operating_point.x.copy()

    # ------------------------------------------------------------------
    # 状態方程式
    # ------------------------------------------------------------------
    def _evaluate(
        self, t: float, x: np.ndarray, stage: Stage
    ) -> tuple[ElectricalSolution, Measurement, float, float, float]:
        """ネットワーク解と制御器出力をまとめて求める。"""
        x_machine = x[self._machine_slice]
        delta = self.machine.rotor_angle(x_machine)
        emf = self.machine.internal_emf(x_machine)
        omega = self.machine.speed(x_machine)

        solution = self.network.solve(stage, emf, delta)
        meas = Measurement(
            t=t, delta=delta, omega=omega, Pe=solution.Pe, Vt=solution.Vt,
            x_machine=x_machine,
        )

        # 1. PSS（補助信号）→ 2. AVR（界磁電圧）→ 3. ガバナ（機械入力）の順に評価する。
        aux = 0.0
        for controller, sl in zip(self.controllers, self._controller_slices):
            if controller.kind is ControllerKind.STABILIZER:
                aux += controller.output(t, x[sl], meas)

        Efd = self._Efd0
        if self.exciter is not None:
            Efd = self.exciter.output(t, x[self._exciter_slice], meas, aux=aux)

        Pm = self._Pm0
        for controller, sl in zip(self.controllers, self._controller_slices):
            if controller.kind is ControllerKind.GOVERNOR:
                Pm += controller.output(t, x[sl], meas)

        return solution, meas, Pm, Efd, aux

    def derivatives(
        self, t: float, x: np.ndarray, stage: Stage | None = None
    ) -> np.ndarray:
        """状態微分 dx/dt を返す。

        Parameters
        ----------
        stage:
            ネットワークの状態を明示的に指定する。``None`` なら事故
            スケジュールから決める。:func:`~genstab.simulate.simulate` は
            積分区間ごとに固定した値を渡す（不連続点を積分器にまたがせ
            ないため）。
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_states,):
            raise ValueError(
                f"状態ベクトルの形状 {x.shape} が系の状態数 "
                f"({self.n_states},) と一致しない。"
            )
        if stage is None:
            stage = self.fault.stage(t)

        solution, meas, Pm, Efd, aux = self._evaluate(t, x, stage)

        dx = np.empty_like(x)
        dx[self._machine_slice] = self.machine.derivatives(
            x[self._machine_slice], solution, Pm, Efd, self.base
        )

        for controller, sl in zip(self.controllers, self._controller_slices):
            if controller.n_states == 0:
                continue
            kwargs = {"aux": aux} if controller.kind is ControllerKind.EXCITER else {}
            dx[sl] = controller.derivatives(t, x[sl], meas, **kwargs)

        return dx

    # ------------------------------------------------------------------
    # 事後処理
    # ------------------------------------------------------------------
    def switching_times(self, t_end: float) -> list[float]:
        """積分区間を分割すべき時刻（事故発生・除去）。"""
        return self.fault.switching_times(t_end)

    def stage_at(self, t: float) -> Stage:
        """時刻 t におけるネットワーク状態。"""
        return self.fault.stage(t)

    def algebraic_outputs(
        self, t: np.ndarray, x: np.ndarray
    ) -> dict[str, np.ndarray]:
        """軌道全体について代数量（Pe, Vt, Pm, Efd）を再計算する。

        Parameters
        ----------
        t:
            時刻の配列 (n_t,)。
        x:
            状態の配列 (n_states, n_t)。
        """
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        n = t.size
        out = {k: np.empty(n) for k in ("Pe", "Vt", "Pm", "Efd")}
        for i in range(n):
            stage = self.stage_at(t[i])
            solution, _, Pm, Efd, _aux = self._evaluate(t[i], x[:, i], stage)
            out["Pe"][i] = solution.Pe
            out["Vt"][i] = solution.Vt
            out["Pm"][i] = Pm
            out["Efd"][i] = Efd
        return out

    def describe(self) -> str:
        """構成の要約を返す（notebook で構成を確認するため）。"""
        lines = [
            f"SMIBSystem  (状態数 {self.n_states})",
            f"  発電機   : {type(self.machine).__name__} "
            f"(H={self.machine.H}, D={self.machine.D}, x_d'={self.machine.x_internal})",
            f"  ネットワーク: x_pre={self.network.x_pre}, "
            f"x_fault={self.network.x_fault}, x_post={self.network.x_post}, "
            f"V_inf={self.network.V_inf}",
            f"  事故     : t_fault={self.fault.t_fault}, t_clear={self.fault.t_clear} "
            f"(除去時間 {self.fault.clearing_time:.3f} s)",
            f"  動作点   : Pe0={self.operating_point.Pe:.4f}, "
            f"delta0={np.degrees(self.operating_point.delta):.2f} deg, "
            f"Vt0={self.operating_point.Vt:.4f}",
        ]
        if self.controllers:
            lines.append("  制御器   :")
            for c in self.controllers:
                lines.append(f"    - {type(c).__name__} [{c.kind.value}]")
        else:
            lines.append("  制御器   : なし（素の動揺方程式）")
        return "\n".join(lines)
