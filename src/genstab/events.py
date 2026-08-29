"""事故（擾乱）のスケジュール。

過渡安定性の計算では、事故発生・事故除去の瞬間にネットワーク構成が
不連続に変化する。この不連続点を数値積分器にまたがせると誤差が乗るため、
`simulate` は本モジュールが返す切替時刻で積分区間を分割する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Stage(Enum):
    """ネットワークの状態。"""

    PRE = "pre"      #: 事故前（定常）
    FAULT = "fault"  #: 事故中
    POST = "post"    #: 事故除去後


@dataclass(frozen=True)
class FaultSchedule:
    """三相地絡と事故除去のスケジュール。

    Parameters
    ----------
    t_fault:
        事故発生時刻 [s]。
    t_clear:
        事故除去時刻 [s]。`math.inf` を与えると永久故障（除去しない）。

    Notes
    -----
    事故除去時間（clearing time）は ``t_clear - t_fault`` である。
    過渡安定性の設計ではこの値が臨界事故除去時間 (CCT) を下回るか
    どうかが問題になる。
    """

    t_fault: float = 1.0
    t_clear: float = 1.2

    def __post_init__(self) -> None:
        if self.t_clear < self.t_fault:
            raise ValueError(
                f"事故除去時刻 t_clear={self.t_clear} が "
                f"事故発生時刻 t_fault={self.t_fault} より前になっている。"
            )

    @property
    def clearing_time(self) -> float:
        """事故除去時間 [s]。"""
        return self.t_clear - self.t_fault

    def stage(self, t: float) -> Stage:
        """時刻 `t` におけるネットワークの状態を返す。"""
        if t < self.t_fault:
            return Stage.PRE
        if t < self.t_clear:
            return Stage.FAULT
        return Stage.POST

    def switching_times(self, t_end: float) -> list[float]:
        """`(0, t_end)` の範囲に含まれる不連続点を昇順で返す。"""
        times = [t for t in (self.t_fault, self.t_clear)
                 if math.isfinite(t) and 0.0 < t < t_end]
        return sorted(set(times))

    @classmethod
    def none(cls) -> "FaultSchedule":
        """事故を起こさないスケジュール（定常応答の確認用）。"""
        return cls(t_fault=math.inf, t_clear=math.inf)
