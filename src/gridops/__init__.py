"""gridops — 電力系統の運用と計画のシミュレーション教材パッケージ。

1 つのケース（既定は WSCC 9 母線系統）を、5 つのテーマ――潮流計算・経済
運用・起動停止計画・供給信頼度・系統安全性――で繰り返し解き直すための道具
一式である。**同じ系統を違う時間スケールで見ると何が問題になるか**を、
学生が自分の手で確かめられることを目的にしている。

設計の要点
----------
* 系統データは :class:`Case` に集約し、注入 :math:`(P^{sp}, Q^{sp})` を
  組み立てるのは :meth:`Case.bus_injection` **だけ**にしてある。負荷を
  負の発電として足し込む符号ミスを構造的に防ぐためである。
* :mod:`pulp` を import してよいのは :mod:`gridops.solvers` だけである。
  ソルバの発見・実行不可能時の診断・双対の符号の規約をそこ 1 箇所に
  集約する。問題を組み立てる下流は :func:`gridops.solvers.problem` /
  :func:`gridops.solvers.lp_sum` を使う。
* 安定度の教材 ``genstab`` は依存に入れていない。橋渡しをする
  :mod:`gridops.interop` は ``genstab`` の import を関数の中で行うので、
  ``genstab`` が無くても gridops のほかの機能はすべて動く。

名前の衝突について
------------------
潮流計算の ``solve`` と数理計画の ``solve`` は別物なので、素の ``solve``
はパッケージ直下に出していない。交流潮流は :func:`solve_powerflow`
（:func:`gridops.powerflow.solve` の別名）、数理計画は
:func:`gridops.solvers.solve` で呼ぶこと。同じ理由で作図の 21 個の
``plot_*`` も直下には出さず、:mod:`gridops.plotting` 越しに使う。

最小の使い方
------------
>>> import gridops
>>> case = gridops.load_case("wscc9")
>>> case.n_bus, case.n_branch, len(case.units)
(9, 9, 7)

交流潮流を Newton 法で解く（第 03 回）。

>>> flow = gridops.solve_powerflow(case)
>>> flow.converged, round(flow.losses, 4)
(True, 0.0464)

経済負荷配分を等増分燃料費法で解く（第 05 回）。

>>> plan = gridops.economic_dispatch(case, 315.0)
>>> round(plan.total_mw(), 6)
315.0

N-1 スクリーニングをかける（第 09 回）。**熱容量では健全なのに電圧の
下限を割る事故**が 1 件あり、これが「熱容量だけを見るスクリーニングは
絞り込みであって判断ではない」ことの実例である。

>>> report = gridops.screen_n1(case)
>>> report.is_secure
False
>>> [r.outage for r in report.results if r.thermal_secure and not r.voltage_secure]
[(4, 6)]
"""

from . import (
    adequacy,
    commitment,
    dc,
    dispatch,
    interop,
    plotting,
    powerflow,
    security,
    solvers,
    voltage,
    ybus,
)
from .adequacy import (
    CapacityOutageTable,
    MonteCarloResult,
    annual_load,
    capacity_outage_table,
    elcc,
    eue,
    load_duration_curve,
    lole,
    lolp,
    monte_carlo_adequacy,
)
from .case import Branch, Bus, BusType, Case, ReferenceSolution, Unit
from .commitment import (
    CommitmentResult,
    demand_profile,
    enumerate_commitment,
    marginal_prices,
    net_demand,
    priority_list,
    unit_commitment,
)
from .dc import DCSolution, dc_powerflow, lodf, ptdf, susceptance_matrix
from .dispatch import (
    DCOPFResult,
    DispatchResult,
    dc_opf,
    dispatch_with_losses,
    economic_dispatch,
    merit_order,
    penalty_factors,
)
from .interop import aggregate_plants, check_against_reference, to_genstab
from .loader import case_path, list_cases, load_case
from .plotting import use_gridops_style
from .powerflow import PowerFlowSolution, jacobian, jacobian_blocks, mismatch
from .powerflow import solve as solve_powerflow
from .security import (
    ContingencyResult,
    SCEDResult,
    SecurityReport,
    performance_index,
    sced,
    screen_n1,
)
from .voltage import (
    PVCurve,
    min_singular_value,
    pv_curve,
    two_bus_nose,
    two_bus_voltages,
    voltage_sensitivity,
)
from .ybus import bridges, build_ybus, incidence_matrix, islands

__version__ = "0.1.0"

__all__ = [
    "Branch",
    "Bus",
    "BusType",
    "CapacityOutageTable",
    "Case",
    "CommitmentResult",
    "ContingencyResult",
    "DCOPFResult",
    "DCSolution",
    "DispatchResult",
    "MonteCarloResult",
    "PVCurve",
    "PowerFlowSolution",
    "ReferenceSolution",
    "SCEDResult",
    "SecurityReport",
    "Unit",
    "adequacy",
    "aggregate_plants",
    "annual_load",
    "bridges",
    "build_ybus",
    "capacity_outage_table",
    "case_path",
    "check_against_reference",
    "commitment",
    "dc",
    "dc_opf",
    "dc_powerflow",
    "demand_profile",
    "dispatch",
    "dispatch_with_losses",
    "economic_dispatch",
    "elcc",
    "enumerate_commitment",
    "eue",
    "incidence_matrix",
    "interop",
    "islands",
    "jacobian",
    "jacobian_blocks",
    "list_cases",
    "load_case",
    "load_duration_curve",
    "lodf",
    "lole",
    "lolp",
    "marginal_prices",
    "merit_order",
    "min_singular_value",
    "mismatch",
    "monte_carlo_adequacy",
    "net_demand",
    "penalty_factors",
    "performance_index",
    "plotting",
    "powerflow",
    "priority_list",
    "ptdf",
    "pv_curve",
    "sced",
    "screen_n1",
    "security",
    "solve_powerflow",
    "solvers",
    "susceptance_matrix",
    "to_genstab",
    "two_bus_nose",
    "two_bus_voltages",
    "unit_commitment",
    "use_gridops_style",
    "voltage",
    "voltage_sensitivity",
    "ybus",
    "__version__",
]
