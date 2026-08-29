"""genstab — 同期発電機の安定性シミュレーション教材パッケージ。

過渡安定性（大擾乱に対する応答）と定態安定性（微小擾乱に対する応答）を、
同じ発電機モデルに対して一貫した枠組みで扱えるようにしたもの。
制御器（AVR・ガバナ・PSS・状態フィードバック）はすべてオプションで、
接続しなければ素の動揺方程式に縮退する。

最小の使い方
------------
>>> from genstab import ClassicalMachine, SMIBNetwork, FaultSchedule, SMIBSystem, simulate
>>> machine = ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
>>> network = SMIBNetwork(x_pre=0.4, x_fault=float("inf"), x_post=0.6)
>>> fault = FaultSchedule(t_fault=1.0, t_clear=1.15)
>>> system = SMIBSystem(machine, network, fault, Pe0=0.8)
>>> result = simulate(system, t_end=5.0)
>>> bool(result.is_stable())
True
"""

from . import eac, frequency, linearize, plotting, smallsignal
from .events import FaultSchedule, Stage
from .frequency import IsolatedSystem, NoLoadChange, RampLoad, StepLoad
from .controllers import (
    Governor,
    IEEEType1Exciter,
    LoadFrequencyControl,
    PowerSystemStabilizer,
    ProportionalGovernor,
    SimpleExciter,
    StateFeedback,
    design_lqr,
    design_pole_placement,
    design_pss,
)
from .machines import ClassicalMachine, Machine, OneAxisMachine
from .network import ElectricalSolution, SMIBNetwork
from .simulate import SimulationResult, simulate
from .system import OperatingPoint, SMIBSystem
from .units import DEFAULT_BASE, SystemBase, deg, rad

__version__ = "0.1.0"

__all__ = [
    "ClassicalMachine",
    "eac",
    "frequency",
    "linearize",
    "plotting",
    "smallsignal",
    "DEFAULT_BASE",
    "ElectricalSolution",
    "FaultSchedule",
    "Governor",
    "IEEEType1Exciter",
    "IsolatedSystem",
    "Machine",
    "LoadFrequencyControl",
    "NoLoadChange",
    "OneAxisMachine",
    "OperatingPoint",
    "PowerSystemStabilizer",
    "ProportionalGovernor",
    "SMIBNetwork",
    "SMIBSystem",
    "RampLoad",
    "SimpleExciter",
    "StateFeedback",
    "StepLoad",
    "SimulationResult",
    "Stage",
    "SystemBase",
    "deg",
    "design_lqr",
    "design_pole_placement",
    "design_pss",
    "rad",
    "simulate",
    "__version__",
]
