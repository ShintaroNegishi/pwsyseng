#!/usr/bin/env python
"""線路の熱容量 `rate_a` / `rate_b` を N-1 潮流から逆算する。

なぜこの道具が要るのか
----------------------
WSCC 9 母線系統の原典（Anderson & Fouad）には線路の熱容量が **無い**。
そのままでは「制約を守っているか」という問い自体が立たないので、同梱
ケースの `rate_a` / `rate_b` は著者が設計した自作の値である。設計の狙いは
教材上の要求そのもので、次の 2 つを同時に満たすことである。

1. **事故前は全枝が健全**であること（そうでないと N-1 以前の話になる）
2. **N-1 で拘束するのが少数の枝だけ**であること（すべてが違反すると
   「どこが弱いか」が読めず、1 本も違反しないと第 09 回の題材が消える）

熱容量は本来、導体の温度上昇から決まる物理量である。それを潮流から
逆算するのは **教材としての割り切り**であり、実系統の設計手順ではない。
この道具は「その割り切りを再現可能にする」ためにあり、ケースを作り替えた
教員が同じ物語を組み直せるようにするのが目的である。

やっていること
--------------
交流潮流を、事故前と N-1 の各事故について解き、枝ごとに

* `base`  : 事故前の :math:`|S|`
* `worst` : 全事故を通じた最悪の :math:`|S|`

を集める。そのうえで

.. math::

    \\mathrm{rate\\_a} = \\lceil \\mathrm{base} \\cdot m_a \\rceil_{step}, \\qquad
    \\mathrm{rate\\_b} = \\lceil \\mathrm{worst} \\cdot m_b \\rceil_{step}

を候補として提案する。余裕率 :math:`m_a` / :math:`m_b` を 1 より小さく
取ると、その枝は**わざと**拘束させられる。同梱ケースの 5-7 と 7-8 が
N-1 で 112% になるのはこの操作の結果である。

橋（開放すると系統が分離する枝）は N-1 の候補から外す。分離した系統の
潮流は解けず、「熱容量が足りない」問題ではないためである。

使い方::

    python tools/design_ratings.py                 # 同梱ケース
    python tools/design_ratings.py wscc9           # 名前を指定
    python tools/design_ratings.py --margin-a 1.3 --margin-b 1.05
    python tools/design_ratings.py --check         # 現在値の検証だけ

`--check` は提案をせず、**いま入っている熱容量で狙いどおりになっているか**
だけを報告する。ケースファイルを書き換えたあとの確認に使う。
"""

from __future__ import annotations

import argparse
import math
import sys

from gridops.loader import list_cases, load_case
from gridops.powerflow import solve
from gridops.ybus import bridges

#: 提案値を丸める刻み [p.u.]。ケースファイルに書く数を読みやすく保つため。
DEFAULT_STEP = 0.05

#: 事故前・事故後それぞれの既定の余裕率。
DEFAULT_MARGIN_A = 1.30
DEFAULT_MARGIN_B = 1.00


def branch_flows(case, key=None):
    """交流潮流を解いて、枝ごとの :math:`|S|` [p.u.] を返す。

    Parameters
    ----------
    case:
        対象のケース。
    key:
        開放する枝。``None`` なら事故前。

    Returns
    -------
    dict[tuple[int, int], float] | None
        収束しなかったときは ``None``。1 件の未収束で掃引全体を止めない。

    Notes
    -----
    ``Case.without_branch(key, keep_generation=True)`` を使うのは、既定の
    ``keep_generation=False`` だと参照解が落ちて **発電ゼロ**の別系統を
    解いてしまうためである（slack 1 台で全負荷を賄う問題になり、WSCC 9
    母線では収束しない）。
    """
    target = case if key is None else case.without_branch(key, keep_generation=True)
    try:
        solution = solve(target)
    except RuntimeError:
        return None
    if not solution.converged:
        return None
    return solution.apparent_flows()


def survey(case):
    """事故前と N-1 の枝潮流を集める。

    Returns
    -------
    tuple
        ``(base, worst, worst_outage, skipped, failed)``。``base`` と
        ``worst`` は枝キーから :math:`|S|` [p.u.] への写像、
        ``worst_outage`` は最悪値を与えた事故、``skipped`` は橋として
        外した枝、``failed`` は収束しなかった事故である。
    """
    base = branch_flows(case)
    if base is None:
        raise SystemExit("事故前の潮流が収束しない。先に tools/check_case.py を通すこと。")

    keys = [branch.key() for branch in case.branches]
    bridge_keys = set(bridges(case))
    candidates = [key for key in keys if key not in bridge_keys]

    worst = dict(base)
    worst_outage = {key: None for key in keys}
    failed: list[tuple[int, int]] = []

    for outage in candidates:
        flows = branch_flows(case, outage)
        if flows is None:
            failed.append(outage)
            continue
        for key, value in flows.items():
            if value > worst[key]:
                worst[key] = value
                worst_outage[key] = outage

    skipped = [key for key in keys if key in bridge_keys]
    return base, worst, worst_outage, skipped, failed


def proposal(value: float, margin: float, step: float) -> float:
    """余裕率をかけて刻みで切り上げる。"""
    return math.ceil(value * margin / step) * step


def report(name: str, *, margin_a: float, margin_b: float, step: float, check: bool) -> int:
    """1 つのケースについて設計値を提案し、拘束する枝の数を返す。"""
    case = load_case(name)
    base, worst, worst_outage, skipped, failed = survey(case)

    print(f"Case '{case.name}'")
    if skipped:
        print(f"  N-1 の候補から外した橋: {skipped}（開放すると系統が分離する）")
    if failed:
        print(f"  収束しなかった事故: {failed}（熱容量ではなく解の存在の問題）")

    header = (
        f"  {'枝':>6} {'事故前|S|':>10} {'N-1 最悪':>10} {'最悪の事故':>12} "
        f"{'rate_a':>8} {'rate_b':>8} {'N-1 負荷率':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    binding: list[tuple[int, int]] = []
    non_binding_max = 0.0

    for branch in case.branches:
        key = branch.key()
        if check:
            rate_a, rate_b = branch.rate_a, branch.rate_b
        else:
            rate_a = proposal(base[key], margin_a, step)
            rate_b = proposal(worst[key], margin_b, step)

        loading = worst[key] / rate_b if rate_b > 0 else math.inf
        outage = worst_outage[key]
        label = "なし" if outage is None else f"{outage[0]}-{outage[1]}"
        print(
            f"  {branch.label:>6} {base[key]:10.4f} {worst[key]:10.4f} {label:>12} "
            f"{rate_a:8.2f} {rate_b:8.2f} {100.0 * loading:9.1f}%"
        )
        if loading > 1.0:
            binding.append(key)
        else:
            non_binding_max = max(non_binding_max, loading)

    print()
    if binding:
        print(f"  N-1 で拘束する枝: {sorted(binding)}（{len(binding)} 本）")
        print(f"  拘束しない枝の最大負荷率: {100.0 * non_binding_max:.1f}%")
        print(
            "  この 2 つの数が離れているほど「どこが弱いか」が一意に読める。"
            "近すぎるなら余裕率を調整すること。"
        )
    else:
        print(
            "  N-1 で拘束する枝が 1 本もない。このままでは第 09 回の題材が"
            "成立しないので、--margin-b を 1.0 より小さくして"
            "わざと拘束させること。"
        )
    print()
    return len(binding)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="ケース名またはパス（既定は同梱ケース全部）")
    parser.add_argument(
        "--margin-a", type=float, default=DEFAULT_MARGIN_A, help="事故前の余裕率"
    )
    parser.add_argument(
        "--margin-b", type=float, default=DEFAULT_MARGIN_B, help="事故後の余裕率"
    )
    parser.add_argument("--step", type=float, default=DEFAULT_STEP, help="丸めの刻み [p.u.]")
    parser.add_argument(
        "--check",
        action="store_true",
        help="提案せず、いま入っている熱容量で狙いどおりかだけを見る",
    )
    args = parser.parse_args(argv)

    if args.step <= 0:
        parser.error("--step は正の数であること")

    targets = args.cases or list_cases()
    for target in targets:
        report(
            target,
            margin_a=args.margin_a,
            margin_b=args.margin_b,
            step=args.step,
            check=args.check,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
