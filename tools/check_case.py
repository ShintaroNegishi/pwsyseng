#!/usr/bin/env python
"""同梱ケース（または指定したファイル）の整合性を確認する。

数値的に解けないときの原因は、ソルバの設定よりもデータとトポロジーの
矛盾にあることのほうが多い。ソルバを疑う前にこれを通すこと。

使い方::

    python tools/check_case.py              # 同梱ケースをすべて
    python tools/check_case.py wscc9        # 名前を指定
    python tools/check_case.py path/to/my_case.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from gridops.loader import list_cases, load_case


def rounding_bound(Y, v, digits: int) -> float:
    """掲載桁数の丸めだけで生じる注入の残差の上界 [p.u.] を返す。

    参照解は教科書に載っている桁数（``digits``）で丸められているので、
    そこから組み直した注入 :math:`S = V \\odot \\overline{YV}` は、
    線路データが完全に正しくても残差を持つ。閾値を定数で直書きすると
    **正しいデータを「壊れている」と報告する**（実際 WSCC 9 母線では
    残差 1.1e-03 に対して 1e-3 の直書きが誤警報を出していた）。

    :math:`V` の摂動 :math:`\\delta V` に対する 1 次の変化は

    .. math::

        \\delta S \\approx \\delta V \\odot \\overline{YV}
                   + V \\odot \\overline{Y\\,\\delta V}

    なので、:math:`|\\delta V| \\le \\tfrac12 10^{-digits}` を代入して

    .. math::

        |\\delta S| \\lesssim \\tfrac12 10^{-digits}
            \\left( \\max|YV| + \\max|V| \\cdot \\|Y\\|_\\infty \\right)

    を上界に取る。位相の丸め（4 桁なら 8.7e-07 rad）は大きさの丸めより
    2 桁小さいので、大きさ側だけで押さえておけば足りる。
    """
    import numpy as np

    step = 0.5 * 10.0 ** (-int(digits))
    y_inf_norm = float(np.abs(Y).sum(axis=1).max())
    return step * (float(np.abs(Y @ v).max()) + float(np.abs(v).max()) * y_inf_norm)


def report(source: str | Path) -> int:
    """1 つのケースを検査して、問題の件数を返す。"""
    case = load_case(source)
    print(case.describe())

    problems = case.check()
    if problems:
        print(f"  問題 {len(problems)} 件:")
        for message in problems:
            print(f"    - {message}")
    else:
        print("  問題なし。")

    # 未知数と方程式の数合わせを明示する。第 01 回の主題そのもの。
    slack, pv, pq = case.type_indices()
    equations = (len(pv) + len(pq)) + len(pq)
    print(
        f"  未知数 2*{len(pq)} + {len(pv)} = {case.n_unknowns()}"
        f"  /  方程式 ({len(pv)}+{len(pq)}) + {len(pq)} = {equations}"
    )
    if case.n_unknowns() != equations:
        print("    未知数と方程式の数が合っていない。母線種別を確認すること。")
        problems = list(problems) + ["未知数と方程式の数が不一致"]

    # 参照解があれば、注入の整合性まで見る。
    if case.reference is not None:
        try:
            from gridops.ybus import build_ybus
        except ImportError:      # pragma: no cover - 実装前でも本体は動く
            pass
        else:
            import numpy as np

            Y = build_ybus(case)
            v = case.reference.voltage
            s = v * np.conj(Y @ v)
            expected_p, expected_q = case.bus_injection()
            error = np.max(np.abs(s - (expected_p + 1j * expected_q)))
            tolerance = rounding_bound(Y, v, case.reference.digits)
            print(
                f"  参照解の注入の残差: {error:.3e} p.u."
                f"（掲載 {case.reference.digits} 桁の丸めだけで {tolerance:.1e} まで出る）"
            )
            if error > tolerance:
                print("    残差が大きい。潮流解と線路データが食い違っている。")
                problems = list(problems) + ["参照解と線路データが不整合"]

    print()
    return len(problems)


def main(argv: list[str]) -> int:
    targets = argv or list_cases()
    total = 0
    for target in targets:
        total += report(target)
    if total:
        print(f"合計 {total} 件の問題が見つかった。")
        return 1
    print(f"{len(targets)} 件のケースをすべて確認した。問題なし。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
