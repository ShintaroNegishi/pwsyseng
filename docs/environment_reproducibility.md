# 計算環境の再現性

`environment.yml` は、授業で必要なパッケージの集合と主要な Python 版を定める
**互換環境仕様**です。インストール日によって依存パッケージの版が変わり得るため、
このファイルだけで将来にわたり完全に同一のバイナリ環境が再現されるわけではありません。

授業期間中は、次の運用を推奨します。

1. 授業開始前に Windows、macOS、Linux の GitHub Actions が通ることを確認する
2. 動作確認済みのコミットまたはリリースタグを学生へ指定する
3. 問題が生じた端末では次を保存し、教員へ共有する

```bash
conda activate pwsyseng
conda list
python -c "import sys; print(sys.version)"
python -c "import numpy, scipy, pulp, control; print(numpy.__version__, scipy.__version__, pulp.__version__, control.__version__)"
```

端末固有の完全なパッケージ一覧を保存する場合は、次を使用します。

```bash
conda list --explicit > pwsyseng-explicit-spec.txt
```

この explicit spec は OS と CPU アーキテクチャに依存します。Windows 用のファイルを
macOS や Linux へそのまま適用しないでください。科目の標準手順は引き続き
`environment.yml`を使用し、explicit spec は不具合調査と同一端末群での再現に用います。
