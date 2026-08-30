# GitHub 公開前のチェックリスト

公開・更新のたびに抜けやすい項目をまとめました。

## 1. 著者情報とリポジトリ URL

著作権表記は MIT ライセンス・個人名義（Shintaro Negishi）です。
メールアドレスは載せていません。載せる場合は `CITATION.cff` の `authors` に
`email:` を、`pyproject.toml` の `authors` に `email = ` を追加してください。
公開リポジトリに載せると収集対象になる点は考慮してください。

## 2. 研究室サーバ内の情報が混ざっていないか

開発は研究室サーバ上で行っており、絶対パスや環境固有の値が紛れ込みやすい
ところです。

```bash
grep -rn "/home/LAB\|/home/negishi\|anaconda3" \
  --include="*.py" --include="*.yaml" --include="*.yml" \
  --include="*.md" --include="*.ipynb" \
  src/ tests/ tools/ notebooks/ exercises/ docs/ .github/ cases/
```

何も出なければ完了です。平文の認証情報がある場合は公開前にマスキングします。

## 3. 公開前の動作確認

```bash
conda activate pwsyseng
python tools/check_case.py         # 同梱ケースの整合性
pytest -q                          # すべて通ること
python tools/build_notebooks.py    # notebook を最新の src から再生成・構造検査
git diff --exit-code -- notebooks exercises  # 生成物のコミット漏れがないこと
```

再生成した `notebooks/*.ipynb` と `exercises/*.ipynb` もコミットしてください。
CI は再生成後に `git diff --exit-code` を実行し、コミット済みの `.ipynb` が
原本と一致しない場合に失敗します。学生が受け取る生成物まで同期していることを確認できます。

## 4. notebook の出力を消しておく

`tools/build_notebooks.py` が生成する `.ipynb` には実行結果が入りません。
手元で実行して確認したあとにコミットする場合は、出力を消してください
（学生が自分で実行することに意味があるため）。

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb exercises/*.ipynb
```

## 5. データの出典

`docs/data_provenance.md` の表と、`cases/wscc9.yaml`・
`src/gridops/casedata/wscc9.yaml` の冒頭コメントが一致していることを
確認してください。特に次を確かめます。

- 公知のベンチマークには書誌が書いてあるか
- **原典にない自作の値（熱容量・燃料費・信頼度データ・時系列需要）が
  「自作である」と明記されているか**
- 第三者が編纂したデータが紛れ込んでいないか

## 6. 商用ソルバへの依存が無いこと

学生の手元でライセンスの問題を起こさないための方針です。

```bash
grep -rn "gurobipy" src/ notebooks/src/ tools/
```

`src/gridops/solvers.py` の「使わない」という方針の記述以外に出てこなければ
完了です。

## 7. 2 つのパッケージの関係

依存は gridops → genstab の **一方向**です。次を確認してください。

- `src/gridops/` のどのモジュールも、`interop.py` 以外で `genstab` を
  import していないこと（`interop.py` の import は関数の中にあること）
- `src/genstab/` が `gridops` を import していないこと

```bash
grep -rn "import genstab" src/gridops/
grep -rn "import gridops" src/genstab/
```

## 8. CI の確認

push 後、GitHub Actions が 3 つの OS で通ることを確認してください。
特に重要なのは次の 2 つです。

- **Windows で `environment.yml` から環境が作れるか**
- **CBC が使えることを確認するステップが通るか**（`coin-or-cbc` の
  Windows 版が入るか）

ここが通れば、学生の PC で環境構築が失敗する主な原因は潰せています。
失敗した場合は Actions のログに conda の解決結果が出るので、
問題のあるパッケージを特定できます。

## 9. 旧リポジトリとの関係

本リポジトリは、単体で公開していた
[genstab](https://github.com/ShintaroNegishi/genstab) に運用・計画パート
（gridops）を加えて統合したものです。授業では本リポジトリだけを使います。
genstab 側の README に統合の案内を載せるかどうかは、単体利用者の有無を
見て判断してください。
