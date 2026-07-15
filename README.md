# ルー語 Activation Steering 実験

Qwen3-8B に対して、日本語・ルー語・英語の同一意味の短文 triplet から hidden state の差分方向を作り、ルー語方向が英語方向とどの程度重なるかを観察する小規模実験です。

目的は「綺麗にルー語化を成功させること」ではなく、ルー語っぽさが残差ストリーム上の単純な方向として見えるか、またそれが英語方向とどれくらい近いかを見ることです。

## 構成

```text
.
├── README.md
├── experiment.py
├── lou_triplets.json
├── pyproject.toml
├── requirements.txt
└── src/lou_steering/
```

主なファイルは次の通りです。

- `experiment.py`: 互換用の薄い実行スクリプトです。内部では `src/lou_steering/cli.py` を呼びます。
- `lou_triplets.json`: 現行実験で使う JA-Lou-EN の短文triplet 300件です。
- `src/lou_steering/`: 実験コード本体です。標準的な `src/` layout にしています。
- `requirements.txt`: Runpod上で実験した主要依存ライブラリのバージョンです。
- `outputs/`: 実験結果の出力先です。git管理対象外です。

コードは役割別に分けています。

- `src/lou_steering/data/`: triplet読み込みとQwen chat templateまわり
- `src/lou_steering/models/`: Qwen3-8Bのロードとlayer情報取得
- `src/lou_steering/steering/`: tripletからのMean pooling / Final token表現抽出とsteering vector作成
- `src/lou_steering/generation/`: seed固定の生成処理
- `src/lou_steering/evaluation/`: Japanese/Katakana/English割合、cosine、summary、グラフ
- `src/lou_steering/runtime/`: GPU/Python/PyTorch等の環境記録
- `src/lou_steering/experiments/`: 実験全体の実行フロー

## Triplet データ

`lou_triplets.json` は、同一意味の短文を `ja`, `lou`, `en` で持ちます。

```json
{
  "ja": "今日は一緒に昼ご飯を食べよう。",
  "lou": "トゥデイはトゥギャザーでランチしよう。",
  "en": "Let's eat lunch together today."
}
```

chat messages はJSONには保存しません。hidden state抽出時は、system promptなしのuser-only chat templateに入れ、`add_generation_prompt=True` で短文本文だけを差し替えます。組み立て処理は `src/lou_steering/data/chat.py` にあります。

## セットアップ

Runpodで使った依存関係は `requirements.txt` に固定しています。

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## 実行

疎通確認だけする場合:

```bash
python experiment.py --mode smoke
```

小さなsweepを回す場合:

```bash
python experiment.py --mode sweep
```

必要なら少し追加で見る場合:

```bash
python experiment.py --mode extended
```

今回の依頼では実行はしていません。コードとデータの準備だけ行っています。

## 実験設計

各tripletについて、system promptなしの同じuser-only chat templateで次の3条件のhidden stateを取得します。`add_generation_prompt=True` とし、assistant応答生成直前までの文脈を入力します。

- $x^{JA}$: 通常日本語短文
- $x^{Lou}$: ルー語短文
- $x^{EN}$: 英語短文

各層 $l$ で、2種類の表現を抽出します。

- Mean pooling: user message内の本文sentenceに対応するtoken位置集合 $C_x$ だけのhidden state平均
- Final token: assistant応答生成開始位置、つまり応答生成直前の文脈hidden state

そのうえで、pooling方法ごと、層ごとに次の方向を作ります。

$$
\mathrm{mean}_l(x) = \frac{1}{|C_x|}\sum_{t \in C_x} h_{l,t}
$$

$$
\mathrm{final}_l(x) = h_{l,t_{\mathrm{assistant\ start}}}
$$

$$
v_{\mathrm{Lou-JA}}^{(l)} = \mathbb{E}\left[h_l(x^{Lou}) - h_l(x^{JA})\right]
$$

$$
v_{\mathrm{EN-JA}}^{(l)} = \mathbb{E}\left[h_l(x^{EN}) - h_l(x^{JA})\right]
$$

Mean poolingは入力本文の平均表現、Final tokenは応答生成直前の文脈表現として解釈します。role token、turn delimiter、assistant generation prompt、BOS/EOSなどはMean pooling対象から除外します。

さらに、各層で $v_{\mathrm{Lou-JA}}$ と $v_{\mathrm{EN-JA}}$ のcosine similarityを計算し、Mean poolingとFinal tokenの違いも比較します。

また、$v_L = v_{\mathrm{Lou-JA}}$、$v_E = v_{\mathrm{EN-JA}}$ として、$v_L$ を $v_E$ 方向の成分と、それを除いた直交成分に分解します。

$$
 v_{\parallel} = \operatorname{proj}_{v_E}(v_L)
 = \frac{v_L^\top v_E}{\|v_E\|^2}v_E
$$

$$
 v_{\perp} = v_L - v_{\parallel}
$$

$v_{\perp}$ はEN-JA方向と直交するLou-JA成分として扱います。数値確認として、各層で次がほぼ0になることを確認します。

$$
\cos(v_{\perp}, v_E) \approx 0
$$

## Activation Steering

抽出した $v_{\mathrm{Lou-JA}}$ と $v_{\mathrm{EN-JA}}$ を使って、日本語promptへのactivation steeringも実装しています。

比較する条件は次の通りです。

- baseline
- Mean pooling由来の $v_{\mathrm{Lou-JA}}$
- Final token由来の $v_{\mathrm{Lou-JA}}$
- Mean pooling由来の $v_{\mathrm{EN-JA}}$
- Final token由来の $v_{\mathrm{EN-JA}}$
- Mean pooling由来の $v_{\parallel}$
- Final token由来の $v_{\parallel}$
- Mean pooling由来の $v_{\perp}$
- Final token由来の $v_{\perp}$

50件の評価promptは、ベクトル生成用のtriplet短文と完全一致しないようにしています。

baseline生成は各promptにつき一度だけ行います。summaryやグラフ上では各pooling/layer/coefficientへ展開しますが、同じbaselineを何度も再生成しません。

steering適用は `steering-vectors` の `SteeringVector.apply` を使います。独自forward hookは実装していません。

## 出力

実行すると、主に次のファイルが `outputs/` に出ます。

- `lou_minus_ja_mean_<mode>.pt`
- `lou_minus_ja_final_<mode>.pt`
- `en_minus_ja_mean_<mode>.pt`
- `en_minus_ja_final_<mode>.pt`
- `lou_parallel_en_mean_<mode>.pt`
- `lou_parallel_en_final_<mode>.pt`
- `lou_perp_en_mean_<mode>.pt`
- `lou_perp_en_final_<mode>.pt`
- `cosine_lou_ja_vs_en_ja_<mode>.json`
- `cosine_lou_ja_vs_en_ja_<mode>.png`
- `projection_lou_perp_vs_en_ja_<mode>.json`
- `projection_lou_perp_vs_en_ja_<mode>.png`
- `generations_<mode>.json`
- `summary_<mode>.json`
- `ratio_japanese_rate_<pooling>_layer<layer>_<mode>.png`
- `ratio_katakana_rate_<pooling>_layer<layer>_<mode>.png`
- `ratio_english_rate_<pooling>_layer<layer>_<mode>.png`

`outputs/` は `.gitignore` で管理対象外にしています。

## 実装メモ

- モデル: `Qwen/Qwen3-8B`
- ロードdtype: BF16
- steeringライブラリ: `steering-vectors`
- 介入対象: `decoder_block`
- Qwen3での実際の位置: `model.layers.{num}` のdecoder block出力、つまりblock output residual stream
- attention head、MLP、embedding、logitへの個別介入はしていません。
- system promptは使いません。
- Qwen3のthinking modeを避けるため、chat templateが対応している場合は `enable_thinking=False` を渡します。

## 評価指標

steering生成の観察補助として、出力文字を次の3カテゴリに分けます。

- Japanese: ひらがな + 漢字
- Katakana: カタカナ
- English: ASCII alphabet

分母は `Japanese + Katakana + English` のみです。句読点、空白、改行、数字、記号は分母に入れません。

$$
r_c = \frac{N_c}{N_{\mathrm{Japanese}} + N_{\mathrm{Katakana}} + N_{\mathrm{English}}}
$$

`baseline / Lou-JA steering / EN-JA steering / parallel steering / perpendicular steering` を同じalpha sweep上で比較し、Japanese/Katakana/Englishの3割合を折れ線で見ます。
