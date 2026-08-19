# Steering Vectorによるルー語生成

Qwen3-8B に対して、日本語・ルー語・英語の同一意味の短文 triplet から hidden state の差分方向を作り、ルー語方向が英語方向とどの程度重なるかを観察する小規模実験です。

実験は RunPod 上の NVIDIA GeForce RTX 4090 を使用し、Docker イメージ `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` の環境で実行しました。

## 構成

```text
.
├── README.md
├── experiment.py
├── lou_triplets.json
├── pyproject.toml
├── requirements.txt
└── src/
```


- `experiment.py`: 互換用の薄い実行スクリプトです。内部では `src/cli.py` を呼びます。
- `lou_triplets.json`: 現行実験で使う JA-Lou-EN の短文triplet 300件です。
- `requirements.txt`: 依存関係
- `outputs/`: 実験結果の出力先です。git管理対象外です。

- `src/data/`: triplet読み込みとQwen chat template
- `src/models/`: Qwen3-8Bのロードとlayer情報取得
- `src/steering/`: tripletからのMean pooling / Final token表現抽出とsteering vector作成
- `src/generation/`: seed固定の生成処理
- `src/evaluation/`: Japanese/Katakana/English割合、cosine、summary、グラフ
- `src/runtime/`: GPU/Python/PyTorch等の環境記録
- `src/experiments/`: 実験全体の実行フロー

## Triplet データ

`lou_triplets.json` は、同一意味の短文を `ja`, `lou`, `en` で持ちます。

```json
{
  "ja": "今日は一緒に昼ご飯を食べよう。",
  "lou": "トゥデイはトゥギャザーでランチしよう。",
  "en": "Let's eat lunch together today."
}
```


## セットアップ

依存関係は `requirements.txt` に固定しています。

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## 実行

`smoke` は疎通確認用です。少数のtriplet、1つのlayer、1つの評価promptだけで、モデルロード、hidden state抽出、vector作成、steering生成、summary/graph出力までが壊れていないかを短時間で確認します。

```bash
python experiment.py --mode smoke
```

`sweep` は本番用の軽量sweepです。300件のtripletで全層のvector/cosine/projectionを作り、介入layerを30%、45%、60%、75%地点の4層に絞って50件の評価promptに対するsteering結果を比較します。

```bash
python experiment.py --mode sweep
```

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

さらに、各層で $v_{\mathrm{Lou-JA}}$ と $v_{\mathrm{EN-JA}}$ の cosine similarity を計算し、Mean pooling と Final token の違いも比較します。

また、$`v_L = v_{\mathrm{Lou-JA}}`$、$`v_E = v_{\mathrm{EN-JA}}`$ として、$`v_L`$ を $`v_E`$ 方向の成分と、それを除いた直交成分に分解します。


$$
v_{\parallel}
= \mathrm{proj}_{v_E}(v_L)
= \frac{v_L^\top v_E}{\lVert v_E \rVert^2}v_E
$$

$$
v_{\perp} = v_L - v_{\parallel}
$$

$v_{\perp}$ は、EN-JA 方向の成分を $v_L$ から除去した、$v_E$ に直交する Lou-JA 成分として扱います。数値的な確認として、各層で次の値がほぼ 0 になることを確認します。

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

baseline生成は各promptにつき一度だけ行い、JSON上でも単独のbaseline行として保存します。グラフではそのbaseline平均を各layer/coefficient sweep上の水平線として重ねます。

steering適用は `steering-vectors` の `SteeringVector.apply` を使います。独自forward hookは実装していません。


## 実装メモ

- モデル: `Qwen/Qwen3-8B`
- ロードdtype: BF16
- steeringライブラリ: `steering-vectors`
- 介入対象: `decoder_block`
- Qwen3での実際の位置: `model.layers.{num}` のdecoder block出力、つまりblock output residual stream
- attention head、MLP、embedding、logitへの個別介入はしていません。
- hidden state抽出にはsystem promptを使いません。生成時のみ「次の質問に対して100字以内で簡潔に答えてください」を使用します。

## 評価指標

steering生成の観察補助として、出力文字を次の3カテゴリに分けます。

- Japanese: ひらがな + 漢字
- Katakana: カタカナ
- English: ASCII alphabet

分母は `Japanese + Katakana + English` のみです。句読点、空白、改行、数字、記号は分母に入れません。

$$
r_c = \frac{N_c}{N_{\mathrm{Japanese}} + N_{\mathrm{Katakana}} + N_{\mathrm{English}}}
$$

`baseline / Lou-JA steering / EN-JA steering / parallel steering / perpendicular steering` を同じalpha sweep上で比較し、Japanese/Katakana/Englishの3割合を比較します
