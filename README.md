# sd-webui-TokenNorm

Port of **token normalization** and **weight interpretation** from ComfyUI's `CLIP Text Encode (Advanced)` node ([BlenderNeko/ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)) to Stable Diffusion WebUI reForge.

This extension does **not** patch any WebUI internals. It appends entries to `modules.sd_emphasis.options`, which is a plain module level list. Nothing else in the WebUI is modified, replaced or wrapped.

---

## What it does

When you write `(cat:1.5)`, two independent questions have to be answered:

1. **Should that 1.5 be conditioned first?** A word split into many tokens, or a textual inversion embedding occupying many vectors, receives the weight on every one of them. That may or may not be desirable. This is **token normalization**.
2. **What does multiplying by 1.5 actually mean?** Where is "zero"? A1111 style scales the embedding from the origin. ComfyUI interpolates from the embedding of an empty prompt. This is **weight interpretation**.

The upstream node exposes both axes. This extension provides both.

## Options

Everything appears in **Settings -> Emphasis**. No UI is added to the generation page.

| normalization | A1111 style application | ComfyUI interpretation |
|---|---|---|
| none | `No norm` (built in) | `TokenNorm: none / comfy` |
| mean | `TokenNorm: mean` | `TokenNorm: mean / comfy` |
| length | `TokenNorm: length` | `TokenNorm: length / comfy` |
| length + mean | `TokenNorm: length+mean` | `TokenNorm: length+mean / comfy` |

Naming rule: **no suffix means A1111 style, the `/ comfy` suffix means the ComfyUI interpretation.** The `none` x A1111 cell is WebUI's own `No norm`, so it is not duplicated here.

### Normalization

**mean** shifts every content token weight by the same amount so that their average becomes exactly 1.0.

```
delta   = 1 - average(content weights)
weights = weights + delta
```

Emphasising one tag therefore pulls every other tag down. The total weight budget stays constant.

**length** divides the weight of a multi-token textual inversion embedding so that the magnitude of the weight change does not grow with the vector count.

```
d = w - 1
w = 1 + sign(d) * sqrt(d * d / n)
```

For a 26 vector embedding, `1.5` becomes `1.098058`. Without this, a large embedding written as `(myEmbedding:1.5)` applies 1.5 to all 26 vectors at once.

### Interpretation

**A1111 style** multiplies the embedding by the weight. Weight 0.0 silences the token completely.

**comfy** interpolates towards the embedding of an empty prompt.

```
z = z_empty + (z - z_empty) * w
```

`z_empty` is a per position reference, not a single vector: it is the encoder output for a chunk containing no content at all. Weight 0.0 therefore does not silence a token, it returns that position to whatever an empty prompt would have produced. Weight 1.0 leaves the position untouched, bit for bit.

This is what ComfyUI does natively, so the `/ comfy` cells are the ones to use when you want a prompt to behave the way it does over there.

---

## Installation

Extensions -> Install from URL, or clone into `extensions/`:

```
git clone https://github.com/seti9585/sd-webui-TokenNorm
```

Restart the WebUI process. "Apply and restart UI" is not enough, because it does not restart Python.

Then pick an option in **Settings -> Emphasis**.

## Requirements

- **reForge only.** Forge Classic / Neo moved text processing to `backend/text_processing/` and this extension's entry point does not exist there. A1111 upstream is untested.
- The `sd_emphasis` path must be intact. Any extension that replaces `process_tokens` at class level bypasses it entirely, and every Emphasis option including these has no effect. This extension checks for that on startup and prints a warning naming the offending class.

---

## Divergences from upstream

Each of these is a deliberate decision, not an oversight.

### The A1111 style cells do not restore the mean

Upstream's `A1111` interpretation multiplies and then rescales the result so the mean is preserved. That is exactly WebUI's built-in `Original`. This extension builds on `No norm` instead, which multiplies without the rescale.

The reason is practical: mean restoration is SD1.x era behaviour, and WebUI's own description of `No norm` says it "seems to work better for SDXL". The `Original` based variants can be added later without renaming anything, since `sd_emphasis.options` is just a list.

### mean is computed per chunk

Upstream flattens the whole prompt and takes one average. reForge creates a separate `Emphasis` object for each 75 token chunk and no state can be carried between them, so the average is taken per chunk.

**For prompts that fit in a single chunk the two are identical.** Longer prompts differ: upstream preserves the relative difference between chunks, this implementation flattens within each chunk independently. Neither is obviously better. A prompt whose first chunk is deliberately heavier keeps that intent upstream; a prompt suffering from "later tags do not take effect" may do better here.

### length only applies to embeddings

Upstream divides by the number of tokens a **word** was split into, using `word_id` from `clip.tokenize(return_word_ids=True)`.

WebUI discards word boundaries. `modules/sd_hijack_clip.py:tokenize_line()` keeps only `(token, multiplier)` pairs, so one word split into three tokens is indistinguishable from three separate words that happen to share a weight.

The one case where the token count survives is a textual inversion embedding: it occupies several consecutive slots whose id is `0`, all carrying the same multiplier. That is verifiable without word boundaries.

So `length` here rescales only runs made entirely of embedding placeholders. Ordinary tokens are left alone, which also matches upstream for single token words, since `sqrt(1) == 1`. Multi word groups such as `(fluffy white cat:1.5)` are deliberately **not** rescaled: upstream treats those as three separate one token words and leaves them at 1.5.

### comfy is sourced from ComfyUI itself

`adv_encode.py` does not contain the comfy formula. Its `comfy` branch hands the token/weight pairs to ComfyUI's own encoder, and the interpolation lives in `comfy/sd1_clip.py`, in `ClipTokenWeightEncoder.encode_token_weights`. That is the reference used here.

The empty chunk is built as `[id_start, id_end, id_pad * chunk_length]`, which is what reForge itself produces for an empty prompt after `process_tokens()` overwrites everything past the first `id_end` with `id_pad`. It is the same sequence as ComfyUI's `gen_empty_tokens()`. This was checked against the actual `z` of an empty prompt and matched to the last digit for both CLIP-L and CLIP-G. The distinction matters only for CLIP-G, where `id_pad` differs from `id_end`.

`z_empty` depends only on the encoder, the checkpoint and the CLIP skip setting, never on the prompt, so it is computed once and cached on the encoder object, one slot each. This was verified across four prompts, two checkpoints and two CLIP skip settings: recomputations differed by exactly 0.

The pooled output is not affected by weights in ComfyUI either, and reForge carries pooled around the Emphasis step, so nothing is needed there.

---

## Not implemented

Upstream offers five interpretations. Only `comfy` is ported.

| interpretation | reference | status |
|---|---|---|
| `A1111` | A1111 WebUI | already built in as `Original` / `No norm` |
| `comfy` | ComfyUI core | **implemented** |
| `down_weight` | upstream only | not implemented |
| `compel` | damian0815/compel | not implemented |
| `comfy++` | upstream only | not implemented |

`comfy` is the only one with a reference outside the upstream extension that this extension can reproduce, and the only one that answers the practical question of why the same prompt behaves differently in ComfyUI.

The other three need extra encoder passes over modified token sequences. `down_weight` and `compel` group tokens by weight value, which is expressible in WebUI, but `comfy++` masks per `word_id`, which WebUI does not retain. `comfy++` is also an invention of the upstream author with no external reference, so there would be nothing to verify a port against.

---

## Troubleshooting

**Nothing changes when I switch options.** Check the console at startup. If another extension has replaced `process_tokens` at class level, a warning naming it is printed and no Emphasis option has any effect.

**The infotext does not record my Emphasis setting.** WebUI only writes the `Emphasis` key when the prompt actually contains weighting brackets. Without brackets the setting cannot change anything, so nothing is recorded. This is WebUI behaviour, not this extension.

**`TokenNorm comfy fallback` appears in the infotext.** The comfy interpretation could not resolve the text encoder for that generation and fell back to leaving the embedding untouched, meaning token weights had no effect. The value gives the reason. A pass-through is impossible to spot by eye, which is why it is recorded in the image.

**Debug logging.** Set `SD_WEBUI_SETI_DEBUG` before starting.

```
set SD_WEBUI_SETI_DEBUG=1
```

Level 1 reports registration, the startup check, EOS resolution and each `z_empty` computation. Level 2 adds per chunk detail: the `mean` delta, each `length` run, and how many positions the comfy interpolation touched.

## Notes

- A textual inversion embedding is large enough to dominate a chunk. When it does, `mean` is computed mostly from the embedding and the shift pushes ordinary tags well away from 1.0. Applying `length` first reduces this considerably. Both behaviours follow upstream.
- Switching Emphasis invalidates WebUI's conditioning cache automatically, since `opts.emphasis` is part of the cache key. Fixed seed comparisons are reliable.

---

## License

MIT

## Credits

- [BlenderNeko/ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb) - `adv_encode.py`, the origin of the token normalization methods and of the interpretation axis
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) - `comfy/sd1_clip.py`, `ClipTokenWeightEncoder.encode_token_weights`, the reference for the `comfy` interpretation
- [Panchovix/stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) and [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - `modules/sd_emphasis.py`, the extension point this builds on
- [Shiba-2-shiba](https://note.com/gentle_murre488) - articles that prompted this series of ports

---
---

# sd-webui-TokenNorm (日本語)

ComfyUI の `CLIP Text Encode (Advanced)` ノード（[BlenderNeko/ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)）が持つ **token normalization** と **weight interpretation** を、Stable Diffusion WebUI reForge へ移植したものです。

本体には一切パッチを当てません。`modules.sd_emphasis.options` というモジュールレベルのリストに項目を追加するだけで、それ以外の書き換え・差し替え・ラップは行いません。

---

## 何をするものか

`(cat:1.5)` と書いたとき、独立した 2 つの問いが発生します。

1. **その 1.5 を、そのまま使ってよいか。** 複数トークンに分割された語や、多数のベクトルを占める Textual Inversion 埋め込みには、その全てに重みが乗ります。望ましいとは限りません。これが **token normalization** です。
2. **1.5 倍とは、何を 1.5 倍することか。** ゼロ地点はどこか。A1111 方式は原点からのスケーリング、ComfyUI は空プロンプトの埋め込みからの補間です。これが **weight interpretation** です。

上流ノードはこの 2 軸を持ちます。本拡張も 2 軸を提供します。

## 選択肢

すべて **Settings -> Emphasis** に現れます。生成画面への UI 追加はありません。

| 正規化 | A1111 方式の適用 | ComfyUI 解釈 |
|---|---|---|
| なし | `No norm`（本体標準） | `TokenNorm: none / comfy` |
| mean | `TokenNorm: mean` | `TokenNorm: mean / comfy` |
| length | `TokenNorm: length` | `TokenNorm: length / comfy` |
| length + mean | `TokenNorm: length+mean` | `TokenNorm: length+mean / comfy` |

命名規則は、**接尾辞なしが A1111 方式、`/ comfy` が ComfyUI 解釈**です。「正規化なし × A1111」のセルは本体の `No norm` そのものなので、重複して提供していません。

### 正規化

**mean** は、内容トークンの重みの平均がちょうど 1.0 になるよう、全体を同じ量だけ加算シフトします。

```
delta = 1 - 内容トークンの重みの平均
重み  = 重み + delta
```

したがって、あるタグを強調すると他のタグが引き下げられます。重みの総量が一定に保たれる方式です。

**length** は、複数ベクトルの Textual Inversion 埋め込みについて、重みの変化量がベクトル数に比例して膨らまないよう補正します。

```
d = w - 1
w = 1 + sign(d) * sqrt(d * d / n)
```

26 ベクトルの埋め込みなら `1.5` は `1.098058` になります。補正しない場合、`(myEmbedding:1.5)` は 26 個すべてに 1.5 を掛けることになります。

### 解釈

**A1111 方式**は、埋め込みに重みを乗算します。重み 0.0 でそのトークンは完全に消えます。

**comfy** は、空プロンプトの埋め込みへ向けて補間します。

```
z = z_empty + (z - z_empty) * w
```

`z_empty` は 1 本のベクトルではなく、**位置ごとの基準点の列**です。内容を含まないチャンクをエンコードした結果そのものだからです。したがって重み 0.0 はトークンを消すのではなく、その位置を「何も書かなかったときの状態」へ戻します。重み 1.0 の位置はビット単位で変化しません。

これが ComfyUI 本来の挙動なので、**あちらと同じ効き方を再現したい場合は `/ comfy` 側**を選んでください。

---

## 導入

Extensions -> Install from URL、または `extensions/` に clone してください。

```
git clone https://github.com/seti9585/sd-webui-TokenNorm
```

**WebUI をプロセスごと再起動してください。**「Apply and restart UI」では Python が再起動しないため不十分です。

その後 **Settings -> Emphasis** で方式を選びます。

## 動作要件

- **reForge 専用です。** Forge Classic / Neo はテキスト処理を `backend/text_processing/` へ移行しており、本拡張の介入点が存在しません。A1111 本家は未検証です。
- `sd_emphasis` の経路が生きている必要があります。`process_tokens` をクラスレベルで差し替える拡張が入っていると経路ごと迂回され、**本拡張に限らず Emphasis の設定すべてが無効**になります。起動時に検査し、該当クラス名を挙げて警告します。

---

## 原実装との相違点

いずれも意図的な判断であり、実装漏れではありません。

### A1111 方式のセルは平均復元を行いません

上流の `A1111` 解釈は、乗算したあと平均が保たれるよう再スケールします。これは本体の `Original` そのものです。本拡張は再スケールを行わない `No norm` を土台としています。

理由は実用面です。平均復元は SD1.x 時代の挙動であり、本体自身が `No norm` の説明で「SDXL ではこちらのほうが良さそう」と述べています。`Original` 土台版は、名前を一切変えずに後から追加できます（`sd_emphasis.options` は単なるリストのため）。

### mean はチャンク単位で計算します

上流はプロンプト全体を平坦化して 1 つの平均を取ります。reForge は 75 トークンのチャンクごとに `Emphasis` インスタンスを生成し直し、状態を持ち越せないため、平均はチャンクごとになります。

**1 チャンクに収まるプロンプトでは両者は完全に一致します。** それを超えると挙動が分かれます。上流はチャンク間の相対差を保存し、本実装はチャンク内で独立に平坦化します。どちらが優れているとは言えません。前半を意図的に濃くした構成なら上流方式が意図を保ちますが、「後半のタグが効かない」という症状には本実装のほうが向く可能性があります。

### length は埋め込み限定です

上流は、**単語**が何トークンに分割されたかで割ります。`clip.tokenize(return_word_ids=True)` の `word_id` を使うためです。

WebUI は単語境界を捨てます。`modules/sd_hijack_clip.py` の `tokenize_line()` が `(トークン, 重み)` の組しか保持しないため、1 語が 3 トークンに割れたものと、たまたま同じ重みを持つ 3 語とを区別できません。

トークン数が生き残る唯一の例が Textual Inversion 埋め込みです。ID が `0` の連続したスロットを占め、全てが同じ重みを持つため、単語境界なしで判定できます。

そのため本実装の `length` は、埋め込みプレースホルダのみで構成される連続領域だけを補正します。通常トークンは素通しです。これは単一トークン語に対する上流の挙動（`sqrt(1) == 1`）とも一致します。`(fluffy white cat:1.5)` のような複数語のまとまりは**意図的に補正しません**。上流はこれを 1 トークン語 3 個として扱い、1.5 のままにするためです。

### comfy の典拠は ComfyUI 本体です

`adv_encode.py` に comfy の式はありません。`comfy` の分岐はトークンと重みの組を ComfyUI 本体のエンコーダへ渡すだけで、補間は `comfy/sd1_clip.py` の `ClipTokenWeightEncoder.encode_token_weights` にあります。本実装はそちらを典拠としています。

空チャンクは `[id_start, id_end, id_pad * chunk_length]` として構成します。これは `process_tokens()` が最初の `id_end` 以降を `id_pad` で塗り潰した結果、reForge 自身が空プロンプトに対して生成する列であり、ComfyUI の `gen_empty_tokens()` と同型です。実際の空プロンプトの `z` と突き合わせ、CLIP-L / CLIP-G の双方で全桁一致することを確認しています。この違いが問題になるのは `id_pad` が `id_end` と異なる CLIP-G 側だけです。

`z_empty` はエンコーダ・チェックポイント・CLIP skip 設定のみに依存し、プロンプトには依存しません。したがって 1 度だけ計算し、エンコーダオブジェクトに 1 スロットずつキャッシュします。4 種のプロンプト、2 つのチェックポイント、2 つの CLIP skip 設定で検証し、再計算値の差は厳密に 0 でした。

pooled 出力は ComfyUI 側でも重みの影響を受けません。reForge も Emphasis の前後で pooled を退避・復帰させるため、こちらで対処すべきことはありません。

---

## 実装していないもの

上流は 5 種類の解釈を提供しますが、移植したのは `comfy` のみです。

| 解釈 | 典拠 | 状態 |
|---|---|---|
| `A1111` | A1111 WebUI | 本体に `Original` / `No norm` として実装済み |
| `comfy` | ComfyUI 本体 | **実装** |
| `down_weight` | 上流のみ | 未実装 |
| `compel` | damian0815/compel | 未実装 |
| `comfy++` | 上流のみ | 未実装 |

`comfy` は、上流拡張の外に典拠があり、かつ本拡張で再現可能な唯一の解釈です。また「同じプロンプトなのに ComfyUI と効き方が違う」という実務上の疑問に直接答えるものでもあります。

残る 3 つは、トークン列を書き換えた追加エンコードを必要とします。`down_weight` と `compel` は重み値でトークンをまとめるため WebUI でも表現可能ですが、`comfy++` は `word_id` 単位でマスクするため成立しません。加えて `comfy++` は上流作者の独自実装であり外部典拠が存在しないため、移植しても照合対象がありません。

---

## トラブルシューティング

**方式を切り替えても何も変わらない。** 起動時のコンソールを確認してください。`process_tokens` をクラスレベルで差し替える拡張が入っている場合、そのクラス名を挙げた警告が出ます。この状態では Emphasis の設定は一切効きません。

**infotext に Emphasis 設定が記録されない。** WebUI は、プロンプトに重み指定の括弧が含まれるときにのみ `Emphasis` キーを書き出します。括弧が無ければ設定は何も変えないため、記録もされません。本体の挙動であり、本拡張によるものではありません。

**infotext に `TokenNorm comfy fallback` が出た。** その生成で comfy 解釈がテキストエンコーダを特定できず、埋め込みに触れずに素通ししたことを意味します。つまりトークンの重みが効いていません。値に理由が入ります。素通しは目視では判別できないため、画像側に記録する設計にしています。

**デバッグログ。** 起動前に `SD_WEBUI_SETI_DEBUG` を設定してください。

```
set SD_WEBUI_SETI_DEBUG=1
```

レベル 1 で、登録・起動時検査・EOS の解決・`z_empty` の計算が出ます。レベル 2 でチャンクごとの詳細（`mean` の delta、`length` の各 run、comfy が補間した位置数）が追加されます。

## 補足

- Textual Inversion 埋め込みはチャンクの大半を占めることがあります。その場合 `mean` の平均は埋め込みに支配され、シフトによって通常タグが 1.0 から大きく引き離されます。`length` を先に適用するとこれはかなり緩和されます。どちらも上流どおりの挙動です。
- Emphasis を切り替えると本体の conditioning キャッシュは自動的に無効化されます（`opts.emphasis` がキャッシュキーに含まれるため）。固定 seed での比較は信頼できます。

---

## ライセンス

MIT

## 典拠・謝辞

- [BlenderNeko/ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb) — `adv_encode.py`。token normalization 各方式および解釈軸の原典
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) — `comfy/sd1_clip.py` の `ClipTokenWeightEncoder.encode_token_weights`。`comfy` 解釈の典拠
- [Panchovix/stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) および [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) — `modules/sd_emphasis.py`。本拡張が乗っている拡張点
- [Shiba-2-shiba](https://note.com/gentle_murre488) — 一連の移植のきっかけとなった記事
