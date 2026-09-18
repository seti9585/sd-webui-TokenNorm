# sd-webui-TokenNorm

**English** | [日本語](#日本語)

An extension for **Stable Diffusion WebUI reForge** that changes how prompt weights such as `(cat:1.5)` are applied.

- **mean:** shifts token weights so their average is 1 within each prompt chunk.
- **length:** moves the weights of multi-slot Textual Inversions closer to 1.
- **/ comfy:** applies weights using ComfyUI's formula instead of simple multiplication.

Options appear in **Settings → Stable Diffusion → Emphasis mode**. They affect both positive and negative prompts. TokenNorm does not change LoRA strength settings such as `<lora:name:0.8>`.

The weight adjustment methods come from ComfyUI's [CLIP Text Encode (Advanced)](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb) node. The ComfyUI application formula comes from ComfyUI itself. This is a partial port; it does not make reForge and ComfyUI produce identical images.

## Installation and use

1. Open **Extensions → Install from URL** and install:

   ```text
   https://github.com/seti9585/sd-webui-TokenNorm
   ```

2. Stop WebUI and start it again. Use a full process restart after installation.
3. Select an option under **Settings → Stable Diffusion → Emphasis mode** using the guide below.
4. Press **Apply settings**, then generate normally.

To return to WebUI's default weighting, select `Original` and press **Apply settings**.

**Requirements:** reForge is the supported target. Forge Classic / Neo use a different prompt-processing path; A1111 is untested. Keep **Settings → Compatibility → Use old emphasis implementation** off. Extensions that bypass the Emphasis step can also prevent these options from working.

## Choosing an option

| Purpose | Option |
| --- | --- |
| Keep WebUI's default weighting | `Original` (built in) |
| Reduce how far a multi-slot Textual Inversion's weight deviates from 1 | `TokenNorm: length` |
| Keep the average token weight at 1 within each chunk | `TokenNorm: mean` |
| Apply length adjustment, then mean adjustment | `TokenNorm: length+mean` |
| Use ComfyUI's weight application formula without adjusting weights first | `TokenNorm: none / comfy` |
| Combine an adjustment with ComfyUI's formula | The corresponding `/ comfy` option below |

**Switching from `Original` can change the image even if your written weights stay the same.** `Original` restores the average of the encoded values after weighting; TokenNorm does not perform that final correction. Compare `length` with `No norm` to isolate the length adjustment.

All combinations:

| Weight adjustment | Multiply (based on No norm) | ComfyUI formula |
| --- | --- | --- |
| none | `No norm` (built in) | `TokenNorm: none / comfy` |
| mean | `TokenNorm: mean` | `TokenNorm: mean / comfy` |
| length | `TokenNorm: length` | `TokenNorm: length / comfy` |
| length + mean | `TokenNorm: length+mean` | `TokenNorm: length+mean / comfy` |

## What changes

A **token** is a piece of the prompt: one tag can contain several tokens, and commas count too. WebUI processes content in **chunks of up to 75 tokens**. A **Textual Inversion** is an embedding file placed in `embeddings` and used by writing its name in the prompt; it can occupy one or more token slots.

### mean: keep the average weight at 1

Within each chunk, mean adds the same amount to every content token's weight. An average above 1 lowers all weights; an average below 1 raises them. Commas and Textual Inversion slots participate; start/end markers and padding do not.

For 10 content tokens, with one at 1.5 and nine at 1.0, the average is 1.05. Subtracting 0.05 gives **1.45 and 0.95**.

This adjusts the numerical weights, not a guaranteed amount of influence in the image. A Textual Inversion occupying most of a chunk can dominate that chunk's average.

### length: move Textual Inversion weights closer to 1

length adjusts each multi-slot Textual Inversion according to its number of slots. It reduces both emphasis above 1 and suppression below 1.

For an embedding with 26 slots:

| Input weight | After length |
| --- | --- |
| 1.5 | about 1.098 |
| 0.5 | about 0.902 |
| 1.0 | 1.0 |

Ordinary words and one-slot embeddings are unchanged. Adjacent embeddings and embeddings grouped as `(A, B, C)` are handled separately when their exact boundaries are available.

**length+mean runs length first, then mean.** Its final weights can therefore differ from the values in this table.

### / comfy: change how the adjusted weight is applied

Without `/ comfy`, the encoded values at each token position are multiplied by the adjusted weight, as in `No norm`.

With `/ comfy`, the reference is the value an empty prompt produces at that position:

- **1:** leaves the position unchanged.
- **Between 0 and 1:** moves it towards the empty-prompt value.
- **0:** returns it to the empty-prompt value.
- **Above 1:** moves it farther away.
- **Below 0:** moves it past the empty-prompt value to the opposite side.

These are **weights after adjustment**. An input such as `(tag:0)` can become nonzero after mean or length. Neither application method removes the word from the text: weighting happens after encoding, so the word's influence on other positions remains.

With `none / comfy`, an embedding's input weight applies to every slot. With `mean / comfy`, its mean-adjusted weight applies to every slot. Neither adjusts for slot count. If a weighted multi-slot embedding has too much influence, try the corresponding `length / comfy` or `length+mean / comfy` option.

## When an option makes no difference

- **All input weights are 1:** TokenNorm adds no weighting change. Parentheses such as `(tag)` mean 1.1, and `[tag]` means about 0.909, so those are weighted prompts too.
- **No multi-slot Textual Inversion is weighted:** with normal boundary detection, `length` behaves like `No norm`; `length / comfy` behaves like `none / comfy`.
- **All content weights in a chunk are equal:** mean brings them to 1. Increasing all of them together does not preserve that increase.
- **The weights entering mean already average 1:** mean adds no shift. In `length+mean`, this refers to the weights after length.

Emphasis mode is part of reForge's prompt cache key, so changing it triggers prompt re-encoding. Keep the seed and other generation settings fixed when comparing options.

## Troubleshooting

| Symptom | Check or meaning |
| --- | --- |
| Switching options has no effect | Check the conditions above, press **Apply settings**, and make sure **Use old emphasis implementation** is off. Check startup warnings for a prompt-processing override. |
| A startup warning names classes | TokenNorm detected a possible prompt-processing replacement. The warning reports class names, not necessarily the responsible extension. `sd-webui-prevent-artifact` is a known conflicting example. |
| No `Emphasis` entry in image information | In the standard reForge path, this entry is added when processed prompt text contains `(` or `[` and the mode is not `Original`. `Original` is omitted. |
| `TokenNorm comfy fallback` | ComfyUI weighting was skipped in some or all processing for that image. The value records a reason; this entry alone does not identify the full affected scope. Check the console too. |
| `TokenNorm length fallback` | Exact embedding boundaries were unavailable in some processing, so the older detection method was used. Adjacent embeddings may be merged or detected incorrectly. |

For debug output, add `set SD_WEBUI_SETI_DEBUG=2` to `webui-user.bat` **before the line that launches WebUI**, then restart. Level `1` reports startup information; `2` also reports weight adjustments and ComfyUI application. Remove the line and restart to disable it.

<details>
<summary>Technical notes: formulas, implementation, and differences from ComfyUI</summary>

### Processing order and formulas

Weight adjustment runs before application. `length+mean` runs length, then mean, then the selected application method.

mean, for content weights within one chunk:

```text
delta = 1 - average(weights)
weights = weights + delta
```

length, for an embedding with n slots and weight w:

```text
d = w - 1
w = 1 + sign(d) * sqrt(d * d / n)
```

Multiply and ComfyUI application, where w is the adjusted weight and z_empty is the encoded empty prompt at the same position:

```text
multiply: z_new = z * w
comfy:    z_new = z_empty + (z - z_empty) * w
```

The ComfyUI path explicitly preserves positions whose adjusted weight is 1, avoiding rounding changes from subtracting and adding z_empty. These formulas describe the intended arithmetic; floating-point results may involve rounding.

### Integration

TokenNorm registers options in `modules.sd_emphasis.options` without replacing or wrapping WebUI methods.

For length adjustment, it reads `PromptChunk.fixes` from the caller's chunk objects. The temporary `hijack.fixes` reference has already been cleared, but those objects remain available. The caller's token list must be the same object received by Emphasis, and the embedding spans are checked against the token positions. If this fails, the extension uses its older token-run detection and records `TokenNorm length fallback`.

### Differences from the Advanced node

- **mean:** the node calculates over the whole prompt; TokenNorm calculates separately for each chunk.
- **length:** the node uses word IDs to count tokens per word. TokenNorm uses Textual Inversion boundaries and leaves ordinary words unchanged, even if they span several tokens.
- **A1111 interpretation:** the node restores the encoded mean after multiplication, corresponding to the principle used by WebUI's `Original`. TokenNorm's multiplication options use `No norm` and omit that restoration.
- **comfy interpretation:** TokenNorm uses ComfyUI's empty-prompt formula. Tokenization and other processing differences still prevent a guarantee of equivalent conditioning or images.
- **Other interpretations:** `down_weight`, `compel`, and `comfy++` are not implemented.

</details>

---

# 日本語

[English](#sd-webui-tokennorm) | **日本語**

`(cat:1.5)` のようなプロンプトの重み指定の効き方を変える、**Stable Diffusion WebUI reForge** 用の拡張機能です。

- **mean：** プロンプトの区切りごとに、トークンの重みの平均を 1 にそろえます。
- **length：** 複数の枠を使う Textual Inversion の重みを 1 に近づけます。
- **/ comfy：** 単純な乗算に代えて、ComfyUI と同じ式で重みを適用します。

選択肢は **Settings → Stable Diffusion → Emphasis mode** に追加されます。ポジティブ・ネガティブの両方に作用します。`<lora:name:0.8>` のような LoRA の強度設定は変更しません。

重みの調整方法は、ComfyUI の [CLIP Text Encode (Advanced)](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb) ノードに由来します。ComfyUI 方式の適用式は ComfyUI 本体に由来します。一部の機能を移植したもので、reForge と ComfyUI で同じ画像を作るための互換機能ではありません。

## インストールと使い方

1. **Extensions → Install from URL** を開き、次の URL からインストールします。

   ```text
   https://github.com/seti9585/sd-webui-TokenNorm
   ```

2. WebUI を終了し、起動し直します。インストール後はプロセスごと再起動してください。
3. 下の選び方を参考に、**Settings → Stable Diffusion → Emphasis mode** で選択します。
4. **Apply settings** を押し、いつもどおり生成します。

WebUI 標準の重み付けに戻す場合は、`Original` を選び、**Apply settings** を押してください。

**動作要件：** 対象は reForge です。Forge Classic / Neo はプロンプトの処理経路が異なり、A1111 は未検証です。**Settings → Compatibility → Use old emphasis implementation** は OFF にしてください。Emphasis の処理を通らないように差し替える拡張との併用でも、選択肢が効かなくなる場合があります。

## どれを選ぶか

| 目的 | 選択肢 |
| --- | --- |
| WebUI 標準の重み付けを使う | `Original`（本体標準） |
| 複数枠の Textual Inversion に付けた重みを、1 に近づけたい | `TokenNorm: length` |
| 各チャンク内のトークンの重みの平均を 1 に保ちたい | `TokenNorm: mean` |
| length で調整してから、mean で平均をそろえたい | `TokenNorm: length+mean` |
| 重みを事前に調整せず、ComfyUI の式で適用したい | `TokenNorm: none / comfy` |
| 重みの調整と ComfyUI の式を組み合わせたい | 下表の対応する `/ comfy` 付きの選択肢 |

**`Original` から切り替えると、入力した重みが同じでも画像が変わることがあります。** `Original` は重み付け後にエンコード結果の平均を元に戻しますが、TokenNorm はこの最後の補正を行いません。length 自体の効果を見るときは、`No norm` と比較してください。

選択肢の全組み合わせ：

| 重みの調整 | 乗算（No norm が土台） | ComfyUI の式 |
| --- | --- | --- |
| なし | `No norm`（本体標準） | `TokenNorm: none / comfy` |
| mean | `TokenNorm: mean` | `TokenNorm: mean / comfy` |
| length | `TokenNorm: length` | `TokenNorm: length / comfy` |
| length + mean | `TokenNorm: length+mean` | `TokenNorm: length+mean / comfy` |

## 何が変わるのか

**トークン**はプロンプトを分割する単位です。1 つのタグが複数トークンになることがあり、カンマも数えます。WebUI は本文を**最大 75 トークンのチャンク**に区切って処理します。**Textual Inversion** は `embeddings` に置き、プロンプトに名前を書いて使う埋め込みファイルで、1 個以上のトークン枠を使います。

### mean：重みの平均を 1 にそろえる

チャンク内の本文トークンの重みに、同じ量を足し引きします。平均が 1 より大きければ全体が下がり、1 より小さければ全体が上がります。カンマや Textual Inversion の枠も対象です。開始・終了マーカーや、長さを埋めるための余白は含みません。

例えば本文トークンが 10 個で、1 個が 1.5、9 個が 1.0 なら、平均は 1.05 です。すべてから 0.05 を引き、**1.45 と 0.95** にします。

そろえるのは重みの数値の平均であり、画像への影響力が一定になる保証ではありません。Textual Inversion がチャンクの大半を占めると、その重みが平均を大きく左右します。

### length：Textual Inversion の重みを 1 に近づける

複数枠の Textual Inversion ごとに、枠数に応じて重みを調整します。1 より上の強調も、1 より下の抑制も弱めます。

26 枠の埋め込みの場合：

| 入力した重み | length 適用後 |
| --- | --- |
| 1.5 | 約 1.098 |
| 0.5 | 約 0.902 |
| 1.0 | 1.0 |

普通の単語や、1 枠だけの埋め込みは変わりません。埋め込みを隣接させたり `(A, B, C)` とまとめたりした場合も、正確な境界を取得できた場合は別々に処理します。

**length+mean は、length のあとに mean を行います。** そのため、最終的な重みは上の表と異なる場合があります。

### / comfy：調整後の重みの適用方法を変える

`/ comfy` が無い場合は、`No norm` と同じく、各トークンの位置のエンコード後の値に重みを掛けます。

`/ comfy` がある場合は、空のプロンプトが同じ位置に作る値を基準にします。

- **1：** その位置の値を変えません。
- **0 と 1 の間：** 空プロンプトの値へ近づけます。
- **0：** 空プロンプトの値に戻します。
- **1 より上：** 空プロンプトの値からさらに遠ざけます。
- **0 より下：** 空プロンプトの値を越え、反対側へ動かします。

これは**調整後の重み**の話です。`(tag:0)` と入力しても、mean や length のあとでは 0 以外になる場合があります。どちらの適用方法も、文章から単語を削除する処理ではありません。エンコード後に重みを適用するため、その単語が他の位置に与えた影響は残ります。

`none / comfy` は入力した重みを、`mean / comfy` は平均調整後の重みを、埋め込みの各枠に適用します。どちらも枠数に応じた調整は行いません。重みを付けた複数枠の埋め込みが効きすぎる場合は、対応する `length / comfy` または `length+mean / comfy` を試してください。

## 変化がなくても正常な場合

- **入力の重みがすべて 1：** TokenNorm による追加の重み付けは行われません。なお、`(tag)` は 1.1、`[tag]` は約 0.909 なので、これらも重み指定です。
- **複数枠の Textual Inversion に重みを付けていない：** 境界を正常に取得できていれば、`length` は `No norm` と、`length / comfy` は `none / comfy` と同じ動作になります。
- **チャンク内の本文トークンがすべて同じ重み：** mean によって 1 にそろいます。全体を一律に強調しても、その強調は維持されません。
- **mean に入る時点の重みの平均が 1：** mean による移動はありません。length+mean では、length 適用後の重みについての条件です。

Emphasis mode は reForge のプロンプトのキャッシュ判定に含まれるため、設定を変えると再エンコードされます。選択肢を比較するときは、シードと他の生成設定を固定してください。

## 困ったとき

| 症状 | 確認すること・意味 |
| --- | --- |
| 切り替えても変わらない | 上の条件を確認し、**Apply settings** を押してください。**Use old emphasis implementation** が OFF か、起動時に処理の差し替えに関する警告が出ていないかも確認します。 |
| 起動時の警告にクラス名が出る | プロンプト処理が差し替えられた可能性を検出しています。表示されるのはクラス名で、原因の拡張名とは限りません。既知の競合例に `sd-webui-prevent-artifact` があります。 |
| 画像情報に `Emphasis` が無い | reForge 標準の経路では、処理対象のプロンプトに `(` または `[` があり、かつ `Original` 以外のときに追加されます。`Original` は省略されます。 |
| `TokenNorm comfy fallback` が出る | その画像の一部または全部の処理で、ComfyUI 方式の重み付けを省略しました。値に理由が記録されますが、この項目だけでは影響範囲全体は分かりません。コンソールも確認してください。 |
| `TokenNorm length fallback` が出る | 一部の処理で正確な埋め込みの境界を取得できず、以前の検出方法を使いました。隣接する埋め込みがまとめられたり、誤検出されたりする場合があります。 |

詳しいログが必要な場合は、`webui-user.bat` の**WebUI を起動する行より前**に `set SD_WEBUI_SETI_DEBUG=2` を追加し、再起動してください。`1` は起動情報、`2` は重みの調整や ComfyUI 方式の適用も出力します。無効に戻す場合は、この行を削除して再起動します。

<details>
<summary>技術的な補足：計算式・実装・ComfyUI との違い</summary>

### 処理順序と計算式

重みの調整を先に行い、その後で適用します。length+mean の順序は length → mean → 選択した適用方法です。

mean（1 チャンク内の本文トークンの重み）：

```text
delta = 1 - average(weights)
weights = weights + delta
```

length（n 枠の埋め込み、重み w）：

```text
d = w - 1
w = 1 + sign(d) * sqrt(d * d / n)
```

乗算と ComfyUI 方式。w は調整後の重み、z_empty は空プロンプトの同じ位置のエンコード結果です。

```text
multiply: z_new = z * w
comfy:    z_new = z_empty + (z - z_empty) * w
```

ComfyUI 方式では、調整後の重みが 1 の位置を明示的に保持し、z_empty の引き算・足し算による丸めの変化を避けています。上の式は計算の定義であり、実際の浮動小数点演算には丸めが生じる場合があります。

### WebUI への組み込み

TokenNorm は `modules.sd_emphasis.options` に選択肢を登録します。WebUI のメソッドを差し替えたり、ラップしたりしません。

length では、呼び出し元のチャンクが保持する `PromptChunk.fixes` を読み取ります。一時的な `hijack.fixes` の参照は既に消されていますが、チャンク自体は残っています。呼び出し元のトークン列が Emphasis の受け取ったものと同一オブジェクトであることと、埋め込みの範囲がトークン位置と一致することを確認します。取得・照合に失敗した場合は、以前のトークン列による推定に戻り、`TokenNorm length fallback` を記録します。

### Advanced ノードとの違い

- **mean：** ノードはプロンプト全体で計算しますが、TokenNorm はチャンクごとに独立して計算します。
- **length：** ノードは単語 ID から単語ごとのトークン数を求めます。TokenNorm は Textual Inversion の境界を使い、普通の単語は複数トークンに分かれていても変更しません。
- **A1111 解釈：** ノードは乗算後にエンコード結果の平均を戻します。WebUI の `Original` と同じ考え方です。TokenNorm の乗算の選択肢は `No norm` を土台とし、この補正を行いません。
- **comfy 解釈：** ComfyUI の空プロンプトを基準とする式を使います。ただし、トークン分割など他の処理も異なるため、条件付けの値や画像の一致は保証しません。
- **その他の解釈：** `down_weight`、`compel`、`comfy++` は未実装です。

</details>

---

## License / ライセンス

MIT

## Acknowledgements / 謝辞

- [Shiba-2-shiba](https://note.com/gentle_murre488) — articles that prompted this series of ports / 一連の移植のきっかけとなった記事

## References / 典拠

- [BlenderNeko/ComfyUI_ADV_CLIP_emb — adv_encode.py](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb/blob/master/adv_encode.py) — weight adjustment and interpretation options / 重みの調整と解釈の原典
- [ComfyUI — comfy/sd1_clip.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd1_clip.py) — `ClipTokenWeightEncoder.encode_token_weights`, the ComfyUI application formula / ComfyUI 方式の適用式
- [reForge — modules/sd_emphasis.py](https://github.com/Panchovix/stable-diffusion-webui-reForge/blob/main/modules/sd_emphasis.py) — Emphasis options and application / Emphasis の選択肢と適用処理
- [reForge — modules/sd_hijack_clip.py](https://github.com/Panchovix/stable-diffusion-webui-reForge/blob/main/modules/sd_hijack_clip.py) — prompt chunks and the Emphasis call path / チャンクと Emphasis の呼び出し経路
- [AUTOMATIC1111 — modules/sd_emphasis.py](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_emphasis.py) — upstream Emphasis implementation / 元となる Emphasis の実装
