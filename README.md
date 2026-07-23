# sd-webui-TokenNorm

**EN** | [日本語](#日本語)

Token weight normalization for Stable Diffusion WebUI (Forge-based).
Adds ComfyUI's token normalization methods as additional **Emphasis** options.

Original implementation by **BlenderNeko** — [ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)

> This extension only appends entries to `modules.sd_emphasis.options`.
> It does not patch any WebUI internals.

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TokenNorm
```

---

## Usage

**Settings → User interface → Emphasis**

Three options are added. All of them build on **No norm** (multiply only, no mean restoration).

| Option | Effect |
| ------ | ------ |
| `TokenNorm: mean` | Shifts all token weights so their average becomes 1.0 |
| `TokenNorm: length` | Divides the weight of multi-vector textual inversion embeddings |
| `TokenNorm: length+mean` | `length` first, then `mean` |

No per-generation UI is added. Select an option and generate.

---

## Algorithm

**mean** — additive shift, magnitude of the prompt is normalized while the
relative differences between tokens are preserved.

```
delta   = 1 − mean(content weights)
weights = weights + delta
```

**length** — an N-vector embedding occupies N token slots, so a weight of 1.5
is applied N times and the emphasis is inflated in proportion to N. The
deviation from 1.0 is divided by sqrt(N).

```
d = w − 1
w = 1 + sign(d) × sqrt(d² / N)
```

Example: a 26-vector embedding at weight 1.5 becomes 1.098.

---

## Differences from upstream

**mean is computed per chunk.**
Upstream flattens the whole prompt and takes a single mean. WebUI creates a
separate `Emphasis` instance for every 75-token chunk and no state can be
carried across them. For prompts that fit in one chunk the two are identical.

**length applies to embeddings only.**
Upstream divides by the number of tokens a *word* was split into, using
`word_id` from `clip.tokenize(return_word_ids=True)`. WebUI discards word
boundaries — `modules.sd_hijack_clip.tokenize_line()` keeps only
`(token, multiplier)` pairs — so a word split into several tokens cannot be
told apart from several distinct words sharing one weight.

The one case where the token count survives is a textual inversion embedding:
it occupies N consecutive slots with token id `0`, all carrying the same
multiplier. `length` therefore rescales only runs that consist entirely of
id `0` tokens.

Ordinary tokens are left untouched. This matches upstream for single-token
words, where sqrt(1) = 1 changes nothing. Multi-word parentheses such as
`(fluffy white cat:1.5)` are deliberately **not** rescaled: upstream treats
those as three separate 1-token words and leaves them at 1.5.

---

## Troubleshooting

If another extension replaces `process_tokens` at class level, `sd_emphasis` is
never consulted and **all** Emphasis settings — built-in ones included — stop
having any effect. This extension detects that at startup and prints a warning.

Set `SD_WEBUI_SETI_DEBUG` in `webui-user.bat` for diagnostics.

```bat
set SD_WEBUI_SETI_DEBUG=2
```

| Level | Output |
| ----- | ------ |
| 1 | registration, EOS resolution, liveness check |
| 2 | per-chunk delta and per-run rescale values |

---

## Requirements

- reForge / Forge-based WebUI exposing `modules.sd_emphasis`
- SD 1.x or SDXL lineage checkpoints

---
---

# 日本語

**[English](#sd-webui-tokennorm)** | 日本語

Forge 系 WebUI 向けのトークン重み正規化拡張機能。
ComfyUI のトークン正規化手法を **Emphasis** の選択肢として追加します。

原実装：**BlenderNeko** — [ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)

> 本拡張機能は `modules.sd_emphasis.options` に項目を追加するのみです。
> WebUI 本体への差し替えは一切行いません。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TokenNorm
```

---

## 使い方

**Settings → User interface → Emphasis**

3 つの選択肢が追加されます。いずれも **No norm**（乗算のみ、平均復元なし）を土台としています。

| 選択肢 | 効果 |
| --- | --- |
| `TokenNorm: mean` | 全トークンの重みの平均が 1.0 になるようシフト |
| `TokenNorm: length` | 複数ベクトルの Textual Inversion 埋め込みの重みを割り引く |
| `TokenNorm: length+mean` | `length` を適用した後に `mean` を適用 |

生成画面への UI 追加はありません。選択して生成するだけです。

---

## アルゴリズム

**mean** — 加算シフト。トークン間の相対差を保ったまま、プロンプト全体のマグニチュードを揃えます。

```
delta   = 1 − 平均(内容トークンの重み)
weights = weights + delta
```

**length** — N ベクトルの埋め込みは N 個のトークン枠を占めるため、重み 1.5 が N 回適用され、強調が N に比例して水増しされます。1.0 からの隔たりを sqrt(N) で割ります。

```
d = w − 1
w = 1 + sign(d) × sqrt(d² / N)
```

例：26 ベクトルの埋め込みに重み 1.5 を指定した場合、1.098 になります。

---

## 原実装との相違点

**mean はチャンク単位で計算されます。**
原実装はプロンプト全体を平坦化して 1 つの平均を取ります。WebUI は 75 トークンのチャンクごとに `Emphasis` インスタンスを生成し直すため、チャンクをまたいだ状態を保持できません。1 チャンクに収まるプロンプトでは両者は一致します。

**length は埋め込みのみを対象とします。**
原実装は `clip.tokenize(return_word_ids=True)` から得た `word_id` を使い、単語が分割されたトークン数で割ります。WebUI は単語境界を保持しません。`modules.sd_hijack_clip.tokenize_line()` が `(token, multiplier)` の組しか残さないため、1 単語が複数トークンに割れた場合と、同じ重みを持つ複数の単語とを区別できません。

トークン数が残る唯一のケースが Textual Inversion 埋め込みです。トークン ID `0` の枠を N 個連続で占有し、すべてが同じ重みを持ちます。そのため `length` は、全体が ID `0` で構成される連続領域のみを対象とします。

通常のトークンには一切変更を加えません。これは 1 トークンの単語に対して sqrt(1) = 1 で何も変わらない原実装の挙動と一致します。`(fluffy white cat:1.5)` のような複数単語の括弧は意図的に**対象外**としています。原実装ではこれを 3 つの独立した 1 トークン単語として扱い、1.5 のまま変更しないためです。

---

## トラブルシューティング

他の拡張機能が `process_tokens` をクラスレベルで差し替えている場合、`sd_emphasis` が呼ばれなくなり、組み込みのものも含めて **すべての** Emphasis 設定が無効になります。本拡張機能は起動時にこれを検出して警告を出力します。

診断情報は `webui-user.bat` に `SD_WEBUI_SETI_DEBUG` を設定すると出力されます。

```bat
set SD_WEBUI_SETI_DEBUG=2
```

| レベル | 出力内容 |
| --- | --- |
| 1 | 登録、EOS 解決、生存確認 |
| 2 | チャンクごとの delta、連続領域ごとの補正値 |

---

## 動作環境

- `modules.sd_emphasis` を持つ reForge / Forge 系 WebUI
- SD 1.x 系または SDXL 系のチェックポイント

---

## ライセンス

MIT License — Original implementation © BlenderNeko

### 典拠

- [BlenderNeko/ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb) — `adv_encode.py` の `shift_mean_weight()` / `divide_length()` / `_norm_mag()`
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) — `modules/sd_emphasis.py`, `modules/sd_hijack_clip.py`
