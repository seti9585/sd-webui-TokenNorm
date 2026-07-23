"""
sd-webui-TokenNorm

Port of the token normalization feature from ComfyUI's
"CLIP Text Encode (Advanced)" (BlenderNeko/ComfyUI_ADV_CLIP_emb) to
Stable Diffusion WebUI reForge.

This extension registers additional entries into modules.sd_emphasis.options.
It does not patch any WebUI internals; it only appends to a module level list.

Provided options:
    - "TokenNorm: mean"        shift token weights so their average becomes 1.0
    - "TokenNorm: length"      divide the weight of multi-token embeddings
    - "TokenNorm: length+mean" length first, then mean

The base behaviour is "No norm" (multiply only, no mean restoration), which
matches the practical SDXL setup. The "Original" (mean restoring) variants are
intentionally not provided in this version.

Upstream reference:
    BlenderNeko/ComfyUI_ADV_CLIP_emb -> adv_encode.py
        shift_mean_weight()  -> the "mean" step
        divide_length() / _norm_mag()  -> the "length" step

Divergence from upstream (documented in README):

    mean:
        Upstream computes the mean over the whole prompt (all chunks
        flattened). This implementation computes it per chunk, because reForge
        instantiates a separate Emphasis object for every chunk and no state can
        be carried over. For prompts that fit in a single 75 token chunk the two
        are identical.

    length:
        Upstream divides a weight by sqrt(number of tokens the WORD was split
        into), using word_id from clip.tokenize(return_word_ids=True). WebUI
        discards word boundaries: modules.sd_hijack_clip.tokenize_line() only
        keeps (token, multiplier) pairs, so a single word split into several
        tokens is indistinguishable from several distinct words that happen to
        share a weight.

        The ONE case where the token count of a "word" survives in WebUI is a
        textual inversion embedding: it occupies several consecutive token
        slots whose id is 0 (placeholders filled later by
        sd_hijack.EmbeddingsWithFixes), all carrying the same multiplier. This
        was confirmed empirically: an N-vector embedding appears as N
        consecutive id==0 tokens.

        Therefore "length" here only rescales runs that consist ENTIRELY of
        id==0 tokens (embeddings). Ordinary tokens are left untouched, which
        also matches upstream behaviour for single-token words (sqrt(1) == 1,
        no change). Multi-word parentheses such as "(fluffy white cat:1.5)" are
        deliberately NOT rescaled, because in WebUI they are a single run of
        ordinary tokens and rescaling them would diverge from upstream, which
        treats each of those words as a separate 1-token word and leaves them
        at 1.5.
"""

import logging
import os
import sys
import traceback

from modules import script_callbacks, shared

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

EXTENSION_NAME = "sd-webui-TokenNorm"
MARKER = "sd_webui_tokennorm_v1"

# CLIP / OpenCLIP BPE end-of-text token id. Used only as a last resort when the
# vocabulary cannot be reached. Both CLIP-L and OpenCLIP-bigG use 49407.
FALLBACK_EOS_ID = 49407

# Textual inversion embedding placeholder id in remade_batch_tokens.
EMBEDDING_TOKEN_ID = 0

DEBUG_ENV_VAR = "SD_WEBUI_SETI_DEBUG"


def _debug_level():
    try:
        return int(os.environ.get(DEBUG_ENV_VAR, "0"))
    except Exception:
        return 0


def _log(level, message):
    """Emit to both logging and stderr. Forge Neo suppresses module level
    loggers, so the stderr print is required for cross-backend visibility."""
    if _debug_level() < level:
        return
    text = "[%s] %s" % (EXTENSION_NAME, message)
    logger.warning(text)
    print(text, file=sys.stderr)


def _warn(message):
    """Unconditional warning, independent of the debug level."""
    text = "[%s] %s" % (EXTENSION_NAME, message)
    logger.warning(text)
    print(text, file=sys.stderr)


# --------------------------------------------------------------------------
# vocabulary access
# --------------------------------------------------------------------------

_eos_id_cache = None
_eos_warned = False


def _iter_tokenizer_candidates():
    """Yield objects that may expose a .tokenizer attribute.

    SDXL: shared.sd_model.cond_stage_model is sgm's GeneralConditioner, and the
    hijacked encoders live in .embedders. SD1.5 lineage: the encoder may sit
    directly on cond_stage_model, possibly behind .wrapped.
    """
    model = getattr(shared, "sd_model", None)
    if model is None:
        return

    csm = getattr(model, "cond_stage_model", None)
    if csm is None:
        return

    embedders = getattr(csm, "embedders", None)
    if embedders is not None:
        try:
            for embedder in embedders:
                yield embedder
        except Exception:
            pass

    yield csm
    yield getattr(csm, "wrapped", None)


def _resolve_eos_id():
    """Look up the id of <|endoftext|> from the tokenizer vocabulary."""
    global _eos_id_cache, _eos_warned

    if _eos_id_cache is not None:
        return _eos_id_cache

    for candidate in _iter_tokenizer_candidates():
        if candidate is None:
            continue
        tokenizer = getattr(candidate, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "get_vocab"):
            continue
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            continue
        eos = vocab.get("<|endoftext|>", None)
        if eos is not None:
            _eos_id_cache = int(eos)
            _log(1, "resolved EOS id %d from vocabulary (size %d)"
                 % (_eos_id_cache, len(vocab)))
            return _eos_id_cache

    if not _eos_warned:
        _warn("could not reach the tokenizer vocabulary; falling back to "
              "EOS id %d" % FALLBACK_EOS_ID)
        _eos_warned = True

    _eos_id_cache = FALLBACK_EOS_ID
    return _eos_id_cache


def _clear_eos_cache(*args, **kwargs):
    global _eos_id_cache, _eos_warned
    _eos_id_cache = None
    _eos_warned = False
    _log(1, "EOS id cache cleared")


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _content_span(row, eos_id, width):
    """Return (start, end) of the content region of one token row.

    Index 0 is BOS. The first occurrence of the EOS id marks the end of the
    content; everything from there on is EOS plus padding.

    Note that the padding in remade_batch_tokens uses id_end, so the first
    occurrence is unambiguous for both CLIP-L (id_pad == id_end) and
    CLIP-G (id_pad == 0).
    """
    try:
        sequence = list(row)
    except Exception:
        return 0, 0

    end = len(sequence)
    for index, token_id in enumerate(sequence):
        if int(token_id) == eos_id:
            end = index
            break

    start = 1
    end = min(end, width)
    if end <= start:
        return 0, 0
    return start, end


def _iter_runs(tokens, multipliers, start, end):
    """Yield (run_start, run_end, token_ids) for maximal runs of equal weight.

    A run is a maximal span [run_start, run_end) of consecutive positions in
    [start, end) whose multiplier is identical. token_ids is the list of token
    ids in that run, used to decide whether the run is an embedding.
    """
    pos = start
    while pos < end:
        run_start = pos
        base = float(multipliers[pos])
        ids = [int(tokens[pos])]
        pos += 1
        while pos < end and abs(float(multipliers[pos]) - base) <= 1e-9:
            ids.append(int(tokens[pos]))
            pos += 1
        yield run_start, pos, ids


# --------------------------------------------------------------------------
# mean
# --------------------------------------------------------------------------

def shift_mean_weight(tokens, multipliers, eos_id):
    """Additive mean shift, per chunk.

    Upstream equivalent:
        delta = 1 - mean(weights where word_id != 0)
        weights = weights + delta   (for word_id != 0)

    Here "word_id != 0" is approximated by "not BOS, not EOS, not padding".
    Word boundaries are not needed for the mean variant.
    """
    result = multipliers.clone()
    width = result.shape[-1] if result.dim() >= 1 else 0

    rows = min(len(tokens), result.shape[0]) if result.dim() >= 2 else 0
    if rows == 0:
        return multipliers

    for row_index in range(rows):
        start, end = _content_span(tokens[row_index], eos_id, width)
        if end <= start:
            continue

        segment = result[row_index, start:end]
        delta = 1.0 - segment.mean()
        result[row_index, start:end] = segment + delta

        _log(2, "mean: row %d span [%d, %d) delta %s"
             % (row_index, start, end, format(float(delta), ".6f")))

    return result


# --------------------------------------------------------------------------
# length
# --------------------------------------------------------------------------

def _norm_mag(w, n):
    """Upstream _norm_mag: keep the sign of the deviation from 1.0, scale the
    magnitude of that deviation by 1/sqrt(n).

        d = w - 1
        return 1 + sign(d) * sqrt(d*d / n)

    For n == 1 this returns w unchanged.
    """
    d = w - 1.0
    if d == 0.0:
        return w
    sign = 1.0 if d > 0.0 else -1.0
    return 1.0 + sign * ((d * d) / n) ** 0.5


def divide_length(tokens, multipliers, eos_id):
    """Divide the weight of multi-token embeddings, per chunk.

    Only runs that consist entirely of embedding placeholder tokens
    (id == EMBEDDING_TOKEN_ID) are rescaled. Their token count n is the number
    of vectors the embedding expands to. Ordinary token runs are left
    untouched, matching upstream behaviour for single-token words.
    """
    result = multipliers.clone()
    width = result.shape[-1] if result.dim() >= 1 else 0

    rows = min(len(tokens), result.shape[0]) if result.dim() >= 2 else 0
    if rows == 0:
        return multipliers

    for row_index in range(rows):
        start, end = _content_span(tokens[row_index], eos_id, width)
        if end <= start:
            continue

        token_row = tokens[row_index]
        for run_start, run_end, ids in _iter_runs(
                token_row, result[row_index], start, end):
            n = run_end - run_start
            if n <= 1:
                continue
            # rescale only pure embedding runs
            if not all(i == EMBEDDING_TOKEN_ID for i in ids):
                continue
            old_w = float(result[row_index, run_start])
            new_w = _norm_mag(old_w, n)
            result[row_index, run_start:run_end] = new_w

            _log(2, "length: row %d run [%d, %d) n=%d w %s -> %s"
                 % (row_index, run_start, run_end, n,
                    format(old_w, ".6f"), format(new_w, ".6f")))

    return result


# --------------------------------------------------------------------------
# Emphasis classes
# --------------------------------------------------------------------------

_registration_error = None

try:
    from modules import sd_emphasis

    class EmphasisTokenNormMean(sd_emphasis.EmphasisOriginalNoNorm):
        name = "TokenNorm: mean"
        description = ("ComfyUI token normalization (mean). Shifts all token "
                       "weights so their average becomes 1.0, then applies "
                       "them without mean restoration. Per chunk.")

        def after_transformers(self):
            try:
                eos_id = _resolve_eos_id()
                self.multipliers = shift_mean_weight(
                    self.tokens, self.multipliers, eos_id)
            except Exception:
                # Never abort generation. Fall through with untouched weights.
                _warn("mean normalization failed, weights left unchanged:\n"
                      + traceback.format_exc())
            super().after_transformers()

    class EmphasisTokenNormLength(sd_emphasis.EmphasisOriginalNoNorm):
        name = "TokenNorm: length"
        description = ("ComfyUI token normalization (length). Divides the "
                       "weight of multi-token textual inversion embeddings so "
                       "the magnitude of the weight change is constant "
                       "regardless of vector count. Ordinary tokens unchanged.")

        def after_transformers(self):
            try:
                eos_id = _resolve_eos_id()
                self.multipliers = divide_length(
                    self.tokens, self.multipliers, eos_id)
            except Exception:
                _warn("length normalization failed, weights left unchanged:\n"
                      + traceback.format_exc())
            super().after_transformers()

    class EmphasisTokenNormLengthMean(sd_emphasis.EmphasisOriginalNoNorm):
        name = "TokenNorm: length+mean"
        description = ("ComfyUI token normalization (length + mean). Applies "
                       "length division to embeddings first, then shifts the "
                       "mean to 1.0. Per chunk.")

        def after_transformers(self):
            try:
                eos_id = _resolve_eos_id()
                weights = divide_length(
                    self.tokens, self.multipliers, eos_id)
                self.multipliers = shift_mean_weight(
                    self.tokens, weights, eos_id)
            except Exception:
                _warn("length+mean normalization failed, weights left "
                      "unchanged:\n" + traceback.format_exc())
            super().after_transformers()

    _TOKENNORM_OPTIONS = [EmphasisTokenNormMean,
                          EmphasisTokenNormLength,
                          EmphasisTokenNormLengthMean]

except Exception:
    sd_emphasis = None
    _TOKENNORM_OPTIONS = []
    _registration_error = traceback.format_exc()


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def _register_options():
    if sd_emphasis is None:
        _warn("modules.sd_emphasis is unavailable. This extension targets "
              "reForge and will not work on this backend.\n"
              + str(_registration_error))
        return

    existing = set()
    for option in sd_emphasis.options:
        existing.add(getattr(option, "name", None))

    for option in _TOKENNORM_OPTIONS:
        if option.name in existing:
            _log(1, "option already registered, skipping: %s" % option.name)
            continue
        setattr(option, "_sd_webui_tokennorm_marker", MARKER)
        sd_emphasis.options.append(option)
        _log(1, "registered option: %s" % option.name)


_register_options()


# --------------------------------------------------------------------------
# liveness check
# --------------------------------------------------------------------------

def _collect_process_tokens_overrides():
    """Find classes that define process_tokens in their own __dict__.

    In an unmodified reForge only TextConditionalModel defines it. Anything
    else means a third party has replaced the method at class level, in which
    case sd_emphasis is never consulted and this extension has no effect.
    """
    offenders = []
    seen = set()

    module_names = ["modules.sd_hijack_clip",
                    "modules.sd_hijack_open_clip",
                    "modules_forge.forge_clip"]

    for module_name in module_names:
        module = sys.modules.get(module_name, None)
        if module is None:
            continue
        for attribute_name in dir(module):
            candidate = getattr(module, attribute_name, None)
            if not isinstance(candidate, type):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.__name__ == "TextConditionalModel":
                continue
            if "process_tokens" in candidate.__dict__:
                offenders.append("%s.%s" % (module_name, candidate.__name__))

    return offenders


def _on_app_started(demo, app):
    try:
        offenders = _collect_process_tokens_overrides()
    except Exception:
        _log(1, "liveness check failed:\n" + traceback.format_exc())
        return

    if not offenders:
        _log(1, "liveness check passed; sd_emphasis path is intact")
        return

    _warn("WARNING: process_tokens has been replaced at class level by "
          "another extension (%s). The sd_emphasis mechanism is bypassed, so "
          "Emphasis settings including TokenNorm have NO effect. Known cause: "
          "sd-webui-prevent-artifact. Remove it and select the equivalent "
          "built-in Emphasis option instead." % ", ".join(offenders))


try:
    script_callbacks.on_app_started(_on_app_started)
    script_callbacks.on_model_loaded(_clear_eos_cache)
except Exception:
    _warn("failed to register script callbacks:\n" + traceback.format_exc())
