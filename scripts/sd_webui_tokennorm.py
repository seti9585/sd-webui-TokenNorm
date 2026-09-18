"""
sd-webui-TokenNorm

Port of the token normalization and weight interpretation features from
ComfyUI's "CLIP Text Encode (Advanced)"
(BlenderNeko/ComfyUI_ADV_CLIP_emb) to Stable Diffusion WebUI reForge.

This extension registers additional entries into modules.sd_emphasis.options.
It does not patch any WebUI internals; it only appends to a module level list.

The upstream node exposes two independent axes:

    token normalization   how the weight VALUES are conditioned
    weight interpretation how a weight is APPLIED to the embedding

Provided options:
    - "TokenNorm: mean"         shift token weights so their average is 1.0
    - "TokenNorm: length"       divide the weight of multi-token embeddings
    - "TokenNorm: length+mean"  length first, then mean
    - the same three with a " / comfy" suffix, plus
      "TokenNorm: none / comfy"

Together with the built-in "No norm" these cover the eight cells of the
upstream matrix that this extension targets:

    normalization   A1111 style               comfy
    none            "No norm" (built in)      + "none / comfy"
    mean            "TokenNorm: mean"         + "mean / comfy"
    length          "TokenNorm: length"       + "length / comfy"
    length+mean     "TokenNorm: length+mean"  + "length+mean / comfy"

where "+" is shorthand for the "TokenNorm: " prefix.

Normalization always runs before interpretation, as upstream does: the
normalization step conditions the weight values, the interpretation step
applies them.

The three normalization options use the "No norm" application (multiply only,
no mean restoration), which matches the practical SDXL setup and corresponds to
upstream's "A1111" interpretation minus the mean restoration step. The
"Original" (mean restoring) variants are intentionally not provided.

The "/ comfy" suffix denotes the interpretation axis. An option without a
suffix uses the A1111 style interpretation. The suffix-free name of the
"no normalization, A1111 interpretation" cell is the built-in "No norm", so it
is not duplicated here.

Upstream reference:
    BlenderNeko/ComfyUI_ADV_CLIP_emb -> adv_encode.py
        shift_mean_weight()  -> the "mean" step
        divide_length() / _norm_mag()  -> the "length" step
    ComfyUI -> comfy/sd1_clip.py
        ClipTokenWeightEncoder.encode_token_weights() -> the "comfy" step.
        adv_encode.py delegates the comfy interpretation to ComfyUI itself, so
        the formula lives there rather than in the upstream extension.

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

        Therefore "length" here only rescales textual inversion embeddings.
        Ordinary tokens are left untouched, which also matches upstream
        behaviour for single-token words (sqrt(1) == 1, no change). Multi-word
        parentheses such as "(fluffy white cat:1.5)" are deliberately NOT
        rescaled, because in WebUI they are a single run of ordinary tokens and
        rescaling them would diverge from upstream, which treats each of those
        words as a separate 1-token word and leaves them at 1.5.

        Embedding boundaries (v3):
        The id==0 placeholders alone cannot separate two embeddings that sit
        next to each other with the same weight; they merge into one run. With
        many embeddings in a row this was measured at n=75, which crushed a
        weight of 1.1 down to 1.0115. The exact boundaries do exist in reForge:
        tokenize_line() records one PromptChunkFix(offset, embedding) per
        embedding in PromptChunk.fixes. hijack.fixes is cleared by
        EmbeddingsWithFixes.forward() before the Emphasis object runs, but the
        PromptChunk objects themselves are still alive in the caller's frame
        (TextConditionalModel.forward, local "batch_chunk"). This extension
        reads them from there WITHOUT patching anything, the same technique the
        reForge built-in Differential Diffusion uses to find its sigmas. The
        frame is accepted only if its local "tokens" is the very same list
        object as Emphasis.tokens, so an unrelated frame can never be used.

        Each embedding then gets its own n = embedding.vectors, placed at
        row position offset + 1 (index 0 of a row is BOS), which is exactly
        where EmbeddingsWithFixes inserts the vectors. As a side effect,
        ordinary "!" tokens (BPE id 0) are no longer mistaken for embeddings.

        If the chunk list cannot be found or does not match the token rows
        (for example a third party wrapper rebuilt the token list), the old
        run based detection is used for that generation, a warning is printed
        once, and the infotext receives "TokenNorm length fallback".

    comfy:
        None known. The reference point z_empty is the encoder output for an
        empty chunk, which reForge produces as

            [id_start, id_end, id_pad * chunk_length]

        after process_tokens() overwrites everything past the first id_end with
        id_pad. This is the same sequence as ComfyUI's gen_empty_tokens(), and
        it was verified on hardware by comparing it against the z of an
        actually empty prompt: identical to the last digit for both CLIP-L and
        CLIP-G. The distinction matters only for CLIP-G, where id_pad differs
        from id_end.

        z_empty depends solely on the encoder, the checkpoint and the CLIP skip
        setting, never on the prompt. It is therefore computed once and cached
        on the encoder object. This was verified on hardware across four
        prompts, two checkpoints and two CLIP skip settings: the maximum
        absolute difference between recomputations was exactly 0.

        Note that the pooled output is not affected by weights in ComfyUI
        either; ClipTokenWeightEncoder returns the pooled tensor of the first
        section untouched. reForge likewise carries pooled around the Emphasis
        step in process_tokens(), so nothing is needed here.
"""

import inspect
import logging
import os
import sys
import traceback

import torch

from modules import devices, script_callbacks, scripts, shared

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

EXTENSION_NAME = "sd-webui-TokenNorm"
MARKER = "sd_webui_tokennorm_v3"

# CLIP / OpenCLIP BPE end-of-text token id. Used only as a last resort when the
# vocabulary cannot be reached. Both CLIP-L and OpenCLIP-bigG use 49407.
FALLBACK_EOS_ID = 49407

# Textual inversion embedding placeholder id in remade_batch_tokens.
EMBEDDING_TOKEN_ID = 0

DEBUG_ENV_VAR = "SD_WEBUI_SETI_DEBUG"

# Attribute used to cache the empty chunk embedding on an encoder object. One
# slot per encoder; a changed key overwrites it, so nothing accumulates and no
# explicit invalidation is required.
Z_EMPTY_CACHE_ATTR = "_sd_webui_tokennorm_z_empty"

# Written into the infotext when the comfy interpretation could not be applied.
# Absence of this key means the interpretation was applied normally.
INFOTEXT_FALLBACK_KEY = "TokenNorm comfy fallback"

# Written into the infotext when the length step could not find the exact
# embedding boundaries and had to use the old run based detection.
INFOTEXT_LENGTH_FALLBACK_KEY = "TokenNorm length fallback"

# How many caller frames to inspect when looking for the PromptChunk list.
# The real chain is: helper -> after_transformers -> process_tokens -> forward,
# so a small number is enough. The cap only bounds the cost in odd setups.
FRAME_SEARCH_DEPTH = 12


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


# Once-per-generation warning flag for the length fallback. Reset by the
# companion Script in process().
_length_warned = False


def _find_chunk_fixes(tokens):
    """Recover the PromptChunkFix lists of the chunk being encoded.

    Walks up the call stack looking for TextConditionalModel.forward, which
    holds the PromptChunk objects of the current chunk in its local
    "batch_chunk" and passes "tokens" (built from them) straight on to
    process_tokens() -> Emphasis.tokens. The frame is accepted only when its
    "tokens" local IS the same list object as the tokens given here.

    Returns (fixes_per_row, reason). fixes_per_row is a list with one list of
    (offset, embedding) pairs per row; reason is None on success.
    """
    frame = inspect.currentframe()
    try:
        depth = 0
        current = frame.f_back if frame is not None else None
        while current is not None and depth < FRAME_SEARCH_DEPTH:
            local_vars = current.f_locals
            batch_chunk = local_vars.get("batch_chunk", None)
            frame_tokens = local_vars.get("tokens", None)
            if batch_chunk is not None and frame_tokens is tokens:
                try:
                    fixes = [list(getattr(chunk, "fixes", None) or [])
                             for chunk in batch_chunk]
                except Exception:
                    return None, "chunk list is not iterable"
                if len(fixes) != len(tokens):
                    return None, ("chunk count %d does not match row count %d"
                                  % (len(fixes), len(tokens)))
                return fixes, None
            current = current.f_back
            depth += 1
        return None, "caller frame holding the chunk list was not found"
    except Exception:
        return None, "frame inspection failed"
    finally:
        # Break the reference cycle created by holding a frame object.
        del frame


def _record_length_fallback(reason):
    """Warn once per generation and mark the infotext."""
    global _length_warned

    if not _length_warned:
        _warn("length: exact embedding boundaries unavailable (%s); using "
              "the old run based detection. Adjacent embeddings with the same "
              "weight may be merged." % reason)
        _length_warned = True

    processing = _current_processing
    if processing is None:
        return
    try:
        processing.extra_generation_params[INFOTEXT_LENGTH_FALLBACK_KEY] = reason
    except Exception:
        pass


def _row_spans_from_fixes(token_row, row_fixes, start, end):
    """Convert one row's PromptChunkFix list into (span_start, span_end, name).

    Row position = offset + 1, because index 0 of a row is BOS; this is the
    same placement EmbeddingsWithFixes.forward() uses. Returns None when any
    fix does not line up with id==0 placeholders, so the caller can fall back
    for this row instead of rescaling the wrong tokens.
    """
    spans = []
    for fix in row_fixes:
        try:
            offset, embedding = fix[0], fix[1]
            vectors = int(getattr(embedding, "vectors", 0))
            name = str(getattr(embedding, "name", "?"))
        except Exception:
            return None
        if vectors <= 0:
            return None
        span_start = int(offset) + 1
        span_end = span_start + vectors
        # tokenize_line() starts a new chunk when an embedding would not fit,
        # so a genuine fix never runs past the content end. A span that does
        # means the fixes do not belong to these tokens.
        if span_start < start or span_end > end:
            return None
        for position in range(span_start, span_end):
            if int(token_row[position]) != EMBEDDING_TOKEN_ID:
                return None
        spans.append((span_start, span_end, name))
    return spans


def _divide_length_by_fixes(tokens, multipliers, eos_id, fixes):
    """Length step using exact embedding boundaries.

    Returns (result, bad_rows). bad_rows lists the row indices whose fixes did
    not line up; those rows are left untouched here and handled by the legacy
    path in divide_length().
    """
    result = multipliers.clone()
    width = result.shape[-1] if result.dim() >= 1 else 0

    rows = min(len(tokens), result.shape[0]) if result.dim() >= 2 else 0
    bad_rows = []

    for row_index in range(rows):
        start, end = _content_span(tokens[row_index], eos_id, width)
        if end <= start:
            continue

        token_row = tokens[row_index]
        spans = _row_spans_from_fixes(token_row, fixes[row_index], start, end)
        if spans is None:
            bad_rows.append(row_index)
            continue

        for span_start, span_end, name in spans:
            n = span_end - span_start
            if n <= 1:
                continue
            old_w = float(result[row_index, span_start])
            new_w = _norm_mag(old_w, n)
            result[row_index, span_start:span_end] = new_w

            _log(2, "length: row %d embedding '%s' [%d, %d) n=%d w %s -> %s"
                 % (row_index, name, span_start, span_end, n,
                    format(old_w, ".6f"), format(new_w, ".6f")))

    return result, bad_rows


def _divide_length_legacy(tokens, multipliers, eos_id, only_rows=None):
    """Old run based length step (v2 behaviour), used only as a fallback.

    Only runs that consist entirely of embedding placeholder tokens
    (id == EMBEDDING_TOKEN_ID) are rescaled. Adjacent embeddings with the same
    weight cannot be told apart here and are merged into one run.
    """
    result = multipliers.clone()
    width = result.shape[-1] if result.dim() >= 1 else 0

    rows = min(len(tokens), result.shape[0]) if result.dim() >= 2 else 0
    if rows == 0:
        return multipliers

    for row_index in range(rows):
        if only_rows is not None and row_index not in only_rows:
            continue
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

            _log(2, "length (legacy): row %d run [%d, %d) n=%d w %s -> %s"
                 % (row_index, run_start, run_end, n,
                    format(old_w, ".6f"), format(new_w, ".6f")))

    return result


def divide_length(tokens, multipliers, eos_id):
    """Divide the weight of multi-token embeddings, per chunk.

    Each textual inversion embedding is rescaled by its own vector count n:

        w_new = 1 + sign(w - 1) * sqrt((w - 1)^2 / n)

    Exact boundaries come from the PromptChunk fixes (see _find_chunk_fixes).
    Rows without usable fixes fall back to the old run based detection.
    Ordinary tokens are never changed.
    """
    rows = 0
    try:
        rows = min(len(tokens), multipliers.shape[0]) \
            if multipliers.dim() >= 2 else 0
    except Exception:
        rows = 0
    if rows == 0:
        return multipliers

    fixes, reason = _find_chunk_fixes(tokens)
    if fixes is None:
        _record_length_fallback(reason)
        return _divide_length_legacy(tokens, multipliers, eos_id)

    result, bad_rows = _divide_length_by_fixes(
        tokens, multipliers, eos_id, fixes)
    if bad_rows:
        _record_length_fallback("fixes did not line up with the tokens in "
                                "row(s) %s" % ", ".join(str(r) for r in bad_rows))
        result = _divide_length_legacy(tokens, result, eos_id,
                                       only_rows=set(bad_rows))
    return result


# --------------------------------------------------------------------------
# comfy weight interpretation
# --------------------------------------------------------------------------

# Set by the companion Script so that a failure can be recorded in the
# infotext. None outside of a generation.
_current_processing = None

# One warning per generation instead of one per chunk.
_comfy_warned = False

# Defensive guard. encode_with_transformers() does not go through
# process_tokens(), so after_transformers() cannot be re-entered by the empty
# chunk encode. The flag only exists so that a third party wrapper that routes
# it back would degrade to a pass-through instead of recursing forever.
_in_empty_encode = False


def _iter_encoder_candidates():
    """Yield objects that may be a hijacked text encoder.

    SDXL: shared.sd_model.cond_stage_model is sgm's GeneralConditioner and the
    encoders live in .embedders (index 0 CLIP-L, index 1 CLIP-G, the rest are
    ConcatTimestepEmbedderND). SD1.5 lineage: the encoder may sit directly on
    cond_stage_model.
    """
    model = getattr(shared, "sd_model", None)
    if model is None:
        return

    conditioner = getattr(model, "cond_stage_model", None)
    if conditioner is None:
        return

    embedders = getattr(conditioner, "embedders", None)
    if embedders is not None:
        try:
            for embedder in embedders:
                yield embedder
        except Exception:
            pass

    yield conditioner


def _is_text_encoder(candidate):
    if candidate is None:
        return False
    if not callable(getattr(candidate, "encode_with_transformers", None)):
        return False
    for attribute in ("id_start", "id_end", "chunk_length"):
        if getattr(candidate, attribute, None) is None:
            return False
    return True


def _build_empty_tokens(encoder):
    """Token row that reForge produces for a chunk of an empty prompt.

    tokenize_line() pads the content to chunk_length with id_end and wraps it
    in id_start / id_end, then process_tokens() replaces everything after the
    first id_end with id_pad. With no content that yields

        [id_start, id_end, id_pad * chunk_length]

    which is chunk_length + 2 tokens long, matching the sequence length the
    encoder is fed during a normal generation.
    """
    id_start = int(getattr(encoder, "id_start"))
    id_end = int(getattr(encoder, "id_end"))
    id_pad = getattr(encoder, "id_pad", None)
    id_pad = id_end if id_pad is None else int(id_pad)
    chunk_length = int(getattr(encoder, "chunk_length"))
    return [id_start, id_end] + [id_pad] * chunk_length


def _z_empty_key():
    """Everything z_empty depends on besides the encoder object itself.

    The encoder is identified by the object the slot is attached to, so it does
    not appear here. CLIP skip is included even though it was measured to have
    no effect on the SDXL path: keeping it costs one extra recomputation on a
    setting change, whereas omitting it would silently serve a stale tensor if
    that ever stops being true.
    """
    checkpoint = "unknown"
    try:
        info = getattr(shared.sd_model, "sd_checkpoint_info", None)
        if info is not None:
            checkpoint = str(getattr(info, "name", None)
                             or getattr(info, "title", None))
    except Exception:
        pass

    try:
        clip_skip = int(getattr(shared.opts, "CLIP_stop_at_last_layers", 1))
    except Exception:
        clip_skip = -1

    return (checkpoint, clip_skip)


def _get_z_empty(encoder):
    """Return the encoder output for an empty chunk, cached on the encoder."""
    global _in_empty_encode

    key = _z_empty_key()

    slot = getattr(encoder, Z_EMPTY_CACHE_ATTR, None)
    if isinstance(slot, dict) and slot.get("key") == key:
        cached = slot.get("z", None)
        if cached is not None:
            return cached

    tokens = _build_empty_tokens(encoder)

    _in_empty_encode = True
    try:
        with torch.no_grad():
            row = torch.asarray([tokens]).to(devices.device)
            z_empty = encoder.encode_with_transformers(row)
    finally:
        _in_empty_encode = False

    z_empty = z_empty.detach()
    setattr(encoder, Z_EMPTY_CACHE_ATTR, {"key": key, "z": z_empty})

    _log(1, "computed z_empty for %s: shape %s, key %s"
         % (type(encoder).__name__, tuple(z_empty.shape), (key,)))

    return z_empty


def _resolve_z_empty(z):
    """Find the empty chunk embedding matching z.

    Returns (z_empty, reason). reason is None on success.

    The encoder is identified by output shape rather than by intercepting the
    call, so this extension still patches nothing. SDXL exposes 768 and 1280,
    SD1.5 only 768 and SD2.1 only 1024, so the match is unique in practice. An
    ambiguous or empty match is reported instead of guessed.
    """
    hidden_size = int(z.shape[-1])
    sequence_length = int(z.shape[-2])

    matches = []
    inspected = 0

    for candidate in _iter_encoder_candidates():
        if not _is_text_encoder(candidate):
            continue
        inspected += 1
        try:
            z_empty = _get_z_empty(candidate)
        except Exception:
            _log(1, "empty chunk encode failed for %s:\n%s"
                 % (type(candidate).__name__, traceback.format_exc()))
            continue
        if int(z_empty.shape[-1]) != hidden_size:
            continue
        if int(z_empty.shape[-2]) != sequence_length:
            continue
        matches.append(z_empty)

    if len(matches) == 1:
        return matches[0], None

    if not matches:
        return None, ("no text encoder produced a %d wide empty chunk "
                      "(%d inspected)" % (hidden_size, inspected))

    return None, ("%d text encoders produced a %d wide empty chunk; "
                  "ambiguous" % (len(matches), hidden_size))


def apply_comfy_interpretation(z, multipliers):
    """ComfyUI weight interpretation.

    ComfyUI, comfy/sd1_clip.py, ClipTokenWeightEncoder.encode_token_weights:

        z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]

    applied only where the weight differs from 1.0. z_empty is the encoding of
    an empty chunk, so it is a per position reference rather than a single
    vector: weight 0.0 does not silence a token, it returns that position to
    what an empty prompt would have produced.

    The selection is done with torch.where rather than relying on a weight of
    1.0 being a no-op, because (z - z_empty) + z_empty is not guaranteed to
    round back to z. Untouched positions stay bit identical.

    Returns (new_z, reason). reason is None on success; when it is not, z is
    returned unchanged and the weights have no effect.
    """
    if multipliers is None:
        return z, "multipliers unavailable"

    weights = multipliers.to(device=z.device, dtype=z.dtype)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)

    if int(weights.shape[-1]) != int(z.shape[-2]):
        return z, ("weight length %d does not match sequence length %d"
                   % (int(weights.shape[-1]), int(z.shape[-2])))

    changed = weights != 1.0
    if not bool(changed.any()):
        # Nothing to interpret. Skipping here also means an empty prompt never
        # needs an encoder lookup, which matters during model load: the empty
        # prompt is encoded before shared.sd_model is fully assigned.
        return z, None

    z_empty, reason = _resolve_z_empty(z)
    if z_empty is None:
        return z, reason

    z_empty = z_empty.to(device=z.device, dtype=z.dtype)

    factor = weights.unsqueeze(-1)
    blended = z_empty + (z - z_empty) * factor

    _log(2, "comfy: %d of %d positions reweighted"
         % (int(changed.sum()), int(changed.numel())))

    return torch.where(changed.unsqueeze(-1), blended, z), None


def _record_comfy_failure(reason):
    """Warn once and mark the infotext.

    The failure mode is a pass-through, which is indistinguishable from a
    correct result by eye. Recording it in the infotext is what makes an
    affected image identifiable afterwards.
    """
    global _comfy_warned

    if not _comfy_warned:
        _warn("comfy interpretation unavailable (%s); token weights have no "
              "effect for this generation" % reason)
        _comfy_warned = True

    processing = _current_processing
    if processing is None:
        return
    try:
        processing.extra_generation_params[INFOTEXT_FALLBACK_KEY] = reason
    except Exception:
        pass


def _apply_comfy_to(emphasis):
    """Shared body of every "/ comfy" cell.

    Replaces the multiplication that the A1111 style cells perform. Any
    failure degrades to a pass-through, which leaves the token weights without
    effect but never aborts generation.
    """
    if _in_empty_encode:
        return

    try:
        new_z, reason = apply_comfy_interpretation(
            emphasis.z, emphasis.multipliers)
    except Exception:
        _record_comfy_failure("unhandled exception, see console")
        _warn("comfy interpretation failed, token weights ignored:\n"
              + traceback.format_exc())
        return

    if reason is not None:
        _record_comfy_failure(reason)
        return

    emphasis.z = new_z


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

    class EmphasisTokenNormNoneComfy(sd_emphasis.Emphasis):
        name = "TokenNorm: none / comfy"
        description = ("ComfyUI weight interpretation, no token "
                       "normalization. Interpolates each weighted token "
                       "towards the embedding of an empty prompt instead of "
                       "scaling it, reproducing what ComfyUI does with the "
                       "same prompt.")

        def after_transformers(self):
            # Note that the "/ comfy" cells do NOT derive from
            # EmphasisOriginalNoNorm: in the comfy interpretation the
            # interpolation IS the application of the weight, so no
            # multiplication takes place.
            _apply_comfy_to(self)

    class EmphasisTokenNormMeanComfy(sd_emphasis.Emphasis):
        name = "TokenNorm: mean / comfy"
        description = ("ComfyUI token normalization (mean) with the ComfyUI "
                       "weight interpretation. Shifts all token weights so "
                       "their average becomes 1.0, then interpolates towards "
                       "the empty prompt embedding. Per chunk.")

        def after_transformers(self):
            try:
                eos_id = _resolve_eos_id()
                self.multipliers = shift_mean_weight(
                    self.tokens, self.multipliers, eos_id)
            except Exception:
                # Never abort generation. Continue with untouched weights.
                _warn("mean normalization failed, weights left unchanged:\n"
                      + traceback.format_exc())
            _apply_comfy_to(self)

    class EmphasisTokenNormLengthComfy(sd_emphasis.Emphasis):
        name = "TokenNorm: length / comfy"
        description = ("ComfyUI token normalization (length) with the ComfyUI "
                       "weight interpretation. Divides the weight of "
                       "multi-token textual inversion embeddings, then "
                       "interpolates towards the empty prompt embedding.")

        def after_transformers(self):
            try:
                eos_id = _resolve_eos_id()
                self.multipliers = divide_length(
                    self.tokens, self.multipliers, eos_id)
            except Exception:
                _warn("length normalization failed, weights left unchanged:\n"
                      + traceback.format_exc())
            _apply_comfy_to(self)

    class EmphasisTokenNormLengthMeanComfy(sd_emphasis.Emphasis):
        name = "TokenNorm: length+mean / comfy"
        description = ("ComfyUI token normalization (length + mean) with the "
                       "ComfyUI weight interpretation. Applies length "
                       "division to embeddings first, then shifts the mean to "
                       "1.0, then interpolates towards the empty prompt "
                       "embedding. Per chunk.")

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
            _apply_comfy_to(self)

    _TOKENNORM_OPTIONS = [EmphasisTokenNormMean,
                          EmphasisTokenNormLength,
                          EmphasisTokenNormLengthMean,
                          EmphasisTokenNormNoneComfy,
                          EmphasisTokenNormMeanComfy,
                          EmphasisTokenNormLengthComfy,
                          EmphasisTokenNormLengthMeanComfy]

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


# --------------------------------------------------------------------------
# companion Script
# --------------------------------------------------------------------------

class TokenNormScript(scripts.Script):
    """Adds no UI. Its only job is to expose the processing object so that a
    failed comfy interpretation or a length fallback can be recorded in the
    infotext, and to reset the once-per-generation warning flags.

    Conditioning happens after process() and the infotext is built after
    sampling, so writing into extra_generation_params from the Emphasis object
    lands in the saved image regardless of the exact script callback order.
    """

    def title(self):
        return EXTENSION_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args, **kwargs):
        global _current_processing, _comfy_warned, _length_warned
        _current_processing = p
        _comfy_warned = False
        _length_warned = False
        try:
            p.extra_generation_params.pop(INFOTEXT_FALLBACK_KEY, None)
            p.extra_generation_params.pop(INFOTEXT_LENGTH_FALLBACK_KEY, None)
        except Exception:
            pass

    def postprocess(self, p, processed, *args, **kwargs):
        global _current_processing
        _current_processing = None
