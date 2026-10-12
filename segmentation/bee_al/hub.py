"""Pagine Flask dell'active learning, registrate dall'hub come blueprint."""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional, Union

from flask import Blueprint, Response, abort, redirect, render_template_string, request, send_file, session, url_for

from . import REMOVED_LABEL, UNKNOWN_LABEL
from .audit import parse_scores
from .engine import ALEngine, ALError
from .export import list_exports
from .report import REPORT_DIR, REPORT_FILES, build_report
from .review import read_log
from .retrain import RetrainManager
from .snapshot import snapshot_path


class EngineLoading(Exception):
    """L'engine si sta ancora caricando in background."""


class HubState:
    """Engine condiviso da tutte le richieste, caricato in background alla prima che ne ha bisogno.

    Su dataset grandi il primo avvio richiede circa un minuto (grafo dei vicini): nel
    frattempo le pagine mostrano un avviso invece di restare bloccate.
    """

    def __init__(self) -> None:
        self.engine: Optional[ALEngine] = None
        self.config_path: Union[str, Path, None] = None
        self.lock = threading.Lock()
        self.error: Optional[Exception] = None
        self.started_at = 0.0
        self._thread: Optional[threading.Thread] = None

    def _load(self) -> None:
        try:
            engine = ALEngine.from_config(self.config_path)
            RetrainManager.attach(engine)
            self.engine = engine
        except Exception as e:
            self.error = e

    def get_engine(self) -> ALEngine:
        """Engine pronto, altrimenti ``EngineLoading`` o l'errore dell'ultimo tentativo di caricamento."""
        with self.lock:
            if self.engine is not None:
                return self.engine
            if self._thread is not None and self._thread.is_alive():
                raise EngineLoading()
            if self.error is not None:
                error, self.error, self._thread = self.error, None, None  # la prossima richiesta riprova
                raise error
            self.started_at = time.time()
            self._thread = threading.Thread(target=self._load, daemon=True, name="bee-al-load")
            self._thread.start()
        self._thread.join(0.5)  # se il caricamento è rapido la pagina arriva subito
        with self.lock:
            if self.engine is not None:
                return self.engine
            if self.error is not None:
                error, self.error, self._thread = self.error, None, None
                raise error
        raise EngineLoading()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


STATE = HubState()

_STAGE_BANNER = """{% if al_stage and al_stage.title %}
<div style="background: #1d4ed8; color: white; padding: 8px 14px; border-radius: 8px; margin-bottom: 10px; font-size: 15px;">
    <b>{{ al_stage.title }}</b>{% if al_stage.description %} &mdash; {{ al_stage.description }}{% endif %}
    {% if al_stage.exported %}<span style="background: #16a34a; padding: 2px 8px; border-radius: 10px; margin-left: 8px; font-size: 13px;">clusters saved as {{ al_stage.exported }}</span>{% endif %}
</div>
{% endif %}
"""

_NAV = _STAGE_BANNER + """
<div class="nav">
    <span><a href="/">&larr; Dashboard</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.grid') }}">Grid</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.labeling') }}">Labeling</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.labeling', mode='validation') }}">Validation set</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.labeling', mode='audit') }}">Audit</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.clusters') }}">Clusters</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.review_page') }}">Review{% if al_review_count %} ({{ al_review_count }}){% endif %}</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.status') }}">Status</a></span>
    <span style="font-size: 14px; color: #94a3b8;">User: {{ user }}</span>
</div>
"""

_MESSAGE_PAGE = """
{% if reload %}<meta http-equiv="refresh" content="{{ reload }}">{% endif %}
{{ css|safe }}
<div class="container">
    """ + _NAV + """
    <h2>{{ title }}</h2>
    <p>{{ message }}</p>
</div>
"""

_LABELING_PAGE = """
{{ css|safe }}
<div class="container">
    """ + _NAV + """
    <h2>{{ {'validation': 'Validation Set Labeling', 'audit': 'Audit Labeling'}.get(mode, 'Active Learning Labeling') }}</h2>
    <p style="color: #94a3b8; font-size: 14px;">{{ progress }}</p>
    {% if error %}<p style="color: var(--btn-red);"><b>{{ error }}</b></p>{% endif %}
    {% if redo %}<p style="color: var(--btn-yellow);">Fixing your previous label: <b>{{ previous_label }}</b></p>{% endif %}
    <div style="text-align: center; margin: 20px 0;">
        <img src="{{ url_for('al.image', image_id=image_id) }}" style="max-width: 400px; max-height: 400px; border-radius: 8px;">
    </div>
    {% if suggestion %}
    <p style="color: #94a3b8;">kNN suggestion: <b>{{ suggestion[0] }}</b> ({{ '%.0f' % (suggestion[1] * 100) }}%)</p>
    {% endif %}
    <form method="POST" action="{{ url_for('al.labeling', mode=mode) }}">
        <input type="hidden" name="image_id" value="{{ image_id }}">
        <label><b>Quick Labels:</b></label><br>
        <div style="margin: 10px 0;">
            {% for lbl in quick_labels %}
            <button type="button" class="quick-label-btn" data-label="{{ lbl }}">{{ lbl }}</button>
            {% else %}
            <span style="color: #94a3b8;">No classes yet: type the first one below.</span>
            {% endfor %}
        </div>
        <label><b>Class (pick one above or type a new one):</b></label><br>
        <input type="text" id="al_label_input" name="label" required autocomplete="off" style="width: 100%; padding: 12px; margin: 8px 0 20px; font-size:16px;"><br>
        <label><b>Confidence:</b></label><br>
        <div class="action-group">
            <input type="radio" id="conf-high" name="confidence" value="High" checked><label for="conf-high" class="action-btn btn-high">High</label>
            <input type="radio" id="conf-med" name="confidence" value="Medium"><label for="conf-med" class="action-btn btn-medium">Medium</label>
            <input type="radio" id="conf-low" name="confidence" value="Low"><label for="conf-low" class="action-btn btn-low">Low</label>
        </div>
        <button class="btn" type="submit" name="action" value="label">Submit & Next</button>
        <button class="btn" type="submit" name="action" value="unknown" formnovalidate style="background: var(--input-bg);">Scarta ape</button>
        {% if mode == 'train' and not redo %}
        <button class="btn" type="submit" formaction="{{ url_for('al.reshuffle', next='labeling') }}" formnovalidate style="background: var(--input-bg);" title="Replace the unlabeled bees of this batch with other ones">Shuffle: propose other bees</button>
        {% endif %}
        {% if redo %}
        <button class="btn" type="submit" name="action" value="remove" formnovalidate style="background: var(--btn-red);">Remove label</button>
        {% endif %}
    </form>
    {% if has_previous and not redo %}
    <p style="margin-top: 15px;"><a href="{{ url_for('al.labeling', mode=mode, redo=1) }}">&larr; Fix my previous label</a></p>
    {% endif %}
</div>
<script>
document.querySelectorAll('.quick-label-btn').forEach(function (b) {
    b.addEventListener('click', function () { document.getElementById('al_label_input').value = b.dataset.label; });
});
</script>
"""

GRID_COLORS = (("Azure", "#38bdf8"), ("Red", "#ef4444"), ("Green", "#22c55e"), ("Yellow", "#eab308"),
               ("Violet", "#a78bfa"), ("Pink", "#f472b6"))
GRID_SIZE = 12  # 4 colonne x 3 righe

# Stile, menu, righe dei colori e script comuni alle pagine a griglia (etichettatura e ispezione dei cluster).
_GRID_STYLE = """<style>
    body { margin: 12px; }
    .grid-page { display: flex; gap: 18px; align-items: flex-start; }
    .grid-left { flex: 0 0 auto; width: min(calc(100vw - 440px), calc((100vh - 70px) * 4 / 3)); min-width: 420px; }
    .grid-tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .grid-tile { position: relative; cursor: pointer; border: 5px solid var(--border-color); border-radius: 8px; background: #000; aspect-ratio: 1 / 1; overflow: hidden; }
    .grid-tile img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .grid-tile .badge { position: absolute; left: 0; top: 0; padding: 3px 8px; font-size: 13px; font-weight: bold; color: #0f172a; border-bottom-right-radius: 6px; display: none; }
    .grid-tile .cap { position: absolute; left: 0; right: 0; bottom: 0; padding: 2px 7px; font-size: 12px; color: #e2e8f0; background: rgba(0, 0, 0, 0.6); }
    .grid-right { flex: 1 1 340px; max-width: 420px; background: var(--container-bg); padding: 14px 16px; border-radius: 10px; }
    .group-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; padding: 5px; border-radius: 8px; border: 2px solid transparent; cursor: pointer; }
    .group-row.active { border-color: var(--text-color); background: var(--bg-color); }
    .group-row .swatch { flex: 0 0 34px; height: 34px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #0f172a; }
    .group-row input[type="text"] { flex: 1; min-width: 0; padding: 8px; font-size: 15px; }
    .group-row .count { flex: 0 0 26px; text-align: right; color: #94a3b8; }
</style>"""

_SIDE_NAV = """
        <div style="font-size: 14px; margin-bottom: 8px;">
            <a href="/">&larr; Dashboard</a> &nbsp;|&nbsp; <a href="{{ url_for('al.grid') }}">Grid</a> &nbsp;|&nbsp;
            <a href="{{ url_for('al.grid', mode='validation') }}">Validation grid</a> &nbsp;|&nbsp;
            <a href="{{ url_for('al.clusters') }}">Clusters</a> &nbsp;|&nbsp; <a href="{{ url_for('al.review_page') }}">Review{% if al_review_count %} ({{ al_review_count }}){% endif %}</a> &nbsp;|&nbsp;
            <a href="{{ url_for('al.status') }}">Status</a>
        </div>
"""

_GRID_GROUPS = """
        {% for g in groups %}
        <div class="group-row" data-group="{{ loop.index0 }}" data-color="{{ g.color }}">
            <span class="swatch" style="background: {{ g.color }};">{{ loop.index }}</span>
            <input type="text" name="label_{{ loop.index0 }}" value="{{ g.label }}" list="al_classes" autocomplete="off" placeholder="class for {{ g.name|lower }}">
            <span class="count">0</span>
        </div>
        {% endfor %}
        {% if can_remove %}
        <div class="group-row" data-group="remove" data-color="#f87171" data-name="Remove label">
            <span class="swatch" style="background: #f87171;">{{ groups|length + 1 }}</span>
            <span style="flex: 1; padding: 8px 0;"><b>Remove label</b> <span style="color: #94a3b8; font-size: 13px;">&mdash; back to unlabeled</span></span>
            <span class="count">0</span>
        </div>
        {% endif %}
        <div class="group-row" data-group="discard" data-color="#94a3b8" data-name="Scarta ape">
            <span class="swatch" style="background: #94a3b8;">0</span>
            <span style="flex: 1; padding: 8px 0;"><b>Scarta ape</b> <span style="color: #94a3b8; font-size: 13px;">&mdash; set aside, never shown again</span></span>
            <span class="count">0</span>
        </div>
        <datalist id="al_classes">{% for lbl in quick_labels %}<option value="{{ lbl }}">{% endfor %}</datalist>
"""

_GRID_SCRIPT = """<script>
(function () {
    var rows = Array.prototype.slice.call(document.querySelectorAll('.group-row'));
    var tiles = Array.prototype.slice.call(document.querySelectorAll('.grid-tile'));
    var active = rows[0];
    function pick(row) { active = row; rows.forEach(function (r) { r.classList.toggle('active', r === row); }); }
    function labelOf(row) {
        if (row.dataset.name) return row.dataset.name;
        return row.querySelector('input[type="text"]').value.trim() || ('group ' + (parseInt(row.dataset.group, 10) + 1));
    }
    function paint(tile, row) {
        var input = tile.querySelector('input[name^="group_"]');
        var badge = tile.querySelector('.badge');
        input.value = row ? row.dataset.group : '';
        tile.style.borderColor = row ? row.dataset.color : '';
        badge.style.display = row ? 'block' : 'none';
        if (row) { badge.style.background = row.dataset.color; badge.textContent = labelOf(row); }
    }
    function refresh() {
        var total = 0;
        rows.forEach(function (row) {
            var n = tiles.filter(function (t) { return t.querySelector('input[name^="group_"]').value === row.dataset.group; }).length;
            row.querySelector('.count').textContent = n;
            total += n;
            tiles.forEach(function (t) {
                if (t.querySelector('input[name^="group_"]').value === row.dataset.group) t.querySelector('.badge').textContent = labelOf(row);
            });
        });
        document.getElementById('grid_marked').textContent = total;
    }
    rows.forEach(function (row) {
        row.addEventListener('click', function () { pick(row); });
        var text = row.querySelector('input[type="text"]');
        if (text) { text.addEventListener('focus', function () { pick(row); }); text.addEventListener('input', refresh); }
    });
    tiles.forEach(function (tile) {
        tile.addEventListener('click', function () {
            var current = tile.querySelector('input[name^="group_"]').value;
            paint(tile, current === active.dataset.group ? null : active);
            refresh();
        });
    });
    document.getElementById('grid_clear').addEventListener('click', function () { tiles.forEach(function (t) { paint(t, null); }); refresh(); });
    document.addEventListener('keydown', function (e) {
        if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
        var discard = rows.filter(function (r) { return r.dataset.group === 'discard'; })[0];
        if (e.key === '0' && discard) pick(discard);
        var n = parseInt(e.key, 10);
        if (n >= 1 && n <= rows.length && rows[n - 1] !== discard) pick(rows[n - 1]);
    });
    pick(rows[0]);
    refresh();
})();
</script>"""

_GRID_PAGE = """
{{ css|safe }}
""" + _GRID_STYLE + """
<form method="POST" action="{{ url_for('al.grid', mode=mode) }}" id="grid_form">
<div class="grid-page">
    <div class="grid-left">
        <div class="grid-tiles">
            {% for item in items %}
            <div class="grid-tile" data-index="{{ loop.index0 }}">
                <img src="{{ url_for('al.image', image_id=item.image_id) }}" title="{{ item.image_id }}">
                <span class="badge"></span>
                <input type="hidden" name="image_{{ loop.index0 }}" value="{{ item.image_id }}">
                <input type="hidden" name="group_{{ loop.index0 }}" value="">
            </div>
            {% endfor %}
        </div>
    </div>
    <div class="grid-right">
        """ + _STAGE_BANNER + """
""" + _SIDE_NAV + """        <h2 style="margin: 4px 0;">{{ 'Validation grid' if mode == 'validation' else 'Grid labeling' }}</h2>
        <p style="color: #94a3b8; font-size: 13px; margin: 4px 0 8px;">{{ progress }}</p>
        {% if message %}<p style="color: var(--btn-green); margin: 4px 0;"><b>{{ message }}</b></p>{% endif %}
        {% if error %}<p style="color: var(--btn-red); margin: 4px 0;"><b>{{ error }}</b></p>{% endif %}
        <p style="font-size: 13px; margin: 4px 0;">1. Pick a colour and give it a class. 2. Click the bees of that class. 3. Repeat with another colour, then save.</p>
""" + _GRID_GROUPS + """        <button class="btn" type="submit">Save <span id="grid_marked">0</span> bees &amp; next grid</button>
        <button class="btn" type="button" id="grid_clear" style="background: var(--input-bg);">Clear selection</button>
        {% if mode == 'train' %}
        <button class="btn" type="submit" formaction="{{ url_for('al.reshuffle', next='grid') }}" formnovalidate style="background: var(--input-bg);" title="Replace the unlabeled bees of this batch with other ones; colours not saved yet are lost">Shuffle: propose other bees</button>
        {% endif %}
        <p style="color: #94a3b8; font-size: 12px; margin-top: 10px;">Bees you leave without a colour stay in your grid for the next round. Keys 1&ndash;{{ groups|length }} pick a colour, 0 picks "Scarta ape".</p>
    </div>
</div>
</form>
""" + _GRID_SCRIPT + """
"""

_STATUS_PAGE = """
{% if (s.retrain and s.retrain.state == 'running') or s.export.state == 'running' %}<meta http-equiv="refresh" content="5">{% endif %}
{{ css|safe }}
<div class="container">
    """ + _NAV + """
    <h2>Active Learning Status</h2>
    <table>
        <tr><th>Labels in memory bank</th><td>{{ s.n_labels }} of {{ s.n_images }} images</td></tr>
        <tr><th>Set aside ("Scarta ape")</th><td>{{ s.n_unknown }}</td></tr>
        <tr><th>Unassigned (far from every label)</th><td>{{ s.n_unassigned }}</td></tr>
        <tr><th>Round</th><td>{{ s.round }}</td></tr>
        <tr><th>Regime</th><td>{{ 'Cold start (diversity)' if s.cold_start else 'Uncertainty + diversity + novelty' }}
            {% if s.cold_start %}<span style="color:#94a3b8;">&mdash; until {{ s.cold_start_threshold }} labels and 2 classes</span>{% endif %}</td></tr>
        <tr><th>Current batch</th><td>{{ s.batch_id }}: {{ s.batch_done }} / {{ s.batch_size }} done</td></tr>
        <tr><th>Validation set</th><td>{{ s.validation_labeled }} / {{ s.validation_size }} labeled</td></tr>
        <tr><th>Model version / config</th><td>{{ s.model_version }} / {{ s.config_id }}{% if s.config_file %} ({{ s.config_file }}){% endif %}</td></tr>
        {% if s.subset_file %}<tr><th>Working on a subset</th><td>{{ s.n_images }} bees listed in {{ s.subset_file }}</td></tr>{% endif %}
    </table>
    <form method="POST" action="{{ url_for('al.refresh') }}">
        <button class="btn" type="submit">Recompute propagation &amp; start next batch</button>
    </form>

    {% if s.retrain %}{% set r = s.retrain %}
    <h2>Heavy re-train (contrastive fine-tuning)</h2>
    {% if retrain_error %}<p style="color: var(--btn-red);"><b>{{ retrain_error }}</b></p>{% endif %}
    <table>
        <tr><th>State</th><td>
            {% if r.state == 'running' %}<b style="color: var(--btn-yellow);">running</b>
                {% if r.progress %}&mdash; {{ r.progress.phase }}{% if r.progress.total %} {{ r.progress.step }} / {{ r.progress.total }}{% endif %}{% if r.progress.loss is not none %}, loss {{ '%.3f' % r.progress.loss }}{% endif %}, {{ r.progress.elapsed_s }} s{% endif %}
                <span style="color:#94a3b8;">(started: {{ r.reason }}). You can keep labeling; this page reloads by itself.</span>
            {% elif r.state == 'failed' %}<b style="color: var(--btn-red);">failed</b> &mdash; {{ r.error }}
            {% else %}idle{% endif %}
        </td></tr>
        <tr><th>Automatic</th><td>{% if r.auto %}on ({{ r.policy }}): next after {{ r.next_threshold }} new labels, {{ r.new_labels }} so far{% else %}off &mdash; only when you press the button{% endif %}</td></tr>
        {% if r.last %}
        <tr><th>Last re-train</th><td>{{ r.last.version }} on {{ r.last.created }}, {{ r.last.n_labels }} labels + {{ r.last.n_pseudo }} pseudo-labels, {{ r.last.duration_s }} s.
            {% if r.last.validation_before and r.last.validation_after %}
            Validation {{ r.last.metric }}: {{ '%.3f' % r.last.validation_before[r.last.metric] }} &rarr; {{ '%.3f' % r.last.validation_after[r.last.metric] }}.
            {% else %}<span style="color: var(--btn-yellow);">No validation labels: it could not be checked whether it helped.</span>{% endif %}
            {% if r.last.accepted %}<b style="color: var(--btn-green);">In use.</b>{% else %}<b style="color: var(--btn-red);">Discarded (worse than before); the previous model is still in use.</b>{% endif %}
        </td></tr>
        {% endif %}
    </table>
    <form method="POST" action="{{ url_for('al.retrain') }}">
        <button class="btn" type="submit" {% if r.blocked_by %}disabled style="opacity: 0.5; cursor: not-allowed;"{% endif %}>Start re-train now</button>
        {% if r.blocked_by and r.state != 'running' %}<p style="color: #94a3b8; font-size: 14px;">Not available: {{ r.blocked_by }}.</p>{% endif %}
    </form>
    {% endif %}

    {% set rv = s.review %}
    <h2>Label review</h2>
    {% if not rv.enabled %}<p style="color: #94a3b8;">Switched off in the config (review.enabled).</p>
    {% else %}
    <table>
        <tr><th>Hand labels checked</th><td>{{ rv.n_checked }}</td></tr>
        <tr><th>Suspicious now</th><td>{{ rv.n_suspects }} to review{% if rv.n_rare %} + {{ rv.n_rare }} in rare classes{% endif %}{% if rv.n_confirmed_hidden %} ({{ rv.n_confirmed_hidden }} kept as they are){% endif %}
            {% if rv.n_suspects or rv.n_rare %}&middot; <a href="{{ url_for('al.review_page') }}">review &rarr;</a>{% endif %}</td></tr>
        <tr><th>Decisions so far</th><td>{{ rv.n_reviewed }}: {{ rv.decisions.confirmed }} kept, {{ rv.decisions.corrected }} corrected, {{ rv.decisions.set_aside }} set aside</td></tr>
        {% if rv.corrections %}<tr><th>Corrections</th><td>{% for pair, n in rv.corrections.items() %}{{ pair }}: {{ n }}{{ '; ' if not loop.last }}{% endfor %}</td></tr>{% endif %}
        {% if rv.suspects_by_confidence %}<tr><th>Suspicious by declared confidence</th>
            <td>{% for level, c in rv.suspects_by_confidence.items() %}{{ level }}: {{ c.suspects }} of {{ c.labels }}{% if c.share is not none %} ({{ '%.0f' % (100 * c.share) }}%){% endif %}{{ '; ' if not loop.last }}{% endfor %}</td></tr>{% endif %}
    </table>
    <p style="font-size: 14px;">Download: <a href="{{ url_for('al.review_file', filename='review_log.jsonl') }}">review_log.jsonl</a> &middot;
        <a href="{{ url_for('al.review_file', filename='review_report.csv') }}">review_report.csv</a></p>
    {% endif %}

    <h2>Stopping criteria</h2>
    <table>
        {% for key, name in (('f1_plateau', 'Validation macro-F1 has stopped improving'), ('predictions_stable', 'Predictions are stable'), ('no_new_classes', 'No new classes are appearing')) %}
        {% set c = s.stopping[key] %}
        <tr><th>{{ name }}</th>
            <td style="width: 110px;">{% if c.ok is none %}<b style="color: #94a3b8;">not yet known</b>{% elif c.ok %}<b style="color: var(--btn-green);">met</b>{% else %}<b style="color: var(--btn-yellow);">not met</b>{% endif %}</td>
            <td>{{ c.detail }}</td></tr>
        {% endfor %}
    </table>
    <p style="color: #94a3b8; font-size: 14px;">
        {% if s.stopping.all_ok %}<b style="color: var(--btn-green);">All three criteria are met.</b>{% endif %}
        This is only an indication, evaluated over the last {{ s.stopping.window }} rounds that received new labels: nothing stops by itself.
    </p>

    <h2>Finish this phase: save configuration and clusters</h2>
    {% set x = s.export %}
    {% if not x.clusters_dir %}
    <p style="color: #94a3b8;">Disabled: export.clusters_dir is not set in the config.</p>
    {% else %}
    <p style="color: #94a3b8; font-size: 14px;">When the labeling of this phase is done, this freezes a snapshot and writes the clusters to disk as a dataset:
        one folder per class with the bee images ({{ x.mode }}), plus clusters.csv, config.yaml and meta.json, in
        <code>{{ x.clusters_dir }}/&lt;name&gt;/</code>. Labeling can go on meanwhile; which clusters feed the next phase is decided later.</p>
    {% if export_error %}<p style="color: var(--btn-red);"><b>{{ export_error }}</b></p>{% endif %}
    {% if x.state == 'running' %}
    <p><b style="color: var(--btn-yellow);">Export running</b>: {{ x.progress.copied }} / {{ x.progress.total }} images written for '{{ x.progress.name }}'. This page reloads by itself.</p>
    {% elif x.state == 'failed' %}
    <p><b style="color: var(--btn-red);">Last export failed</b>: {{ x.error }}</p>
    {% endif %}
    <form method="POST" action="{{ url_for('al.export_clusters') }}" style="display: flex; gap: 10px; align-items: center;">
        <input type="text" name="name" required maxlength="80" placeholder="name, e.g. {{ s.stage.name or 'phase1' }}_final" style="flex: 3; padding: 12px; font-size: 16px;">
        <button class="btn" type="submit" style="flex: 2; margin-top: 0; background: var(--btn-green);" {% if x.state == 'running' or not s.n_labels %}disabled{% endif %}>Save configuration &amp; export clusters</button>
    </form>
    {% if x.exports %}
    <table>
        <tr><th>Export</th><th>Created</th><th>Phase</th><th>Clusters (bees: by hand + by the model)</th><th>Folder</th></tr>
        {% for e in x.exports %}
        <tr><td>{{ e.name }}</td><td>{{ e.created }}</td><td>{{ e.stage or '-' }}</td>
            <td>{% for label, c in e.clusters.items() %}{{ label }}: {{ c.total }} ({{ c.human }} + {{ c.predicted }}){{ '; ' if not loop.last }}{% endfor %}</td>
            <td style="font-size: 12px;">{{ e.path }}</td></tr>
        {% endfor %}
    </table>
    {% endif %}
    {% endif %}

    <h2>Snapshots</h2>
    <p style="color: #94a3b8; font-size: 14px;">A snapshot freezes the current predictions of every bee, with its source frame. It never changes afterwards: audits and estimates are computed on it.</p>
    {% if snapshot_error %}<p style="color: var(--btn-red);"><b>{{ snapshot_error }}</b></p>{% endif %}
    {% if snapshot_message %}<p style="color: var(--btn-green);"><b>{{ snapshot_message }}</b></p>{% endif %}
    <form method="POST" action="{{ url_for('al.snapshot') }}" style="display: flex; gap: 10px; align-items: center;">
        <input type="text" name="name" required maxlength="80" placeholder="snapshot name, e.g. 2026-10-05_round42" style="flex: 3; padding: 12px; font-size: 16px;">
        <button class="btn" type="submit" style="flex: 1; margin-top: 0;">Freeze snapshot</button>
    </form>
    {% if s.snapshots %}
    <table>
        <tr><th>Snapshot</th><th>Created</th><th>Round</th><th>Model</th><th>Bees</th><th>Predicted / labeled / unassigned</th><th>Audit</th></tr>
        {% for snap in s.snapshots %}
        <tr><td>{{ snap.name }}</td><td>{{ snap.created }}</td><td>{{ snap.round }}</td><td>{{ snap.model_version }}</td><td>{{ snap.n_bees }}</td>
            <td>{{ snap.status_counts.get('predicted', 0) }} / {{ snap.status_counts.get('labeled', 0) }} / {{ snap.status_counts.get('unassigned', 0) }}</td>
            <td>
            {% if s.audit and s.audit.snapshot == snap.name %}
                <b style="color: var(--btn-green);">active</b>: {{ s.audit.done }} / {{ s.audit.total }} labeled
                &middot; <a href="{{ url_for('al.audit_results', snapshot=snap.name) }}">results</a>
            {% elif snap.has_audit %}
                <a href="{{ url_for('al.audit_results', snapshot=snap.name) }}">results</a>
                <form method="POST" action="{{ url_for('al.audit_activate') }}" style="margin: 0;">
                    <input type="hidden" name="snapshot" value="{{ snap.name }}">
                    <button class="quick-label-btn" type="submit">Make this audit active</button>
                </form>
            {% else %}
                <form method="POST" action="{{ url_for('al.audit_create') }}" enctype="multipart/form-data" style="margin: 0;">
                    <input type="hidden" name="snapshot" value="{{ snap.name }}">
                    <button class="quick-label-btn" type="submit">Create audit set</button><br>
                    <label style="font-size: 12px; color: #94a3b8;">optional anomaly scores (CSV image_id,score): <input type="file" name="scores" accept=".csv"></label>
                </form>
            {% endif %}
            </td></tr>
        {% endfor %}
    </table>
    {% else %}<p>No snapshots yet.</p>{% endif %}
    {% if s.audit %}
    <h2>Audit of snapshot {{ s.audit.snapshot }}</h2>
    <p>{{ s.audit.done }} of {{ s.audit.total }} sampled bees labeled ({{ s.audit.double }} need two different annotators), out of {{ s.audit.population }} predicted bees.
       <a href="{{ url_for('al.labeling', mode='audit') }}">Label the audit &rarr;</a> &middot;
       <a href="{{ url_for('al.audit_results', snapshot=s.audit.snapshot) }}">See the estimates &rarr;</a></p>
    <p style="color: #94a3b8; font-size: 14px;">Audit bees are labeled blind and never enter the model. They are no longer proposed in normal labeling.
        {% if s.audit.annotators %}Answers so far: {% for a, n in s.audit.annotators.items() %}{{ a }} {{ n }}{{ ', ' if not loop.last }}{% endfor %}.{% endif %}</p>
    <table>
        <tr><th>Stratum</th><th>Bees in the snapshot</th><th>Sampled</th><th>Labeled</th></tr>
        {% for st in s.audit.strata %}
        <tr><td>{{ st.id|replace('__unassigned__', 'Unassigned') }}</td><td>{{ st.N_h }}</td><td>{{ st.n_h }}</td><td>{{ st.done }}</td></tr>
        {% endfor %}
    </table>
    {% endif %}

    <h2>Validation metrics</h2>
    {% if s.metrics.accuracy is not none %}
    <table>
        <tr><th></th><th>Accuracy</th><th>Macro-F1</th></tr>
        <tr><td>Final (propagation)</td><td>{{ '%.3f' % s.metrics.accuracy }}</td><td>{{ '%.3f' % s.metrics.macro_f1 }}</td></tr>
        <tr><td>kNN only</td><td>{{ '%.3f' % s.metrics.knn_accuracy }}</td><td>{{ '%.3f' % s.metrics.knn_macro_f1 }}</td></tr>
    </table>
    <p style="color: #94a3b8; font-size: 14px;">Computed on {{ s.metrics.n_validation }} validation labels at the last recompute.</p>
    {% else %}
    <p>No metrics yet: label the <a href="{{ url_for('al.labeling', mode='validation') }}">validation set</a> and some training images, then recompute.</p>
    {% endif %}

    <h2>Class distribution</h2>
    {% if s.class_counts %}
    <table>
        <tr><th>Class</th><th>Labels</th><th></th></tr>
        {% for name, count in s.class_counts.items() %}
        <tr><td>{{ name }}</td><td>{{ count }}</td><td>{{ 'reject class' if name in s.reject_classes else '' }}</td></tr>
        {% endfor %}
    </table>
    {% else %}<p>No labels yet.</p>{% endif %}
    {% if s.reject_counts %}
    <h3>Reject classes</h3>
    <table>
        <tr><th>Reject class</th><th>Labeled by hand</th><th>Assigned by the model</th></tr>
        {% for name, c in s.reject_counts.items() %}
        <tr><td>{{ name }}</td><td>{{ c.labeled }}</td><td>{{ c.predicted }}</td></tr>
        {% endfor %}
    </table>
    <p style="color: #94a3b8; font-size: 14px;">Reject bees stay in the model (so that similar ones are recognised) but are left out of the final clusters and of the prevalence denominators.
       Rule for annotators: if the defect makes it impossible to judge the bee it is a reject; if you can still tell what the bee is, label what it is.</p>
    {% endif %}

    <h2>Rounds</h2>
    <table>
        <tr><th>Round</th><th>Labels</th><th>Classes</th><th>Val. labels</th><th>Accuracy</th><th>Macro-F1</th>
            <th>&Delta;pred %</th><th>New classes</th><th>Novelty &rarr; new class</th><th>Model</th></tr>
        {% for h in s.history[-15:]|reverse %}
        <tr><td>{{ h.round }}</td><td>{{ h.n_labels }}</td><td>{{ h.n_classes }}</td><td>{{ h.n_validation }}</td>
            <td>{{ '%.3f' % h.accuracy if h.accuracy is not none else '-' }}</td>
            <td>{{ '%.3f' % h.macro_f1 if h.macro_f1 is not none else '-' }}</td>
            <td>{{ '%.2f' % (100 * h.pred_change) if h.get('pred_change') is not none else '-' }}</td>
            <td>{{ h.get('new_classes', [])|join(', ') or '-' }}</td>
            <td>{% if h.get('novelty_queries') %}{{ h.novelty_queries }} &rarr; {{ h.novelty_new_class }}{% else %}-{% endif %}</td>
            <td>{{ h.model_version }}{% if h.get('model_changed') %} <b style="color: var(--btn-yellow);" title="The extractor changed: this round is not comparable with the previous one">new model</b>{% endif %}</td></tr>
        {% endfor %}
    </table>
    <p style="color: #94a3b8; font-size: 14px;">&Delta;pred % is the share of bees whose final class changed since the previous round.</p>
</div>
"""


_RESULTS_PAGE = """
{{ css|safe }}
<div class="container" style="max-width: 1100px;">
    """ + _NAV + """
    {% set e = report.estimates %}{% set ind = report.indicators %}
    {% macro pct(v) %}{{ '-' if v is none else '%.1f%%' % (100 * v) }}{% endmacro %}
    <h2>Audit results &mdash; snapshot {{ report.snapshot }}</h2>
    <p>{{ e.audit.answered }} of {{ e.audit.sampled }} sampled bees answered ({{ e.totals.audit_unknown }} "Scarta ape"),
       out of {{ e.totals.population }} predicted bees; {{ e.totals.human }} more were labeled by hand and count exactly.
       Computed {{ report.created }} with {{ e.n_boot }} bootstrap replicates.</p>
    {% if not e.complete %}<p style="color: var(--btn-yellow);"><b>The audit is not complete: these figures will still move.</b></p>{% endif %}
    {% if e.strata_without_audit %}<p style="color: var(--btn-yellow);">Strata without any audit answer (the predicted class is used as is): {{ e.strata_without_audit|join(', ') }}.</p>{% endif %}
    {% if not e.frame_resampling %}<p style="color: #94a3b8; font-size: 14px;">Frames could not be resampled (bee-to-frame map missing or unreliable): intervals reflect the audit sample only.</p>{% endif %}
    <p style="font-size: 14px;">Download:
        {% for name in files %}<a href="{{ url_for('al.audit_file', snapshot=report.snapshot, filename=name) }}">{{ name }}</a>{{ ' · ' if not loop.last }}{% endfor %}</p>

    <h2>Precision per stratum</h2>
    <table>
        <tr><th>Stratum</th><th>Bees</th><th>Sampled</th><th>Answered</th><th>"Scarta ape"</th><th>Precision</th><th>95% CI</th></tr>
        {% for s in e.strata %}
        <tr><td>{{ s.id|replace('__unassigned__', 'Unassigned') }}{% if s.kind == 'reject' %} <span style="color: var(--btn-red); font-size: 12px;">reject</span>{% endif %}</td>
            <td>{{ s.N_h }}</td><td>{{ s.n_sampled }}</td><td>{{ s.n_labeled }}</td><td>{{ s.unknown }}</td>
            <td>{{ pct(s.precision) }}</td><td>{% if s.ci_low is not none %}{{ pct(s.ci_low) }} &ndash; {{ pct(s.ci_high) }}{% else %}-{% endif %}</td></tr>
        {% endfor %}
    </table>
    <p style="color: #94a3b8; font-size: 14px;">Precision = share of audited bees that really are the class of the stratum (for reject strata: that really are a reject).</p>

    <h2>Prevalence</h2>
    <table>
        <tr><th>Class</th><th>Classify and count</th><th>Stratified estimate</th><th>95% CI (bootstrap)</th><th>95% CI (analytic)</th><th>Labeled by hand</th><th>Seen in audit</th></tr>
        {% for p in e.prevalence if not p.reject %}
        <tr><td>{{ p.class }}</td><td>{{ pct(p.cc_share) }}</td><td><b>{{ pct(p.share) }}</b></td>
            <td>{{ pct(p.ci_boot_low) }} &ndash; {{ pct(p.ci_boot_high) }}</td><td>{{ pct(p.ci_analytic_low) }} &ndash; {{ pct(p.ci_analytic_high) }}</td>
            <td>{{ p.human }}</td><td>{{ p.audit_observed }}{% if p.upper_limit is not none %} <span style="color: #94a3b8;">(at most {{ pct(p.upper_limit) }})</span>{% endif %}</td></tr>
        {% endfor %}
    </table>
    <p style="color: #94a3b8; font-size: 14px;">Shares of valid bees. Estimated reject share: {{ pct(e.totals.reject_share) }} ({{ pct(e.totals.reject_share_ci[0]) }} &ndash; {{ pct(e.totals.reject_share_ci[1]) }}){% for p in e.prevalence if p.reject %}{{ ': ' if loop.first else ', ' }}{{ p.class }} &asymp; {{ '%.0f' % p.n_hat }} bees{% endfor %}.</p>

    <h2>Agreement between annotators</h2>
    {% if e.agreement.n_pairs %}
    <p>{{ e.agreement.n_pairs }} bees labeled twice: raw agreement {{ pct(e.agreement.agreement) }}, Cohen's kappa {{ '%.2f' % e.agreement.kappa if e.agreement.kappa is not none else 'not defined' }}.</p>
    {% else %}<p>No bee has been labeled by two annotators yet.</p>{% endif %}

    <h2>Colony indicators by {{ ind.group_by }}</h2>
    {% for note in ind.notes %}<p style="color: var(--btn-yellow);">{{ note }}</p>{% endfor %}
    {% if ind.groups %}
    <table>
        <tr><th>Group</th><th>Light</th><th>Rules fired</th><th>JSD (I3)</th><th>Unexplained (I4)</th><th>Estimated reject share</th><th>Discarded by cleaning</th><th>Raw bees / frame</th></tr>
        {% for g in ind.groups %}
        <tr><td>{{ g.group }}</td>
            <td><b style="color: var(--btn-{{ {'green': 'green', 'yellow': 'yellow', 'red': 'red'}[g.light.color] }});">{{ g.light.color }}</b></td>
            <td>{% for r in g.light.rules %}{{ r.rule }}: {{ r.detail }}{{ '; ' if not loop.last }}{% else %}-{% endfor %}</td>
            <td>{{ '%.3f' % g.I3_jsd.value }} ({{ '%.3f' % g.I3_jsd.ci_low }} &ndash; {{ '%.3f' % g.I3_jsd.ci_high }})</td>
            <td>{{ pct(g.I4_unexplained.share) }}</td><td>{{ pct(g.quality.reject_share) }}</td><td>{{ pct(g.quality.cleaning_discard_rate) }}</td>
            <td>{{ '%.0f' % g.strength.raw_bees_per_frame if g.strength.raw_bees_per_frame is not none else '-' }}</td></tr>
        {% endfor %}
    </table>
    <p style="color: #94a3b8; font-size: 14px;">JSD threshold {{ '%.3f' % ind.reference.jsd_threshold }}, reference profile: {{ 'given explicitly' if ind.reference.explicit else 'mean of ' ~ ind.reference.groups|join(', ') }}.
       Group estimates assume that the model's errors inside a stratum are the same in every group.</p>
    {% for title, key in (('Demographic profile (I1)', 'I1_demographics'), ('Disease signs (I2)', 'I2_disease')) %}
    {% if ind.groups[0][key] %}
    <h3>{{ title }}</h3>
    <table>
        <tr><th>Group</th><th>Class</th><th>Share</th><th>95% CI</th><th>Status</th></tr>
        {% for g in ind.groups %}{% for name, x in g[key].items() %}
        <tr><td>{{ g.group }}</td><td>{{ name }}</td><td>{{ pct(x.share) }}</td>
            <td>{% if x.share is not none %}{{ pct(x.ci_low) }} &ndash; {{ pct(x.ci_high) }}{% else %}-{% endif %}</td>
            <td>{{ x.get('status', x.get('note', '')) }}{% if x.get('upper_limit') is not none %} (at most {{ pct(x.upper_limit) }}){% endif %}</td></tr>
        {% endfor %}{% endfor %}
    </table>
    {% endif %}
    {% endfor %}
    {% if ind.trend %}
    <h3>Trend between sessions (I5)</h3>
    <table>
        <tr><th>Colony</th><th>From &rarr; to</th><th>Quantity</th><th>Change</th><th>95% CI</th><th></th></tr>
        {% for t in ind.trend %}
        {% if t.get('missing') %}<tr><td>{{ t.colony }}</td><td>{{ t.from }} &rarr; {{ t.to }}</td><td colspan="4">no data for {{ t.missing|join(', ') }}</td></tr>{% endif %}
        {% for name, d in t.deltas.items() %}
        <tr><td>{{ t.colony }}</td><td>{{ t.from }} &rarr; {{ t.to }}</td><td>{{ name }}</td><td>{{ '%+.3f' % d.delta }}</td>
            <td>{{ '%+.3f' % d.ci_low }} &ndash; {{ '%+.3f' % d.ci_high }}</td><td>{{ 'significant' if d.significant else '' }}</td></tr>
        {% endfor %}{% endfor %}
    </table>
    {% endif %}
    {% else %}<p>No groups to report.</p>{% endif %}
</div>
"""


_REVIEW_PAGE = """
{{ css|safe }}
<div class="container" style="max-width: 1000px;">
    """ + _NAV + """
    <h2>Label review</h2>
    <p style="color: #94a3b8; font-size: 14px;">{{ q.main|length }} suspects left, {{ stats.n_reviewed }} reviewed.
        A label is suspicious when the most similar labeled bees say, with at least {{ '%.0f' % (100 * threshold) }}% of the vote, that it is another class.
        Nothing is changed automatically: you decide. Validation and audit labels are never shown here.</p>
    {% if message %}<p style="color: var(--btn-green);"><b>{{ message }}</b></p>{% endif %}
    {% if error %}<p style="color: var(--btn-red);"><b>{{ error }}</b></p>{% endif %}

    {% if s %}
    <div style="display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap;">
        <div style="flex: 0 0 360px; text-align: center;">
            <img src="{{ url_for('al.image', image_id=s.image_id) }}" style="max-width: 360px; max-height: 360px; border-radius: 8px;">
        </div>
        <div style="flex: 1 1 380px;">
            <table>
                <tr><th>Current label</th><td><b>{{ s.label }}</b> <span style="color: #94a3b8;">({{ '%.0f' % (100 * s.p_label) }}% of the neighbours' vote)</span></td></tr>
                <tr><th>Neighbours suggest</th><td><b style="color: var(--btn-yellow);">{{ s.suggested }}</b> <span style="color: #94a3b8;">({{ '%.0f' % (100 * s.p_suggested) }}%)</span></td></tr>
                <tr><th>Labeled by</th><td>{{ s.annotator or '-' }}{% if s.timestamp %} on {{ s.timestamp }}{% endif %}</td></tr>
                <tr><th>Declared confidence</th><td>{{ s.confidence or 'not given' }}{% if s.query_type %} <span style="color: #94a3b8;">({{ s.query_type }} query)</span>{% endif %}</td></tr>
            </table>
            {% if s.rare_class %}
            <p style="color: var(--btn-yellow); font-size: 14px;">"{{ s.label }}" has fewer than {{ min_class_size }} labels: its neighbours are almost always from other classes even when the label is right. Weak signal.</p>
            {% endif %}
            <form method="POST" action="{{ url_for('al.review_page') }}">
                <input type="hidden" name="image_id" value="{{ s.image_id }}">
                <button class="btn" type="submit" name="decision" value="confirmed" style="background: var(--btn-green);">Keep "{{ s.label }}"</button>
                <button class="btn" type="submit" name="decision" value="suggested" style="background: var(--btn-yellow); color: #0f172a;">Change to "{{ s.suggested }}"</button>
                <div style="margin: 12px 0 4px;">
                    {% for lbl in quick_labels %}<button type="button" class="quick-label-btn" data-label="{{ lbl }}">{{ lbl }}</button>{% endfor %}
                </div>
                <div style="display: flex; gap: 10px;">
                    <input type="text" id="al_label_input" name="new_label" autocomplete="off" placeholder="another class" style="flex: 3; padding: 12px; font-size: 16px;">
                    <button class="btn" type="submit" name="decision" value="corrected" style="flex: 2; margin-top: 0;">Change to this class</button>
                </div>
                <button class="btn" type="submit" name="decision" value="set_aside" style="background: var(--input-bg);">Scarta ape</button>
            </form>
        </div>
    </div>
    <h3>Most similar labeled bees</h3>
    <div class="inspect-images">
        {% for n in s.neighbors %}
        <div style="text-align: center; font-size: 12px; color: #94a3b8; width: 130px;">
            <img src="{{ url_for('al.image', image_id=n.image_id) }}" title="{{ n.image_id }}"
                 style="border: 3px solid {{ 'var(--btn-green)' if n.label == s.label else ('var(--btn-yellow)' if n.label == s.suggested else 'var(--border-color)') }};"><br>
            <b style="color: var(--text-color);">{{ n.label }}</b><br>sim {{ '%.2f' % n.similarity }}
        </div>
        {% endfor %}
    </div>
    <p style="color: #94a3b8; font-size: 13px;">Green border: same class as the current label. Yellow border: the suggested class.</p>
    {% else %}
    <p>No suspicious labels to review{% if not enabled %}: the review queue is switched off in the config{% elif not q.n_checked %} yet: there are no hand labels{% else %} among the {{ q.n_checked }} hand labels checked{% endif %}.
       {% if q.n_confirmed_hidden %}{{ q.n_confirmed_hidden }} suspicious label(s) were kept as they are and are hidden.{% endif %}</p>
    {% endif %}

    {% if q.rare %}
    <h2>Rare classes</h2>
    <p style="color: #94a3b8; font-size: 14px;">Labels of classes with fewer than {{ min_class_size }} examples. Their neighbours are nearly always from other classes,
        even when the label is right &mdash; a newly found rare phenotype looks exactly like this. Check them, but expect most to be correct.</p>
    <div class="inspect-images">
        {% for r in q.rare %}
        <div style="text-align: center; font-size: 12px; color: #94a3b8; width: 130px;">
            <a href="{{ url_for('al.review_page', image=r.image_id) }}"><img src="{{ url_for('al.image', image_id=r.image_id) }}" title="{{ r.image_id }}"></a><br>
            <b style="color: var(--text-color);">{{ r.label }}</b><br>&rarr; {{ r.suggested }} {{ '%.0f' % (100 * r.p_suggested) }}%
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>
<script>
document.querySelectorAll('.quick-label-btn[data-label]').forEach(function (b) {
    b.addEventListener('click', function () { document.getElementById('al_label_input').value = b.dataset.label; });
});
</script>
"""


_CLUSTERS_PAGE = """
{{ css|safe }}
""" + _GRID_STYLE + """
<form method="POST" action="{{ url_for('al.clusters', group=group, page=view.page if view else 0) }}" id="grid_form">
<div class="grid-page">
    <div class="grid-left">
        <div class="grid-tiles">
            {% for b in (view.samples if view else []) %}
            <div class="grid-tile" data-index="{{ loop.index0 }}">
                <img src="{{ url_for('al.image', image_id=b.image_id) }}" title="{{ b.image_id }}{{ b.tip }}">
                <span class="badge"></span>
                <span class="cap">{{ b.band }} &middot; sim {{ '%.2f' % b.similarity }}</span>
                <input type="hidden" name="image_{{ loop.index0 }}" value="{{ b.image_id }}">
                <input type="hidden" name="group_{{ loop.index0 }}" value="">
            </div>
            {% endfor %}
        </div>
        {% if not view %}<p>No clusters yet: label a few bees first.</p>{% endif %}
    </div>
    <div class="grid-right">
        """ + _STAGE_BANNER + _SIDE_NAV + """
        <h2 style="margin: 4px 0;">Clusters</h2>
        <p style="color: #94a3b8; font-size: 13px; margin: 4px 0 8px;">Round {{ s.round }}, {{ s.n_labels }} labels.</p>
        {% if message %}<p style="color: var(--btn-green); margin: 4px 0;"><b>{{ message }}</b></p>{% endif %}
        {% if error %}<p style="color: var(--btn-red); margin: 4px 0;"><b>{{ error }}</b></p>{% endif %}
        {% if view %}
        <select id="cluster_select" style="width: 100%; padding: 10px; font-size: 15px; background: var(--input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 6px;">
            {% for title, options in menu %}
            <optgroup label="{{ title }}">
                {% for o in options %}<option value="{{ url_for('al.clusters', group=o.key) }}" {% if o.key == group %}selected{% endif %}>{{ o.name }} ({{ o.n }})</option>{% endfor %}
            </optgroup>
            {% endfor %}
        </select>
        <p style="font-size: 13px; margin: 8px 0 4px;">{{ view.summary }}{% if view.reject %} <span style="color: var(--btn-red);">Reject class.</span>{% endif %}</p>
        <p style="font-size: 13px; margin: 4px 0;">
            Row 1: the bees most similar to {{ view.reference }}. Row 2: around the median. Row 3: the least similar.
        </p>
        <p style="font-size: 14px; margin: 6px 0;">
            {% if view.n_pages > 1 %}
            <a href="{{ url_for('al.clusters', group=group, page=view.page - 1) }}">&larr; previous</a> &nbsp;
            set {{ view.page + 1 }} of {{ view.n_pages }} &nbsp;
            <a href="{{ url_for('al.clusters', group=group, page=view.page + 1) }}">other bees &rarr;</a>
            {% else %}<span style="color: #94a3b8;">All {{ view.n_members }} bees of this group are shown.</span>{% endif %}
        </p>
        <p style="font-size: 13px; margin: 4px 0;">To change bees: pick a colour, give it a class, click the bees, then save.</p>
        """ + _GRID_GROUPS + """
        <button class="btn" type="submit">Save <span id="grid_marked">0</span> changes</button>
        <button class="btn" type="button" id="grid_clear" style="background: var(--input-bg);">Clear selection</button>
        <p style="color: #94a3b8; font-size: 12px; margin-top: 10px;">Keys 1&ndash;{{ groups|length }} pick a colour, 0 picks "Scarta ape". Saving recomputes the predictions.</p>
        {% endif %}
    </div>
</div>
</form>
{% if view %}
""" + _GRID_SCRIPT + """
<script>
document.getElementById('cluster_select').addEventListener('change', function () { window.location = this.value; });
</script>
{% endif %}
"""


def create_blueprint(base_css: str, config_path: Union[str, Path, None] = None) -> Blueprint:
    """Blueprint con ``/al-grid``, ``/al-labeling``, ``/al-clusters``, ``/al-status``, ``/al-refresh`` e ``/al-retrain``.

    ``base_css`` è il CSS dell'hub, così le pagine nuove hanno lo stesso aspetto.
    """
    bp = Blueprint("al", __name__)

    @bp.app_context_processor
    def stage_banner():
        """Fase corrente, mostrata in cima a ogni pagina di active learning."""
        engine = STATE.engine
        if engine is None:
            return {"al_stage": None, "al_review_count": 0}
        stage = engine.cfg.stage
        done = [e["name"] for e in list_exports(engine.cfg) if e.get("stage") == stage.name and e.get("config_id")]
        return {"al_stage": {"name": stage.name, "title": stage.title, "description": stage.description,
                             "exported": done[0] if done else ""},
                "al_review_count": len(engine.suspects()["main"])}
    # La config si può scegliere all'avvio con BEE_AL_CONFIG=config/al_fase2.yaml (una fase per file).
    STATE.config_path = config_path or os.environ.get("BEE_AL_CONFIG") or None

    def message(title: str, text: str, code: int = 200, reload: int = 0):
        html = render_template_string(
            _MESSAGE_PAGE, css=base_css, user=session.get("user_email"), title=title, message=text, reload=reload
        )
        return html, code

    def engine_or_message():
        try:
            return STATE.get_engine(), None
        except EngineLoading:
            waited = int(time.time() - STATE.started_at)
            text = ("Loading embeddings and building the neighbour graph. The first start on a new dataset takes "
                    f"about a minute, later ones a few seconds. This page reloads by itself ({waited} s so far).")
            return None, message("Active learning is starting", text, 503, reload=3)
        except (FileNotFoundError, ValueError) as e:
            return None, message("Active learning not ready", str(e), 503)

    @bp.route("/al-image/<image_id>")
    def image(image_id: str):
        """Immagine di un'ape, letta dal percorso salvato con gli embedding (più cartelle possibili)."""
        if not session.get("user_email"):
            abort(403)
        engine, _ = engine_or_message()
        if engine is None or image_id not in engine.bank:
            abort(404)
        path = engine.bank.paths[engine.bank.index_of(image_id)]
        if not os.path.isfile(path):
            abort(404)
        return send_file(path, max_age=3600)

    @bp.route("/al-reshuffle", methods=["POST"])
    def reshuffle():
        """Pulsante "Shuffle": le api del batch non ancora etichettate vengono sostituite con altre."""
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        target = "al.grid" if request.args.get("next") == "grid" else "al.labeling"
        try:
            n = engine.reshuffle_batch(user)
        except ALError as e:
            return redirect(url_for(target, shuffle_error=f"Not shuffled: {e}."))
        return redirect(url_for(target, shuffled=n))

    @bp.route("/al-labeling", methods=["GET", "POST"])
    def labeling():
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        mode = request.args.get("mode")
        mode = mode if mode in ("validation", "audit") else "train"
        last_key = f"al_last_{mode}"
        error = None
        if mode == "audit" and engine.audit_progress() is None:
            return message("No audit yet", "Freeze a snapshot and create its audit set from the Status page first.")

        if request.method == "POST":
            image_id = request.form.get("image_id", "")
            action = request.form.get("action", "label")
            label = {"unknown": UNKNOWN_LABEL, "remove": REMOVED_LABEL}.get(action, request.form.get("label", ""))
            try:
                engine.submit(user, image_id, label, request.form.get("confidence", ""), mode=mode)
            except ALError as e:
                error = str(e)
            else:
                if action == "remove":
                    session.pop(last_key, None)
                else:
                    session[last_key] = image_id
                return redirect(url_for("al.labeling", mode=mode))

        previous = session.get(last_key)
        redo = bool(request.args.get("redo")) and previous is not None and previous in engine.bank
        previous_label = ""
        if redo:
            image_id = previous
            if mode == "audit":
                previous_label = engine.audit_label_of(image_id, user)
            else:
                previous_label = engine.val_labels.get(image_id) if mode == "validation" else engine.bank.label_of(image_id)
            previous_label = "Scarta ape" if previous_label == UNKNOWN_LABEL else (previous_label or "")
        else:
            item = engine.next_image(user, mode=mode)
            if item is None:
                if mode == "validation":
                    return message("Validation set complete", "Every validation image has been labeled (or is assigned to another annotator).")
                if mode == "audit":
                    return message("Nothing left for you in this audit", "Every audit bee is labeled, assigned to another annotator, or waiting for a second annotator other than you.")
                return message("Nothing to label right now", "The rest of the batch is assigned to other annotators, or no unlabeled images are left. Try again in a moment.")
            image_id = item.image_id

        st = engine.status()
        if mode == "validation":
            progress = f"Validation set: {st['validation_labeled']} / {st['validation_size']} labeled. These labels are used only for evaluation."
        elif mode == "audit":
            progress = (f"Audit: {st['audit']['done']} / {st['audit']['total']} labeled. Label what you see: these bees were never "
                        "labeled before, and your answers are used only to check the clusters.")
        else:
            progress = f"Batch {st['batch_id']}: {st['batch_done']} / {st['batch_size']} done. {st['n_labels']} labels in total."
        return render_template_string(
            _LABELING_PAGE, css=base_css, user=user, mode=mode, image_id=image_id, progress=progress,
            quick_labels=engine.quick_labels(), error=error or request.args.get("shuffle_error"),
            suggestion=None if mode == "audit" else engine.suggestion(image_id),  # l'audit è alla cieca
            redo=redo, previous_label=previous_label, has_previous=previous is not None,
        )

    @bp.route("/al-grid", methods=["GET", "POST"])
    def grid():
        """Etichettatura a griglia: 12 api per volta, più classi in un colpo solo.

        Ogni colore è un gruppo con la sua classe: si sceglie un colore, si cliccano le api,
        si passa al colore successivo e si salva tutto insieme. Le classi dei colori restano
        nella sessione dell'annotatore tra una griglia e l'altra.
        """
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        mode = "validation" if request.args.get("mode") == "validation" else "train"
        # Le classi dei colori si ricordano per fase: quelle della fase 1 non devono ricomparire nella fase 2.
        labels_key = f"al_grid_labels_{engine.cfg.stage.name}"
        labels = list(session.get(labels_key) or [])
        if not labels:  # prima volta: le classi suggerite in config, poi quelle già viste
            suggested = [engine.store.normalize_label(c) for c in engine.cfg.labels.classes]
            labels = list(dict.fromkeys(suggested + engine.quick_labels()))
        labels = (labels + [""] * len(GRID_COLORS))[:len(GRID_COLORS)]
        error = None

        if request.method == "POST":
            labels = [request.form.get(f"label_{g}", "").strip() for g in range(len(GRID_COLORS))]
            session[labels_key] = labels
            chosen: list[tuple[str, str]] = []
            for k in range(GRID_SIZE):
                image_id, group = request.form.get(f"image_{k}"), request.form.get(f"group_{k}", "")
                if not image_id or not group:
                    continue
                if group == "discard":
                    chosen.append((image_id, UNKNOWN_LABEL))
                elif group.isdigit() and int(group) < len(labels) and labels[int(group)]:
                    chosen.append((image_id, labels[int(group)]))
                else:
                    error = "A colour is used on some bees but has no class: type its class and save again."
            if error is None and not chosen:
                error = "No bee has a colour yet: pick a colour and click the bees."
            if error is None:
                try:  # tutto o niente: si controlla prima di scrivere
                    for image_id, label in chosen:
                        if image_id not in engine.bank or not engine.store.normalize_label(label):
                            raise ALError(f"cannot label {image_id}")
                    for image_id, label in chosen:
                        engine.submit(user, image_id, label, "", mode=mode)
                except ALError as e:
                    error = str(e)
                else:
                    return redirect(url_for("al.grid", mode=mode, saved=len(chosen)))

        items = engine.next_images(user, GRID_SIZE, mode=mode)
        if not items:
            if mode == "validation":
                return message("Validation set complete", "Every validation image has been labeled (or is assigned to another annotator).")
            return message("Nothing to label right now", "The rest of the batch is assigned to other annotators, or no unlabeled images are left. Try again in a moment.")
        st = engine.status()
        if mode == "validation":
            progress = f"Validation set: {st['validation_labeled']} / {st['validation_size']} labeled. Used only for evaluation."
        else:
            progress = f"Batch {st['batch_id']}: {st['batch_done']} / {st['batch_size']} done. {st['n_labels']} labels in total."
        saved, shuffled = request.args.get("saved", type=int), request.args.get("shuffled", type=int)
        note = None if saved is None else f"{saved} bees saved."
        if shuffled is not None:
            note = f"{shuffled} bees of the batch replaced with other ones."
        return render_template_string(
            _GRID_PAGE, css=base_css, mode=mode, items=items, progress=progress,
            error=error or request.args.get("shuffle_error"), message=note,
            groups=[{"name": name, "color": color, "label": labels[g]} for g, (name, color) in enumerate(GRID_COLORS)],
            quick_labels=engine.quick_labels(),
        )

    @bp.route("/al-status")
    def status():
        if not session.get("user_email"):
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        return render_template_string(
            _STATUS_PAGE, css=base_css, user=session.get("user_email"), s=engine.status(),
            retrain_error=request.args.get("retrain_error"),
            snapshot_error=request.args.get("snapshot_error"), snapshot_message=request.args.get("snapshot_message"),
            export_error=request.args.get("export_error"),
        )

    @bp.route("/al-export", methods=["POST"])
    def export_clusters():
        """Chiude la fase: congela uno snapshot e scrive su disco configurazione e cluster (in background)."""
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        try:
            engine.exporter.start(request.form.get("name", "").strip(), user)
        except ValueError as e:
            return redirect(url_for("al.status", export_error=f"Export not started: {e}"))
        return redirect(url_for("al.status"))

    @bp.route("/al-snapshot", methods=["POST"])
    def snapshot():
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        name = request.form.get("name", "").strip()
        try:
            engine.export_snapshot(name, user)
        except ALError as e:
            return redirect(url_for("al.status", snapshot_error=f"Snapshot not created: {e}"))
        return redirect(url_for("al.status", snapshot_message=f"Snapshot '{name}' created."))

    @bp.route("/al-audit-create", methods=["POST"])
    def audit_create():
        if not session.get("user_email"):
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        name = request.form.get("snapshot", "")
        try:
            upload = request.files.get("scores")
            scores = None
            if upload is not None and upload.filename:
                scores = parse_scores(upload.read().decode("utf-8-sig"))
            progress = engine.create_audit(name, scores)
        except (ALError, ValueError, UnicodeDecodeError) as e:
            return redirect(url_for("al.status", snapshot_error=f"Audit set not created: {e}"))
        text = f"Audit set of '{name}' created: {progress['total']} bees in {len(progress['strata'])} strata. It is now the active audit."
        return redirect(url_for("al.status", snapshot_message=text))

    @bp.route("/al-audit-activate", methods=["POST"])
    def audit_activate():
        if not session.get("user_email"):
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        name = request.form.get("snapshot", "")
        try:
            engine.load_audit(name)
        except ALError as e:
            return redirect(url_for("al.status", snapshot_error=f"Audit not activated: {e}"))
        return redirect(url_for("al.status", snapshot_message=f"The audit of '{name}' is now active."))

    def _audit_folder(engine: ALEngine, name: str) -> Optional[Path]:
        try:
            folder = snapshot_path(engine.cfg, name)
        except ValueError:
            return None
        return folder if (folder / "audit_set.json").exists() else None

    @bp.route("/al-audit-results")
    def audit_results():
        """Stime corrette e indicatori di colonia di uno snapshot con audit (in cache finché non arrivano risposte nuove)."""
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        progress = engine.audit_progress()
        name = request.args.get("snapshot") or (progress["snapshot"] if progress else "")
        folder = _audit_folder(engine, name)
        if folder is None:
            return message("No audit results", "This snapshot does not exist or has no audit set yet. Create one from the Status page.", 404)
        report = build_report(folder, engine.cfg, store=engine.store)
        return render_template_string(_RESULTS_PAGE, css=base_css, user=user, report=report,
                                      files=[f for f in REPORT_FILES if (folder / REPORT_DIR / f).exists()])

    @bp.route("/al-audit-file/<snapshot>/<filename>")
    def audit_file(snapshot: str, filename: str):
        if not session.get("user_email"):
            abort(403)
        engine, _ = engine_or_message()
        folder = _audit_folder(engine, snapshot) if engine is not None else None
        if folder is None or filename not in REPORT_FILES or not (folder / REPORT_DIR / filename).exists():
            abort(404)
        return send_file(folder / REPORT_DIR / filename, as_attachment=True, download_name=f"{snapshot}_{filename}")

    @bp.route("/al-retrain", methods=["POST"])
    def retrain():
        if not session.get("user_email"):
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        try:
            manager = engine.retrain or RetrainManager.attach(engine)
            manager.start(f"button, {session.get('user_email')}")
        except ValueError as e:
            return redirect(url_for("al.status", retrain_error=f"Re-train not started: {e}."))
        return redirect(url_for("al.status"))

    @bp.route("/al-review", methods=["GET", "POST"])
    def review_page():
        """Revisione delle etichette sospette, una alla volta: si conferma, si corregge o si scarta."""
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        error = None
        if request.method == "POST":
            image_id = request.form.get("image_id", "")
            decision = request.form.get("decision", "")
            new_label = request.form.get("new_label", "")
            try:
                if decision == "suggested":   # "Change to <classe suggerita>": la classe la sa il server
                    current = next((x for x in engine.suspects()["main"] + engine.suspects()["rare"] if x["image_id"] == image_id), None)
                    if current is None:
                        raise ALError(f"{image_id} is no longer in the review queue")
                    decision, new_label = "corrected", current["suggested"]
                engine.review(user, image_id, decision, new_label)
            except ALError as e:
                error = str(e)
            else:
                return redirect(url_for("al.review_page", done=decision))

        queue = engine.suspects()
        wanted = request.args.get("image")
        everything = queue["main"] + queue["rare"]
        suspect = next((x for x in everything if x["image_id"] == wanted), None) if wanted else None
        if suspect is None:
            suspect = queue["main"][0] if queue["main"] else None
        done = {"confirmed": "Label kept.", "corrected": "Label changed.", "set_aside": "Bee set aside."}.get(request.args.get("done", ""))
        return render_template_string(
            _REVIEW_PAGE, css=base_css, user=user, q=queue, s=suspect, stats=engine.review_stats(),
            quick_labels=engine.quick_labels(), error=error, message=done, enabled=engine.cfg.review.enabled,
            threshold=engine.cfg.review.min_disagreement, min_class_size=engine.cfg.review.min_class_size,
        )

    @bp.route("/al-review-file/<filename>")
    def review_file(filename: str):
        """Scarica il log delle decisioni (jsonl) o lo stesso contenuto come CSV, una riga per decisione."""
        if not session.get("user_email"):
            abort(403)
        engine, _ = engine_or_message()
        if engine is None or filename not in ("review_log.jsonl", "review_report.csv"):
            abort(404)
        log = read_log(engine.review_log_path)
        if filename == "review_log.jsonl":
            body, mime = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log), "application/x-ndjson"
        else:
            fields = ["timestamp", "annotator", "image_id", "decision", "old_label", "new_label", "suggested", "p_suggested",
                      "annotator_confidence", "model_version", "round", "rare_class"]
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(log)
            body, mime = buffer.getvalue(), "text/csv"
        return Response(body, mimetype=mime, headers={"Content-Disposition": f"attachment; filename={filename}"})

    @bp.route("/al-clusters", methods=["GET", "POST"])
    def clusters():
        """Ispezione di un gruppo alla volta, scelto da un menù a tendina, con le api grandi come nella griglia.

        Dodici api: le più simili al gruppo, quelle intorno alla mediana e le meno simili. Si correggono
        con i colori, come nell'etichettatura a griglia.
        """
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        labels_key = f"al_grid_labels_{engine.cfg.stage.name}"
        labels = list(session.get(labels_key) or [])
        if not labels:
            suggested = [engine.store.normalize_label(c) for c in engine.cfg.labels.classes]
            labels = list(dict.fromkeys(suggested + engine.quick_labels()))
        labels = (labels + [""] * len(GRID_COLORS))[:len(GRID_COLORS)]

        def shown(name: str) -> str:
            return "Scarta ape (set aside)" if name == UNKNOWN_LABEL else name

        predicted = [{"key": f"p:{c['label']}", "name": c["label"], "n": c["n_predicted"]} for c in engine.cluster_samples(0)]
        n_unassigned = engine.unassigned_samples(0)["n"]
        if n_unassigned:
            predicted.append({"key": "u", "name": "Unassigned - possible new class", "n": n_unassigned})
        by_hand = [{"key": f"h:{c['label']}", "name": shown(c["label"]), "n": c["n_human"]} for c in engine.labeled_samples(0)]
        menu = [(title, options) for title, options in (("Model predictions", predicted), ("Labels given by hand", by_hand)) if options]
        sizes = {o["key"]: o["n"] for _, options in menu for o in options}
        group = request.args.get("group", "")
        if not sizes.get(group):  # pagina aperta senza scelta, o gruppo svuotato da una correzione
            wanted = "h:" if request.args.get("view") == "labeled" else "p:"
            filled = [k for k, size in sizes.items() if size]
            group = next((k for k in filled if k.startswith(wanted)), filled[0] if filled else "")
        page_no = request.args.get("page", default=0, type=int) or 0
        kind = {"p": "predicted", "h": "labeled", "u": "unassigned"}.get(group[:1])

        error = None
        if request.method == "POST" and kind:
            labels = [request.form.get(f"label_{g}", "").strip() for g in range(len(GRID_COLORS))]
            session[labels_key] = labels
            chosen: dict[str, list[str]] = {}
            for k in range(GRID_SIZE):
                image_id, colour = request.form.get(f"image_{k}"), request.form.get(f"group_{k}", "")
                if not image_id or not colour:
                    continue
                if colour == "discard":
                    chosen.setdefault(UNKNOWN_LABEL, []).append(image_id)
                elif colour == "remove" and kind == "labeled":
                    chosen.setdefault(REMOVED_LABEL, []).append(image_id)
                elif colour.isdigit() and int(colour) < len(labels) and labels[int(colour)]:
                    chosen.setdefault(labels[int(colour)], []).append(image_id)
                else:
                    error = "A colour is used on some bees but has no class: type its class and save again."
            if error is None and not chosen:
                error = "No bee has a colour yet: pick a colour and click the bees."
            if error is None:
                try:  # tutto o niente: si controlla prima di scrivere
                    for label, image_ids in chosen.items():
                        if not engine.store.normalize_label(label) or any(i not in engine.bank for i in image_ids):
                            raise ALError("cannot label the selected bees")
                    changed = sum(engine.assign_many(user, image_ids, label) for label, image_ids in chosen.items())
                except ALError as e:
                    error = str(e)
                else:
                    return redirect(url_for("al.clusters", group=group, page=page_no, changed=changed))

        view = engine.cluster_view(kind, group[2:], GRID_SIZE, page_no) if kind else None
        if view is not None:
            bands = {"high": "very similar", "median": "median", "low": "least similar"}
            for b in view["samples"]:
                b["band"] = bands[b["band"]]
                b["tip"] = "".join(f" - {k}: {b[k]}" for k in ("confidence", "nearest_class", "annotator") if b.get(k) not in (None, ""))
            if kind == "predicted":
                view["reference"] = "the centre of this cluster"
                view["summary"] = f"{view['n_members']} bees assigned by the model to \"{view['label']}\" (hand-labeled bees are listed apart)."
            elif kind == "labeled":
                view["reference"] = "the centre of this class"
                view["summary"] = f"{view['n_members']} bees labeled by hand as \"{shown(view['label'])}\"."
            else:
                view["reference"] = "their closest labeled bee"
                view["summary"] = f"{view['n_members']} bees far from every labeled example, kept out of the clusters."
        changed = request.args.get("changed", type=int)
        return render_template_string(
            _CLUSTERS_PAGE, css=base_css, user=user, s=engine.status(), group=group, menu=menu, view=view,
            groups=[{"name": name, "color": color, "label": labels[g]} for g, (name, color) in enumerate(GRID_COLORS)],
            quick_labels=engine.quick_labels(), can_remove=kind == "labeled", error=error,
            message=None if changed is None else f"{changed} bee(s) updated. Predictions recomputed.",
        )

    @bp.route("/al-refresh", methods=["POST"])
    def refresh():
        if not session.get("user_email"):
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        engine.refresh()
        return redirect(url_for("al.status"))

    return bp
