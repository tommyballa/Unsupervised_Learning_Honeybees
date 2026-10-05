"""Pagine Flask dell'active learning, registrate dall'hub come blueprint."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional, Union

from flask import Blueprint, abort, redirect, render_template_string, request, send_file, session, url_for

from . import REMOVED_LABEL, UNKNOWN_LABEL
from .audit import parse_scores
from .engine import ALEngine, ALError
from .report import REPORT_DIR, REPORT_FILES, build_report
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

_NAV = """
<div class="nav">
    <span><a href="/">&larr; Dashboard</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.labeling') }}">Labeling</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.labeling', mode='validation') }}">Validation set</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.labeling', mode='audit') }}">Audit</a> &nbsp;|&nbsp;
        <a href="{{ url_for('al.clusters') }}">Clusters</a> &nbsp;|&nbsp;
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

_STATUS_PAGE = """
{% if s.retrain and s.retrain.state == 'running' %}<meta http-equiv="refresh" content="5">{% endif %}
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
        <tr><th>Model version / config</th><td>{{ s.model_version }} / {{ s.config_id }}</td></tr>
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


_BEE_TILE = """
<label class="grid-item-container" style="width: 120px; text-align: center; font-size: 12px; color: #94a3b8;">
    <input type="checkbox" name="selected_images" value="{{ b.image_id }}">
    <img src="{{ url_for('al.image', image_id=b.image_id) }}" title="{{ b.image_id }}{{ tip }}">
    {{ caption }}
</label>
"""

_CLUSTERS_PAGE = """
{{ css|safe }}
<div class="container" style="max-width: 1100px;">
    """ + _NAV + """
    <h2>Clusters so far</h2>
    <p style="font-size: 14px;">
        <a href="{{ url_for('al.clusters', view='predicted', n=n, order=order) }}" class="quick-label-btn" {% if view == 'predicted' %}style="background: var(--btn-blue); color: white;"{% endif %}>Model predictions</a>
        <a href="{{ url_for('al.clusters', view='labeled', n=n) }}" class="quick-label-btn" {% if view == 'labeled' %}style="background: var(--btn-blue); color: white;"{% endif %}>Labels given by hand</a>
    </p>
    {% if view == 'predicted' %}
    <p>Bees the model assigned to each class (round {{ s.round }}, {{ s.n_labels }} labels). Bees labeled by hand are in the other view.</p>
    {% else %}
    <p>Bees labeled by hand, most recent first. Select the ones you changed your mind about and give them another class.</p>
    {% endif %}
    <p style="color: #94a3b8; font-size: 14px;">
        Show per cluster:
        {% for k, name in sizes %}
        <a href="{{ url_for('al.clusters', view=view, n=k, order=order) }}" {% if k == n %}style="color: var(--text-color); text-decoration: underline;"{% endif %}>{{ name }}</a>{{ ' · ' if not loop.last }}
        {% endfor %}
        {% if view == 'predicted' %}
        &nbsp;|&nbsp; Pick:
        {% for key, name in orders %}
        <a href="{{ url_for('al.clusters', view=view, n=n, order=key) }}" {% if key == order %}style="color: var(--text-color); text-decoration: underline;"{% endif %}>{{ name }}</a>{{ ' · ' if not loop.last }}
        {% endfor %}
        {% endif %}
    </p>
    {% if message %}<p style="color: var(--btn-green);"><b>{{ message }}</b></p>{% endif %}
    {% if error %}<p style="color: var(--btn-red);"><b>{{ error }}</b></p>{% endif %}

    <form method="POST" action="{{ url_for('al.clusters', view=view, n=n, order=order) }}">
    {% if view == 'predicted' and unassigned.n %}
    <div class="inspect-cluster" style="border-color: var(--btn-yellow);">
        <h3><span>Unassigned &mdash; possible new class
            <span style="font-size: 14px; color: #94a3b8; font-weight: normal;">{{ unassigned.n }} bees far from every labeled example, kept out of the clusters below</span></span>
            <a href="#" class="select-all" style="font-size: 13px;">select all shown</a>
        </h3>
        <div class="inspect-images">
            {% for b in unassigned.samples %}
            {% with tip = ' - nearest class: ' ~ b.nearest_class, caption = 'sim %.2f' % b.similarity %}""" + _BEE_TILE + """{% endwith %}
            {% endfor %}
        </div>
        <p style="color: #94a3b8; font-size: 13px; margin-bottom: 0;">The number under each bee is its similarity to the closest labeled bee.</p>
    </div>
    {% endif %}
    {% for c in clusters %}
    <div class="inspect-cluster">
        <h3><span>{{ 'Scarta ape (set aside)' if c.unknown else c.label }}
            <span style="font-size: 14px; color: #94a3b8; font-weight: normal;">
                {% if view == 'predicted' %}{{ c.n_predicted }} predicted · {{ c.n_human }} labeled by hand{% if c.mean_confidence is not none %} · mean confidence {{ '%.0f' % (c.mean_confidence * 100) }}%{% endif %}
                {% else %}{{ c.n_human }} labeled by hand{% endif %}
            </span>
            {% if c.reject %}<span style="font-size: 14px; color: var(--btn-red);">reject class</span>{% endif %}</span>
            {% if c.samples %}<a href="#" class="select-all" style="font-size: 13px;">select all shown</a>{% endif %}
        </h3>
        <div class="inspect-images">
            {% for b in c.samples %}
            {% if view == 'predicted' %}
            {% with tip = '', caption = '%.0f%%' % (b.confidence * 100) %}""" + _BEE_TILE + """{% endwith %}
            {% else %}
            {% with tip = ' - ' ~ b.annotator ~ ' (' ~ b.query_type ~ ')', caption = b.confidence %}""" + _BEE_TILE + """{% endwith %}
            {% endif %}
            {% else %}
            <p style="color: #94a3b8;">No bee is assigned to this class yet.</p>
            {% endfor %}
        </div>
    </div>
    {% else %}
    <p>{{ 'No clusters yet: label a few bees first.' if view == 'predicted' else 'No labels given by hand yet.' }}</p>
    {% endfor %}

    {% if clusters or (view == 'predicted' and unassigned.n) %}
    <div style="position: sticky; bottom: 0; background: var(--container-bg); border-top: 1px solid var(--border-color); padding: 12px 0 4px; z-index: 20;">
        <b><span id="al_selected_count">0</span> selected</b>
        <span style="color: #94a3b8; font-size: 13px;">&mdash; click bees to select them, then choose what they are</span>
        <div style="margin: 8px 0;">
            {% for lbl in quick_labels %}<button type="button" class="quick-label-btn" data-label="{{ lbl }}">{{ lbl }}</button>{% endfor %}
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <input type="text" id="al_label_input" name="label" autocomplete="off" placeholder="class for the selected bees (existing or new)" style="flex: 3; padding: 12px; font-size: 16px;">
            <button class="btn" type="submit" name="action" value="label" style="flex: 2; margin-top: 0;">Set class</button>
            <button class="btn" type="submit" name="action" value="unknown" style="flex: 1; margin-top: 0; background: var(--input-bg);">Scarta ape</button>
            {% if view == 'labeled' %}
            <button class="btn" type="submit" name="action" value="remove" style="flex: 1; margin-top: 0; background: var(--btn-red);">Remove label</button>
            {% endif %}
        </div>
    </div>
    {% endif %}
    </form>
</div>
<script>
(function () {
    var count = document.getElementById('al_selected_count');
    function update() { if (count) count.textContent = document.querySelectorAll('input[name="selected_images"]:checked').length; }
    document.querySelectorAll('input[name="selected_images"]').forEach(function (c) { c.addEventListener('change', update); });
    document.querySelectorAll('.select-all').forEach(function (a) {
        a.addEventListener('click', function (e) {
            e.preventDefault();
            var boxes = a.closest('.inspect-cluster').querySelectorAll('input[name="selected_images"]');
            var all = Array.prototype.every.call(boxes, function (b) { return b.checked; });
            boxes.forEach(function (b) { b.checked = !all; });
            update();
        });
    });
    document.querySelectorAll('.quick-label-btn[data-label]').forEach(function (b) {
        b.addEventListener('click', function () { document.getElementById('al_label_input').value = b.dataset.label; });
    });
})();
</script>
"""


def create_blueprint(base_css: str, config_path: Union[str, Path, None] = None) -> Blueprint:
    """Blueprint con ``/al-labeling``, ``/al-clusters``, ``/al-status``, ``/al-refresh`` e ``/al-retrain``.

    ``base_css`` è il CSS dell'hub, così le pagine nuove hanno lo stesso aspetto.
    """
    bp = Blueprint("al", __name__)
    STATE.config_path = config_path

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
            quick_labels=engine.quick_labels(), error=error,
            suggestion=None if mode == "audit" else engine.suggestion(image_id),  # l'audit è alla cieca
            redo=redo, previous_label=previous_label, has_previous=previous is not None,
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
        )

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

    @bp.route("/al-clusters", methods=["GET", "POST"])
    def clusters():
        user = session.get("user_email")
        if not user:
            return redirect(url_for("index"))
        engine, page = engine_or_message()
        if engine is None:
            return page
        sizes = ((6, "6"), (12, "12"), (24, "24"), (48, "48"), (2000, "all"))
        orders = (("random", "random sample"), ("confident", "most confident"), ("uncertain", "least confident"))
        view = "labeled" if request.args.get("view") == "labeled" else "predicted"
        n = request.args.get("n", default=12, type=int)
        n = min(max(n, 1), 2000)
        order = request.args.get("order", "random")
        if order not in dict(orders):
            order = "random"

        error = None
        if request.method == "POST":
            selected = request.form.getlist("selected_images")
            action = request.form.get("action", "label")
            label = {"unknown": UNKNOWN_LABEL, "remove": REMOVED_LABEL}.get(action, request.form.get("label", ""))
            try:
                if not selected:
                    raise ALError("Select at least one bee first.")
                changed = engine.assign_many(user, selected, label)
            except ALError as e:
                error = str(e)
            else:
                return redirect(url_for("al.clusters", view=view, n=n, order=order, changed=changed))

        changed = request.args.get("changed", type=int)
        message = None if changed is None else f"{changed} bee(s) updated. Predictions recomputed."
        clusters = engine.labeled_samples(n) if view == "labeled" else engine.cluster_samples(n, order)
        return render_template_string(
            _CLUSTERS_PAGE, css=base_css, user=user, s=engine.status(), view=view,
            clusters=clusters, unassigned=engine.unassigned_samples(n, order),
            quick_labels=engine.quick_labels(), message=message, error=error,
            n=n, order=order, sizes=sizes, orders=orders,
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
