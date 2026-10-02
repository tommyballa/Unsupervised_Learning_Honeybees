"""Pagine Flask dell'active learning, registrate dall'hub come blueprint."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional, Union

from flask import Blueprint, abort, redirect, render_template_string, request, send_file, session, url_for

from . import REMOVED_LABEL, UNKNOWN_LABEL
from .engine import ALEngine, ALError
from .retrain import RetrainManager


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
    <h2>{{ 'Validation Set Labeling' if mode == 'validation' else 'Active Learning Labeling' }}</h2>
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
        <button class="btn" type="submit" name="action" value="unknown" formnovalidate style="background: var(--input-bg);">Non so</button>
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
        <tr><th>Set aside ("Non so")</th><td>{{ s.n_unknown }}</td></tr>
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

    <h2>Rounds</h2>
    <table>
        <tr><th>Round</th><th>Labels</th><th>Classes</th><th>Val. labels</th><th>Accuracy</th><th>Macro-F1</th><th>Model</th></tr>
        {% for h in s.history[-15:]|reverse %}
        <tr><td>{{ h.round }}</td><td>{{ h.n_labels }}</td><td>{{ h.n_classes }}</td><td>{{ h.n_validation }}</td>
            <td>{{ '%.3f' % h.accuracy if h.accuracy is not none else '-' }}</td>
            <td>{{ '%.3f' % h.macro_f1 if h.macro_f1 is not none else '-' }}</td><td>{{ h.model_version }}</td></tr>
        {% endfor %}
    </table>
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
        <h3><span>{{ 'Non so (set aside)' if c.unknown else c.label }}
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
            <button class="btn" type="submit" name="action" value="unknown" style="flex: 1; margin-top: 0; background: var(--input-bg);">Non so</button>
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
        mode = "validation" if request.args.get("mode") == "validation" else "train"
        last_key = f"al_last_{mode}"
        error = None

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
            previous_label = engine.val_labels.get(image_id) if mode == "validation" else engine.bank.label_of(image_id)
            previous_label = "Non so" if previous_label == UNKNOWN_LABEL else (previous_label or "")
        else:
            item = engine.next_image(user, mode=mode)
            if item is None:
                if mode == "validation":
                    return message("Validation set complete", "Every validation image has been labeled (or is assigned to another annotator).")
                return message("Nothing to label right now", "The rest of the batch is assigned to other annotators, or no unlabeled images are left. Try again in a moment.")
            image_id = item.image_id

        st = engine.status()
        if mode == "validation":
            progress = f"Validation set: {st['validation_labeled']} / {st['validation_size']} labeled. These labels are used only for evaluation."
        else:
            progress = f"Batch {st['batch_id']}: {st['batch_done']} / {st['batch_size']} done. {st['n_labels']} labels in total."
        return render_template_string(
            _LABELING_PAGE, css=base_css, user=user, mode=mode, image_id=image_id, progress=progress,
            quick_labels=engine.quick_labels(), suggestion=engine.suggestion(image_id), error=error,
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
        )

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
