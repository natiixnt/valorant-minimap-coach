import time
from typing import TYPE_CHECKING, List, Optional

import yaml

from src.audio.audio_coach import AudioCoach
from src.audio.gunshot_detector import GunEvent
from src.audio.spike_audio import DefuseAdvisor
from src.audio.tts import TTSEngine
from src.capture.screen import ScreenCapture
from src.core.perf_monitor import PerfMonitor
from src.game.economy import EconomyTracker
from src.game.heatmap import Heatmap
from src.game.play_detector import PlayDetector, PlayType
from src.game.retake_advisor import RetakeAdvisor
from src.game.round_state import RoundState, State
from src.game.trajectory import TrajectoryPredictor
from src.game.ult_tracker import UltTracker
from src.game.zone_tracker import ZoneTracker
from src.game.enemy_agents import EnemyAgentTracker
from src.vision.agent_detector import AgentDetector
from src.maps.callouts import enemies_to_callout, pos_to_zone, stack_callout
from src.telemetry.collector import DataCollector
from src.vision.ability_detector import AbilityDetector
from src.vision.ai_analyzer import AIAnalyzer
from src.vision.detector import MinimapDetector
from src.vision.map_detector import MapDetector
from src.vision.player_angle import PlayerAngleDetector
from src.vision.spike_detector import SpikeDetector
from src.vision.team_detector import TeamDetector

if TYPE_CHECKING:
    from src.ui.overlay import OverlayWindow

try:
    from src.vision.local_analyzer import LocalAnalyzer
    _HAS_LOCAL = True
except ImportError:
    _HAS_LOCAL = False


def load_config(path: str = "config.yaml") -> dict:
    import os, sys
    if getattr(sys, "frozen", False) and not os.path.exists(path):
        import shutil
        for candidate in [path, "config.yaml.example"]:
            bundled = os.path.join(sys._MEIPASS, candidate)  # type: ignore[attr-defined]
            if os.path.exists(bundled):
                shutil.copy(bundled, path)
                print(f"[Config] Extracted {bundled} → {path}")
                break
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[Config] ERROR: {path} not found. Copy config.yaml.example to config.yaml.",
              file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[Config] ERROR: {path} is malformed: {e}", file=sys.stderr)
        sys.exit(1)


class Coach:
    def __init__(self, config: dict) -> None:
        self.config = config

        # Core CV
        self.capture               = ScreenCapture(config)
        self.detector              = MinimapDetector(config)
        self.team_detector         = TeamDetector(config)
        self.ability_detector      = AbilityDetector(config)
        self.spike_detector        = SpikeDetector()
        self.player_angle_detector = PlayerAngleDetector()

        # Audio
        self.tts         = TTSEngine(config)
        self.audio_coach = AudioCoach(config)

        # Game intelligence
        self.round_state   = RoundState()
        self.heatmap       = Heatmap()
        self.play_det      = PlayDetector()
        self.zone_tracker  = ZoneTracker()
        self.trajectory    = TrajectoryPredictor()
        self.retake        = RetakeAdvisor()
        self.economy       = EconomyTracker()
        self.ult_tracker   = UltTracker()
        self.defuse_advisor = DefuseAdvisor()
        self.enemy_agents   = EnemyAgentTracker()
        self.agent_detector = AgentDetector(config)
        self.audio_coach.enemy_agents = self.enemy_agents

        # AI + telemetry
        self.collector = DataCollector(config)
        if config["ai"]["enabled"]:
            use_local = config["ai"].get("use_local_model") and _HAS_LOCAL
            self.ai = (LocalAnalyzer(config, collector=self.collector)  # type: ignore[name-defined]
                       if use_local else AIAnalyzer(config, collector=self.collector))
        else:
            self.ai = None

        # Map
        self._map_override: Optional[str] = config.get("map_override")
        self.map_detector = (
            MapDetector(config, collector=self.collector)
            if not self._map_override else None
        )
        self.map_name: str = self._map_override or "unknown"

        # Performance
        self._perf = PerfMonitor(target_interval=0.1)

        # State
        self._prev_enemy_count   = 0
        self._prev_team: List    = []
        self._callout_lang       = "EN"
        self._lang_lock          = __import__("threading").Lock()
        self._stack_frames: dict = {}
        self._stack_warned: set  = set()
        self._spike_plant_time   = 0.0
        self._spike_armed        = False   # beep detector armed (candidate detected)
        self._recent_ai_callouts: List[str] = []
        self._audio_round_ended  = False
        self._running            = True
        self._shutdown_done      = False
        self._overlay: Optional["OverlayWindow"] = None

        # Wire round audio callbacks
        self.audio_coach.round_audio.on_round_start = self._on_round_start_audio
        self.audio_coach.round_audio.on_round_end   = self._on_round_end_audio

    # ------------------------------------------------------------------
    def set_overlay(self, overlay: "OverlayWindow") -> None:
        self._overlay = overlay
        overlay.on_mute_change          = self.tts.set_muted
        overlay.on_volume_change        = self.tts.set_volume
        overlay.on_feedback             = self.collector.submit_feedback
        overlay.on_minimap_region_change = self.capture.set_region

    def set_callout_lang(self, lang: str) -> None:
        with self._lang_lock:
            self._callout_lang = lang

    def _ui(self, fn, *args) -> None:
        if self._overlay:
            self._overlay.after(0, lambda: fn(*args))

    def _speak(self, text: str, priority: bool = False, ttl: float = 2.5) -> None:
        try:
            self.tts.speak(text, priority=priority, ttl=ttl)
        except Exception as e:
            print(f"[Coach] TTS error: {e}")

    # ------------------------------------------------------------------
    # Round audio callbacks (from AudioCoach background thread)
    # ------------------------------------------------------------------

    def _on_round_start_audio(self) -> None:
        self.round_state.on_round_start_sound()
        self.spike_detector.reset()
        self.retake.reset()
        self.zone_tracker.reset()
        self.trajectory.reset()
        self.defuse_advisor.reset()
        self.enemy_agents.new_round()
        self.audio_coach.on_spike_resolved()

        # Round 1: auto-detect enemy agents from HUD portrait strip
        if self.round_state.round_num == 1:
            def _on_detected(agents):
                if agents and self._overlay:
                    self._ui(
                        lambda: setattr(self._overlay, "_current_enemy_agents", agents)
                    )
            self.agent_detector.detect_async(self.enemy_agents, on_done=_on_detected)
        self._ui(self._overlay.hide_defuse_progress)  # type: ignore[union-attr]
        self._stack_frames.clear()
        self._stack_warned.clear()
        self._audio_round_ended = False
        self._spike_armed = False
        ult_warn = self.ult_tracker.update(self.round_state.round_num)
        if ult_warn:
            self._speak(ult_warn, ttl=12.0)          # informational, survives queue
        print(f"[Coach] Round {self.round_state.round_num} (audio trigger)")

    def _on_round_end_audio(self) -> None:
        self._audio_round_ended = True
        our_win = not self.spike_detector.is_planted
        self.economy.on_round_end(our_win=our_win)
        self.ult_tracker.on_round_end(self.round_state.round_num)
        self.round_state.on_round_end_sound()
        self.heatmap.end_round(self.round_state.round_num)
        econ = self.economy.status()
        self._speak(econ.voice, ttl=12.0)
        hot = self.heatmap.hottest_zones(2)
        if hot:
            self._speak(self.heatmap.summary(), ttl=15.0)  # round summary, no rush

    # ------------------------------------------------------------------
    # Map detection
    # ------------------------------------------------------------------

    def _startup_map_detection(self) -> None:
        if self._map_override:
            print(f"[Coach] Map override: {self._map_override}")
            self._ui(self._overlay.update_map, self._map_override)  # type: ignore[union-attr]
            return
        print("[Coach] Detecting map from screen...")
        detected = self.map_detector.wait_for_map()  # type: ignore[union-attr]
        self.map_name = detected
        self._ui(self._overlay.update_map, detected)  # type: ignore[union-attr]
        self._speak(f"Map detected: {detected}")

    def _refresh_map(self) -> None:
        if self._map_override or not self.map_detector:
            return
        new_map = self.map_detector.get_map()
        if new_map and new_map != self.map_name:
            self.map_name = new_map
            print(f"[Coach] Map changed: {new_map}")
            self._ui(self._overlay.update_map, new_map)  # type: ignore[union-attr]
            self._speak(f"New map: {new_map}")
            self.audio_coach.map_name = new_map
            # New game -- reset agent detection and composition
            self.agent_detector.reset()
            self.enemy_agents.clear()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._startup_map_detection()
        self.audio_coach.map_name = self.map_name
        self.audio_coach.start()
        print(f"[Coach] Running. Map: {self.map_name}. Ctrl+C to stop.")
        self._speak("Coach active")

        while self._running:
            self._perf.tick_start()
            try:
                self._tick()
            except Exception as e:
                print(f"[Coach] Tick error: {e}")
            time.sleep(self._perf.sleep_time())
            self._perf.tick_end()

        self._shutdown()

    def _tick(self) -> None:
        # -- Capture frame
        try:
            frame = self.capture.capture()
        except Exception as e:
            print(f"[Coach] Capture error: {e}")
            return
        if frame is None:
            return

        # -- CV detections
        result  = self.detector.detect(frame)
        team    = self.team_detector.detect(frame)
        appeared, gone = self.ability_detector.update(frame)

        # -- Round state machine
        spike_planted = self.spike_detector.is_planted
        rs_event = self.round_state.update(result.enemy_count, spike_planted)

        if rs_event == "round_end" and not self._audio_round_ended:
            # Fallback: audio callback wasn't triggered, update economy here
            our_win = not spike_planted
            self.economy.on_round_end(our_win=our_win)
            self.ult_tracker.on_round_end(self.round_state.round_num)
            self.heatmap.end_round(self.round_state.round_num)
            econ = self.economy.status()
            self._speak(econ.voice, ttl=12.0)          # informational

        if rs_event == "round_start":
            ult_warn = self.ult_tracker.update(self.round_state.round_num)
            if ult_warn:
                self._speak(ult_warn, ttl=12.0)

        # -- Enemy zone heatmap
        for x, y in result.enemies:
            zone = pos_to_zone(x, y, self.map_name)
            self.heatmap.add_sighting(zone, self.round_state.round_num)

        # -- Zone transition callouts (stale fast -- enemy keeps moving)
        for trans in self.zone_tracker.update(result.enemies, self.map_name):
            self._speak(trans, ttl=2.0)
            self._ui(self._overlay.update_callout, trans)  # type: ignore[union-attr]

        # -- Trajectory prediction (stale fast -- TTL matches 1.5s lookahead window)
        for pred in self.trajectory.update(result.enemies, self.map_name):
            self._speak(pred, ttl=1.5)
            self._ui(self._overlay.update_callout, pred)  # type: ignore[union-attr]

        # -- Play pattern detection (rush/split/lurk/execute)
        play = self.play_det.update(result.enemies, self.map_name)
        if play:
            # RUSH / EXECUTE / SPLIT are site-attack patterns -- flush queue and speak now.
            # LURK / MID_CTRL are informational -- queue normally, don't flush.
            urgent = play.play in (PlayType.RUSH, PlayType.EXECUTE, PlayType.SPLIT)
            self._speak(play.voice, priority=urgent, ttl=4.0)
            self._ui(self._overlay.update_callout, play.voice)  # type: ignore[union-attr]

        # -- UI: enemies
        self._ui(self._overlay.update_enemies, result.enemy_count, result.enemies)  # type: ignore[union-attr]

        # -- UI: active utility
        active_abs = [
            {"display": sig["display"], "color": sig["color"], "position": (0.5, 0.5)}
            for sig in self.ability_detector.active.values()
        ]
        self._ui(self._overlay.update_utility, active_abs)  # type: ignore[union-attr]

        # -- Player facing angle -> audio direction estimator
        try:
            player_angle = self.player_angle_detector.detect(frame)
            if player_angle is not None:
                self.audio_coach.player_facing = player_angle
        except Exception:
            pass

        # -- Player position (prefer team dots, fallback to enemy mirror)
        if team:
            avg_tx = sum(x for x, y in team) / len(team)
            avg_ty = sum(y for x, y in team) / len(team)
            self.audio_coach.player_pos = (
                max(0.05, min(0.95, avg_tx)),
                max(0.05, min(0.95, avg_ty)),
            )
        elif result.enemies:
            avg_ex = sum(x for x, y in result.enemies) / len(result.enemies)
            avg_ey = sum(y for x, y in result.enemies) / len(result.enemies)
            self.audio_coach.player_pos = (
                max(0.1, min(0.9, 1.0 - avg_ex)),
                max(0.1, min(0.9, 1.0 - avg_ey)),
            )

        # -- Spike detection (two-phase: minimap candidate → audio beep confirms)
        self.spike_detector.update(frame)

        if self.spike_detector.is_candidate and not self._spike_armed:
            # Phase 1: minimap sees spike for 4+ frames -- arm beep detector silently.
            # Do NOT announce yet: spike may be dropped on ground or mid-plant (cancelled).
            self._spike_armed = True
            self.audio_coach.arm_spike_audio()

        if self._spike_armed and not self.spike_detector.is_planted:
            # Phase 2: wait for first spike beep (audio-confirmed full plant)
            # or fall back to minimap-only after timeout (audio disabled / missed)
            audio_confirmed = self.audio_coach.spike_timer.has_started
            timeout = self.spike_detector.candidate_timeout_reached
            if audio_confirmed or timeout:
                pos = self.spike_detector.candidate_pos or (0.5, 0.5)
                zone = pos_to_zone(pos[0], pos[1], self.map_name)
                spike_text = f"Spike planted at {zone}! Rotate!"
                self._speak(spike_text, priority=True)
                self._ui(self._overlay.update_callout, spike_text)  # type: ignore[union-attr]
                self._spike_plant_time = time.monotonic()
                self.round_state.on_spike_planted()
                self.economy.on_spike_planted()
                self.audio_coach.on_spike_planted(self._spike_plant_time)
                self.defuse_advisor.reset()
                self.spike_detector.confirm_planted(pos)
                self._spike_armed = False

        # -- Retake advisor (post-plant phase)
        if self.spike_detector.is_planted and self.spike_detector.planted_pos and team:
            time_since_plant = time.monotonic() - self._spike_plant_time
            advice = self.retake.advise(
                self.spike_detector.planted_pos, team,
                self.map_name, time_since_plant,
            )
            if advice:
                self._speak(advice, priority=True)
                self._ui(self._overlay.update_callout, advice)  # type: ignore[union-attr]

        # -- Defuse feasibility (post-plant phase)
        if self.spike_detector.is_planted and self.spike_detector.planted_pos:
            remaining = self.audio_coach.spike_timer.remaining()
            if remaining is not None and remaining > 0:
                # Estimate travel time to spike from player position
                spike_zone = pos_to_zone(
                    self.spike_detector.planted_pos[0],
                    self.spike_detector.planted_pos[1],
                    self.map_name,
                )
                travel = self.retake._rank_teammates(
                    team if team else [self.audio_coach.player_pos],
                    spike_zone, self.map_name,
                )
                min_travel = travel[0][0] if travel else 5.0
                # half_defused: enemy stopped defusing at >50% -- only 3.5s needed next time
                defuse_pct_now = self.audio_coach.defuse_sound.progress() or 0.0
                defuse_advice = self.defuse_advisor.update(
                    remaining, min_travel, half_defused=defuse_pct_now >= 0.5
                )
                if defuse_advice:
                    self._speak(defuse_advice, priority=True)
                    self._ui(self._overlay.update_callout, defuse_advice)  # type: ignore[union-attr]

            # -- Defuse progress tracker (wall-clock from detected defuse start click)
            defuse_pct = self.audio_coach.defuse_sound.progress()
            if defuse_pct is not None:
                if defuse_pct >= 1.0:
                    # Timer expired -- hide bar and re-arm so a new E-press can be detected
                    self._ui(self._overlay.hide_defuse_progress)  # type: ignore[union-attr]
                    self.audio_coach.defuse_sound.reset()
                    self.audio_coach.defuse_sound.arm()
                else:
                    self._ui(self._overlay.update_defuse_progress, defuse_pct)  # type: ignore[union-attr]
            else:
                self._ui(self._overlay.hide_defuse_progress)  # type: ignore[union-attr]

        # -- New enemies spotted
        with self._lang_lock:
            _lang = self._callout_lang
        if result.enemy_count > self._prev_enemy_count:
            callout = enemies_to_callout(result.enemies, self.map_name, _lang)
            if callout:
                self._speak(callout, priority=True)
                self._ui(self._overlay.update_callout, callout)  # type: ignore[union-attr]

        # -- Site clear (only during active round to avoid buy-phase false positives)
        if (self._prev_enemy_count >= 2 and result.enemy_count == 0
                and self.round_state.state == State.ROUND_ACTIVE):
            self._speak("Site clear", ttl=4.0)
            self._ui(self._overlay.update_callout, "Site clear")  # type: ignore[union-attr]

        self._prev_enemy_count = result.enemy_count
        self._prev_team        = team

        # -- Stack detection (3+ same zone, 3 frames)
        zone_counts: dict = {}
        for x, y in result.enemies:
            z = pos_to_zone(x, y, self.map_name)
            zone_counts[z] = zone_counts.get(z, 0) + 1
        for z, cnt in zone_counts.items():
            if cnt >= 3:
                self._stack_frames[z] = self._stack_frames.get(z, 0) + 1
                if self._stack_frames[z] == 3 and z not in self._stack_warned:
                    self._stack_warned.add(z)
                    txt = stack_callout(z, cnt, self.map_name, _lang)
                    self._speak(txt, priority=True)
                    self._ui(self._overlay.update_callout, txt)  # type: ignore[union-attr]
            else:
                self._stack_frames.pop(z, None)
                self._stack_warned.discard(z)
        for z in list(self._stack_frames):
            if z not in zone_counts:
                self._stack_frames.pop(z, None)
                self._stack_warned.discard(z)

        # -- Ability callouts (voice=None means overlay-only, no TTS)
        for ab in appeared:
            zone = pos_to_zone(ab.position[0], ab.position[1], self.map_name)
            if ab.voice:
                txt = ab.voice.format(zone=zone)
                self._speak(txt, ttl=4.0)
                self._ui(self._overlay.update_callout, txt)  # type: ignore[union-attr]

        # -- AI deep analysis
        if self.ai and result.enemy_count > 0 and self.ai.should_analyze():
            try:
                active_kinds = list(self.ability_detector.active.keys())
                ai_callout = self.ai.analyze(
                    frame, result.enemy_count, self.map_name, active_kinds,
                    spike_active=self.spike_detector.is_planted,
                    recent_callouts=self._recent_ai_callouts,
                )
                if ai_callout:
                    self._speak(ai_callout, ttl=5.0)
                    sample_ts = self.collector.submit_minimap_callout(
                        frame.data, ai_callout, self.map_name, result.enemies,
                        spike_active=self.spike_detector.is_planted,
                        recent_callouts=self._recent_ai_callouts,
                    )
                    self._ui(self._overlay.update_ai, ai_callout, sample_ts)  # type: ignore[union-attr]
                    self._recent_ai_callouts = (self._recent_ai_callouts + [ai_callout])[-5:]
            except Exception as e:
                print(f"[Coach] AI error: {e}")

        # -- Pro audio: footstep zones + gunshots
        for item in self.audio_coach.poll():
            if isinstance(item, GunEvent):
                self._speak(item.voice, ttl=1.5)
                self._ui(self._overlay.update_callout, item.voice)  # type: ignore[union-attr]
            else:
                # AudioFinding (footstep)
                self._speak(item.voice_text, ttl=2.0)
                self._ui(self._overlay.update_callout, item.voice_text)  # type: ignore[union-attr]
                if item.zone and item.audio_clip is not None:
                    self.collector.submit_footstep_audio(
                        item.audio_clip, item.zone, item.agent,
                        self.map_name, item.surface,
                    )

        self._refresh_map()

    # ------------------------------------------------------------------
    def _shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.audio_coach.stop()
        self.tts.stop()
        self.capture.close()
        if self.map_detector:
            self.map_detector.close()
        self.agent_detector.close()
        print(f"[Coach] Stopped. Avg tick: {self._perf.avg_ms():.1f} ms  "
              f"p95: {self._perf.p95_ms():.1f} ms")
