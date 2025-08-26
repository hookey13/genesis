"""
Unit tests for focus pattern detection.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from genesis.tilt.indicators.focus_patterns import FocusPatternDetector


class TestFocusPatternDetector:
    """Tests for focus pattern detection."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = FocusPatternDetector(
            rapid_switch_threshold_ms=2000,
            window_size=50
        )
        
        assert detector.rapid_switch_threshold == timedelta(milliseconds=2000)
        assert detector.window_size == 50
        assert detector.window_active is True
        assert len(detector.focus_events) == 0
    
    def test_track_window_focus_gain(self):
        """Test tracking window gaining focus."""
        detector = FocusPatternDetector()
        
        # Start unfocused
        detector.window_active = False
        
        # Track gaining focus
        detector.track_window_focus(True, 5000)
        
        assert len(detector.focus_events) == 1
        assert detector.focus_events[0]["window_active"] is True
        assert detector.focus_events[0]["duration_ms"] == 5000
        assert detector.window_active is True
    
    def test_track_window_focus_loss(self):
        """Test tracking window losing focus."""
        detector = FocusPatternDetector()
        
        # Start focused
        detector.window_active = True
        
        # Track losing focus
        detector.track_window_focus(False, 10000)
        
        assert len(detector.focus_events) == 1
        assert detector.focus_events[0]["window_active"] is False
        assert detector.focus_events[0]["duration_ms"] == 10000
        assert detector.window_active is False
    
    def test_duration_tracking(self):
        """Test focus/unfocus duration tracking."""
        detector = FocusPatternDetector()
        
        # Simulate focus changes
        detector.track_window_focus(False, 5000)  # Lost focus for 5s
        detector.track_window_focus(True, 3000)   # Regained focus after 3s
        detector.track_window_focus(False, 8000)  # Lost focus for 8s
        
        assert len(detector.focus_durations) == 1
        assert detector.focus_durations[0] == 8.0  # 8 seconds
        assert len(detector.unfocus_durations) == 1
        assert detector.unfocus_durations[0] == 3.0  # 3 seconds
    
    @patch('genesis.tilt.indicators.focus_patterns.logger')
    def test_rapid_switching_detection(self, mock_logger):
        """Test detection of rapid focus switching."""
        detector = FocusPatternDetector(rapid_switch_threshold_ms=3000)
        
        # Simulate rapid switching
        now = datetime.utcnow()
        detector.last_focus_change = now - timedelta(milliseconds=1000)
        
        detector.track_window_focus(False, 1000)
        
        # Should detect rapid switch
        assert len(detector.rapid_switches) == 1
        mock_logger.warning.assert_called()
    
    def test_rapid_switches_cleanup(self):
        """Test cleanup of old rapid switch records."""
        detector = FocusPatternDetector()
        
        # Add old rapid switch
        old_time = datetime.utcnow() - timedelta(hours=2)
        detector.rapid_switches = [old_time]
        
        # Track new focus change
        detector.track_window_focus(True, 5000)
        
        # Old rapid switch should be removed
        assert len(detector.rapid_switches) == 0
    
    def test_get_focus_metrics_empty(self):
        """Test metrics when no events recorded."""
        detector = FocusPatternDetector()
        
        metrics = detector.get_focus_metrics(5)
        
        assert metrics.total_switches == 0
        assert metrics.switch_frequency == 0.0
        assert metrics.average_focus_duration == 0.0
        assert metrics.distraction_score == 0.0
    
    def test_get_focus_metrics_with_data(self):
        """Test metrics calculation with focus events."""
        detector = FocusPatternDetector()
        
        # Add some focus events
        now = datetime.utcnow()
        events = [
            {"timestamp": now - timedelta(minutes=3), "window_active": False, "duration_ms": 5000, "duration_seconds": 5.0},
            {"timestamp": now - timedelta(minutes=2), "window_active": True, "duration_ms": 10000, "duration_seconds": 10.0},
            {"timestamp": now - timedelta(minutes=1), "window_active": False, "duration_ms": 15000, "duration_seconds": 15.0},
        ]
        detector.focus_events = events
        detector.focus_durations = [10.0, 15.0, 20.0]
        
        metrics = detector.get_focus_metrics(5)
        
        assert metrics.total_switches == 3
        assert metrics.switch_frequency == 0.6  # 3 switches in 5 minutes
        assert metrics.average_focus_duration == 15.0  # (10+15+20)/3
    
    def test_distraction_score_calculation(self):
        """Test distraction score calculation."""
        detector = FocusPatternDetector()
        
        # Test low distraction
        score = detector._calculate_distraction_score(
            switch_frequency=0.5,
            rapid_switches=0,
            avg_focus_duration=60
        )
        assert score == 0.0
        
        # Test moderate distraction
        score = detector._calculate_distraction_score(
            switch_frequency=3.0,
            rapid_switches=1,
            avg_focus_duration=25
        )
        assert score > 20 and score < 50
        
        # Test high distraction
        score = detector._calculate_distraction_score(
            switch_frequency=5.0,
            rapid_switches=4,
            avg_focus_duration=10
        )
        assert score > 50
    
    def test_is_distracted(self):
        """Test distraction detection."""
        detector = FocusPatternDetector()
        
        # Set up high distraction scenario
        now = datetime.utcnow()
        detector.focus_events = [
            {"timestamp": now, "window_active": True, "duration_ms": 1000, "duration_seconds": 1.0}
            for _ in range(10)  # Many switches
        ]
        detector.rapid_switches = [now] * 5  # Many rapid switches
        detector.focus_durations = [5.0] * 10  # Short focus periods
        
        assert detector.is_distracted(threshold=30.0) is True
    
    def test_get_pattern_analysis(self):
        """Test comprehensive pattern analysis."""
        detector = FocusPatternDetector()
        
        # Set up test data
        now = datetime.utcnow()
        detector.focus_events = [
            {"timestamp": now - timedelta(minutes=i), "window_active": i % 2 == 0, 
             "duration_ms": 5000, "duration_seconds": 5.0}
            for i in range(6)
        ]
        detector.focus_durations = [30.0, 40.0, 35.0]
        
        analysis = detector.get_pattern_analysis()
        
        assert "attention_state" in analysis
        assert "distraction_score" in analysis
        assert "switch_frequency" in analysis
        assert "recommendation" in analysis
        assert analysis["attention_state"] in ["focused", "slightly_distracted", 
                                               "moderately_distracted", "highly_distracted"]
    
    def test_recommendation_generation(self):
        """Test recommendation based on metrics."""
        detector = FocusPatternDetector()
        
        # Import FocusMetrics for testing
        from genesis.tilt.indicators.focus_patterns import FocusMetrics
        
        # Test high distraction recommendation
        metrics = FocusMetrics(
            total_switches=20,
            switch_frequency=4.0,
            average_focus_duration=10.0,
            longest_focus=20.0,
            shortest_focus=5.0,
            rapid_switch_count=5,
            distraction_score=75.0
        )
        recommendation = detector._get_recommendation(metrics)
        assert "Take a break" in recommendation
        
        # Test rapid switching recommendation
        metrics = FocusMetrics(
            total_switches=10,
            switch_frequency=2.0,
            average_focus_duration=30.0,
            longest_focus=60.0,
            shortest_focus=10.0,
            rapid_switch_count=4,
            distraction_score=40.0
        )
        recommendation = detector._get_recommendation(metrics)
        assert "Slow down" in recommendation
        
        # Test short attention recommendation
        metrics = FocusMetrics(
            total_switches=5,
            switch_frequency=1.0,
            average_focus_duration=15.0,
            longest_focus=20.0,
            shortest_focus=10.0,
            rapid_switch_count=0,
            distraction_score=25.0
        )
        recommendation = detector._get_recommendation(metrics)
        assert "Practice focus" in recommendation
        
        # Test normal patterns
        metrics = FocusMetrics(
            total_switches=2,
            switch_frequency=0.4,
            average_focus_duration=120.0,
            longest_focus=180.0,
            shortest_focus=60.0,
            rapid_switch_count=0,
            distraction_score=5.0
        )
        recommendation = detector._get_recommendation(metrics)
        assert "normal" in recommendation