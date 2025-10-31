#!/usr/bin/env python3
"""
File I/O utilities for the psychophysical experiment.
Handles communication between stimulus presentation and adaptive engine.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentFileIO:
    """Handles file-based communication for the dual-computer setup."""

    def __init__(self, shared_drive_path: str = "/tmp/shared_drive", participant_id: str = "P01"):
        """
        Initialize file I/O handler.

        Args:
            shared_drive_path: Path to shared network drive
            participant_id: Participant identifier
        """
        self.shared_drive = Path(shared_drive_path)
        self.participant_id = participant_id
        self.participant_dir = self.shared_drive / participant_id

        # Ensure directories exist
        self.participant_dir.mkdir(parents=True, exist_ok=True)

    def write_next_trial(self, session_index: int, trial_data: Dict[str, Any]) -> str:
        """
        Write next trial information for stimulus PC to read.

        Args:
            session_index: Current session number
            trial_data: Trial information dictionary

        Returns:
            Path to the written file
        """
        session_dir = self.participant_dir / f"S{session_index:02d}"
        to_stimulus_dir = session_dir / "to_stimulus_pc"
        to_stimulus_dir.mkdir(parents=True, exist_ok=True)

        # Remove any existing next_trial.json
        next_trial_file = to_stimulus_dir / "next_trial.json"
        if next_trial_file.exists():
            next_trial_file.unlink()

        # Write new trial data
        with open(next_trial_file, 'w') as f:
            json.dump(trial_data, f, indent=2)

        return str(next_trial_file)

    def read_response(self, session_index: int) -> Optional[Dict[str, Any]]:
        """
        Read participant response from stimulus PC.

        Args:
            session_index: Current session number

        Returns:
            Response data dictionary or None if no response available
        """
        session_dir = self.participant_dir / f"S{session_index:02d}"
        from_stimulus_dir = session_dir / "from_stimulus_pc"
        from_stimulus_dir.mkdir(parents=True, exist_ok=True)

        # Find the most recent response file
        response_files = list(from_stimulus_dir.glob("response_*.json"))
        if not response_files:
            return None

        # Sort by timestamp (filename contains timestamp)
        latest_file = max(response_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_file, 'r') as f:
                response_data = json.load(f)

            # Mark as processed by deleting the file
            latest_file.unlink()

            return response_data

        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def write_session_status(self, session_index: int, status: str):
        """
        Write session status.

        Args:
            session_index: Current session number
            status: Status string ("RUNNING", "PAUSED", "COMPLETED")
        """
        session_dir = self.participant_dir / f"S{session_index:02d}"
        status_file = session_dir / "SESSION_STATUS.txt"

        with open(status_file, 'w') as f:
            f.write(status)

    def read_session_status(self, session_index: int) -> str:
        """
        Read session status.

        Args:
            session_index: Current session number

        Returns:
            Status string
        """
        session_dir = self.participant_dir / f"S{session_index:02d}"
        status_file = session_dir / "SESSION_STATUS.txt"

        if not status_file.exists():
            return "UNKNOWN"

        try:
            with open(status_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return "UNKNOWN"

    def generate_trial_data(self, participant_id: str, session_index: int, trial_index: int,
                          trial_type: str, stimuli_rgb: list) -> Dict[str, Any]:
        """
        Generate trial data dictionary in the format expected by stimulus PC.

        Args:
            participant_id: Participant identifier
            session_index: Session number
            trial_index: Trial number
            trial_type: "ADAPTIVE", "VALIDATION", or "FALLBACK"
            stimuli_rgb: List of RGB tuples for the three stimuli

        Returns:
            Trial data dictionary
        """
        return {
            "participant_id": participant_id,
            "session_index": session_index,
            "trial_index": trial_index,
            "trial_type": trial_type,
            "stimuli": [
                {"type": "reference" if i < 2 else "comparison", "rgb": list(rgb)}
                for i, rgb in enumerate(stimuli_rgb)
            ]
        }

    def generate_response_data(self, participant_id: str, session_index: int, trial_index: int,
                             trial_type: str, response_correct: bool, response_time_ms: float,
                             stimuli_shown: Dict[str, list]) -> Dict[str, Any]:
        """
        Generate response data dictionary in the format written by stimulus PC.

        Args:
            participant_id: Participant identifier
            session_index: Session number
            trial_index: Trial number
            trial_type: Type of trial shown
            response_correct: Whether response was correct
            response_time_ms: Response time in milliseconds
            stimuli_shown: Dictionary with reference_rgb and comparison_rgb

        Returns:
            Response data dictionary
        """
        timestamp = int(time.time() * 1000)  # millisecond timestamp

        return {
            "participant_id": participant_id,
            "session_index": session_index,
            "trial_index": trial_index,
            "trial_type_shown": trial_type,
            "response_correct": response_correct,
            "response_time_ms": response_time_ms,
            "stimuli_shown": stimuli_shown
        }

# Convenience functions for simple testing
def create_test_trial_file(shared_drive_path: str = "/tmp/shared_drive"):
    """Create a test trial file for development."""
    file_io = ExperimentFileIO(shared_drive_path)

    # Generate test trial data
    stimuli_rgb = [
        [0.5, 0.5, 0.5],  # Reference 1
        [0.5, 0.5, 0.5],  # Reference 2 (same as reference 1)
        [0.52, 0.48, 0.5]  # Comparison (slightly different)
    ]

    trial_data = file_io.generate_trial_data(
        participant_id="P01",
        session_index=1,
        trial_index=1,
        trial_type="ADAPTIVE",
        stimuli_rgb=stimuli_rgb
    )

    file_path = file_io.write_next_trial(1, trial_data)
    print(f"Test trial file created: {file_path}")
    return file_path

def create_test_response_file(shared_drive_path: str = "/tmp/shared_drive"):
    """Create a test response file for development."""
    file_io = ExperimentFileIO(shared_drive_path)

    response_data = file_io.generate_response_data(
        participant_id="P01",
        session_index=1,
        trial_index=1,
        trial_type="ADAPTIVE",
        response_correct=True,
        response_time_ms=750.5,
        stimuli_shown={
            "reference_rgb": [0.5, 0.5, 0.5],
            "comparison_rgb": [0.52, 0.48, 0.5]
        }
    )

    # Write to from_stimulus directory
    session_dir = file_io.participant_dir / "S01"
    from_stimulus_dir = session_dir / "from_stimulus_pc"
    from_stimulus_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    response_file = from_stimulus_dir / f"response_{timestamp}.json"

    with open(response_file, 'w') as f:
        json.dump(response_data, f, indent=2)

    print(f"Test response file created: {response_file}")
    return str(response_file)

if __name__ == "__main__":
    # Test the file I/O functions
    print("Testing file I/O utilities...")

    # Create test files
    create_test_trial_file()
    create_test_response_file()

    # Test reading
    file_io = ExperimentFileIO()
    response = file_io.read_response(1)
    if response:
        print(f"Read response: {response}")
    else:
        print("No response file found")

    print("File I/O test completed!")