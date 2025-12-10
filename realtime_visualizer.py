#!/usr/bin/env python3
"""
Real-time visualizer for AEPsych color discrimination experiment.

Polls the AEPsych database and displays:
- Collected data points in model space
- Color coded by response (correct=blue, incorrect=red)
- Updates after each trial

Run this script alongside the Unity experiment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse


class RealtimeVisualizer:
    """Real-time visualization of AEPsych experiment progress."""
    
    def __init__(
        self,
        db_path: str = "databases/default.db",
        poll_interval: float = 1.0,
        model_bounds: Tuple[float, float] = (-0.7, 0.7),
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Initialize the visualizer.
        
        Args:
            db_path: Path to AEPsych SQLite database
            poll_interval: Seconds between database polls
            model_bounds: Bounds of the model space
            figsize: Figure size in inches
        """
        self.db_path = Path(db_path)
        self.poll_interval = poll_interval
        self.model_bounds = model_bounds
        self.figsize = figsize
        
        self.last_trial_count = 0
        self.trial_data: List[Dict] = []
        
        # Setup figure
        self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)
        self.fig.suptitle("AEPsych Color Discrimination - Real-time Monitor", fontsize=14)
        
        # Reference locations plot (left)
        self.ax_ref = self.axes#[0]
        self.ax_ref.set_title("Reference Locations")
        self.ax_ref.set_xlabel("Model dimension 1")
        self.ax_ref.set_ylabel("Model dimension 2")
        self.ax_ref.set_xlim(model_bounds[0] - 0.1, model_bounds[1] + 0.1)
        self.ax_ref.set_ylim(model_bounds[0] - 0.1, model_bounds[1] + 0.1)
        self.ax_ref.set_aspect('equal')
        self.ax_ref.grid(True, alpha=0.3)
        
        # Delta (offset) plot (right)
        # self.ax_delta = self.axes[1]
        # self.ax_delta.set_title("Delta Offsets")
        # self.ax_delta.set_xlabel("Delta dimension 1")
        # self.ax_delta.set_ylabel("Delta dimension 2")
        # delta_bound = 0.4
        # self.ax_delta.set_xlim(-delta_bound, delta_bound)
        # self.ax_delta.set_ylim(-delta_bound, delta_bound)
        # self.ax_delta.set_aspect('equal')
        # self.ax_delta.grid(True, alpha=0.3)
        # self.ax_delta.axhline(y=0, color='k', linewidth=0.5)
        # self.ax_delta.axvline(x=0, color='k', linewidth=0.5)
        
        # Legend
        correct_patch = mpatches.Patch(color='blue', alpha=0.6, label='Correct')
        incorrect_patch = mpatches.Patch(color='red', alpha=0.6, label='Incorrect')
        #self.fig.legend(handles=[correct_patch, incorrect_patch], loc='upper right')
        
        # Stats text
        self.stats_text = self.fig.text(
            0.02, 0.02, "", fontsize=10, family='monospace',
            verticalalignment='bottom'
        )
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        
    def parse_trial_from_db(self, message_content) -> Optional[Dict]:
        """Parse a tell message from the database."""
        try:
            # Handle binary (pickle) data
            if isinstance(message_content, bytes):
                import pickle
                try:
                    data = pickle.loads(message_content)
                except:
                    # Try JSON if pickle fails
                    data = json.loads(message_content.decode('utf-8', errors='ignore'))
            elif isinstance(message_content, str):
                data = json.loads(message_content)
            else:
                data = message_content
            
            # Handle different message formats
            if isinstance(data, dict):
                msg_type = data.get('type', '')
                if msg_type != 'tell':
                    return None
                    
                msg = data.get('message', {})
                config = msg.get('config', {})
                outcome = msg.get('outcome')
            else:
                return None
            
            if outcome is None:
                return None
            
            # Handle both parameterizations
            if 'x0_dim1' in config:
                ref = np.array([
                    config['x0_dim1'][0] if isinstance(config['x0_dim1'], list) else config['x0_dim1'],
                    config['x0_dim2'][0] if isinstance(config['x0_dim2'], list) else config['x0_dim2']
                ])
                
                # Check for delta parameterization
                if 'delta_dim1' in config:
                    delta = np.array([
                        config['delta_dim1'][0] if isinstance(config['delta_dim1'], list) else config['delta_dim1'],
                        config['delta_dim2'][0] if isinstance(config['delta_dim2'], list) else config['delta_dim2']
                    ])
                    comp = ref + delta
                else:
                    comp = np.array([
                        config['x1_dim1'][0] if isinstance(config['x1_dim1'], list) else config['x1_dim1'],
                        config['x1_dim2'][0] if isinstance(config['x1_dim2'], list) else config['x1_dim2']
                    ])
                    delta = comp - ref
                
                return {
                    'ref': ref,
                    'comp': comp,
                    'delta': delta,
                    'outcome': float(outcome),
                    'correct': float(outcome) > 0.5
                }
        except Exception as e:
            # Silent fail for unparseable messages
            pass
        
        return None
    
    def load_trials_from_db(self) -> List[Dict]:
        """Load all trials from the database."""
        if not self.db_path.exists():
            print(f"Database not found: {self.db_path}")
            return []
        
        trials = []
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.text_factory = bytes  # Don't auto-decode binary data
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = []
            for row in cursor.fetchall():
                name = row[0]
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                tables.append(name)
            
            # Try replay tables first
            replay_tables = [t for t in tables if t.startswith('replay_')]
            
            for table_name in replay_tables:
                try:
                    # Get column names
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = []
                    for col in cursor.fetchall():
                        col_name = col[1]
                        if isinstance(col_name, bytes):
                            col_name = col_name.decode('utf-8')
                        columns.append(col_name)
                    
                    # Find the message content column
                    content_col = None
                    for col in ['message_contents', 'message', 'content', 'request']:
                        if col in columns:
                            content_col = col
                            break
                    
                    if content_col is None:
                        continue
                    
                    # Try to find type column
                    type_col = 'message_type' if 'message_type' in columns else None
                    
                    if type_col:
                        cursor.execute(f"""
                            SELECT {content_col} FROM {table_name}
                            WHERE {type_col} = 'tell'
                            ORDER BY timestamp
                        """)
                    else:
                        cursor.execute(f"SELECT {content_col} FROM {table_name}")
                    
                    rows = cursor.fetchall()
                    
                    for (content,) in rows:
                        trial = self.parse_trial_from_db(content)
                        if trial:
                            trials.append(trial)
                            
                except Exception as e:
                    print(f"Warning: Error reading table {table_name}: {e}")
                    continue
            
            # Also try raw_data table if it exists (some AEPsych versions)
            if 'raw_data' in tables:
                try:
                    cursor.execute("PRAGMA table_info(raw_data)")
                    columns = []
                    for col in cursor.fetchall():
                        col_name = col[1]
                        if isinstance(col_name, bytes):
                            col_name = col_name.decode('utf-8')
                        columns.append(col_name)
                    
                    # Try to read trial data directly
                    if all(c in columns for c in ['x', 'y']):
                        cursor.execute("SELECT x, y FROM raw_data")
                        for x_data, y_data in cursor.fetchall():
                            try:
                                import pickle
                                x = pickle.loads(x_data) if isinstance(x_data, bytes) else x_data
                                y = pickle.loads(y_data) if isinstance(y_data, bytes) else y_data
                                
                                # x should be [n_trials, n_dims]
                                if hasattr(x, 'numpy'):
                                    x = x.numpy()
                                if hasattr(y, 'numpy'):
                                    y = y.numpy()
                                    
                                x = np.array(x)
                                y = np.array(y)
                                
                                # Handle 4D case (ref + comp or ref + delta)
                                if x.ndim == 1 and len(x) >= 4:
                                    ref = x[:2]
                                    delta_or_comp = x[2:4]
                                    # Assume delta if values are small
                                    if np.abs(delta_or_comp).max() < 0.5:
                                        delta = delta_or_comp
                                        comp = ref + delta
                                    else:
                                        comp = delta_or_comp
                                        delta = comp - ref
                                    
                                    trials.append({
                                        'ref': ref,
                                        'comp': comp,
                                        'delta': delta,
                                        'outcome': float(y),
                                        'correct': float(y) > 0.5
                                    })
                            except:
                                continue
                except Exception as e:
                    print(f"Warning: Error reading raw_data: {e}")
            
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")
        
        return trials
    
    def update_plot(self):
        """Update the visualization with latest data."""
        self.trial_data = self.load_trials_from_db()
        n_trials = len(self.trial_data)
        
        if n_trials == 0:
            return
        
        # Clear axes
        self.ax_ref.cla()
        #self.ax_delta.cla()
        
        # Extract data
        refs = np.array([t['ref'] for t in self.trial_data])
        comps = np.array([t['comp'] for t in self.trial_data])
        deltas = np.array([t['delta'] for t in self.trial_data])
        correct = np.array([t['correct'] for t in self.trial_data])
        
        # Generate colors based on reference location (hue from angle)
        angles = np.arctan2(refs[:, 1], refs[:, 0])
        colors = plt.cm.hsv((angles + np.pi) / (2 * np.pi))
        
        # Left panel: Connected ref-comparison pairs
        for i in range(n_trials):
            # Thin line connecting ref to comparison
            line_color = 'green' if correct[i] else 'red'
            self.ax_ref.plot([refs[i, 0], comps[i, 0]], [refs[i, 1], comps[i, 1]], 
                           color=line_color, linewidth=0.5, alpha=0.7)
            # Comparison stimulus (circle)
            self.ax_ref.scatter(comps[i, 0], comps[i, 1], 
                              c=[colors[i]], marker='o', s=25, alpha=0.8, zorder=2)
            # Reference stimulus (plus)
            self.ax_ref.scatter(refs[i, 0], refs[i, 1], 
                              c=[colors[i]], marker='+', s=50, linewidths=1.2, alpha=0.9, zorder=3)
        
        # Legend entries
        self.ax_ref.scatter([], [], c='gray', marker='+', s=50, linewidths=1.2, label='Reference stimulus')
        self.ax_ref.plot([], [], c='gray', marker='o', markersize=4, linewidth=0.5, label='Paired comparison')
        
        self.ax_ref.set_title(f"Trials (n={n_trials})")
        self.ax_ref.set_xlabel("Model space dimension 1")
        self.ax_ref.set_ylabel("Model space dimension 2")
        self.ax_ref.set_xlim(self.model_bounds[0] - 0.1, self.model_bounds[1] + 0.1)
        self.ax_ref.set_ylim(self.model_bounds[0] - 0.1, self.model_bounds[1] + 0.1)
        self.ax_ref.set_aspect('equal')
        self.ax_ref.grid(True, alpha=0.3)
        #self.ax_ref.legend(loc='upper right', fontsize=8)
        # Shaded valid region
        self.ax_ref.fill_between([-1, 1], -1, 1, alpha=0.08, color='gray')
        
        # Right panel: Delta offsets colored by correctness
        # self.ax_delta.set_title("Delta Offsets")
        # self.ax_delta.set_xlabel("Δ dimension 1")
        # self.ax_delta.set_ylabel("Δ dimension 2")
        
        # if correct.sum() > 0:
        #     self.ax_delta.scatter(deltas[correct, 0], deltas[correct, 1],
        #                          c='green', alpha=0.5, s=30, label='Correct')
        # if (~correct).sum() > 0:
        #     self.ax_delta.scatter(deltas[~correct, 0], deltas[~correct, 1],
        #                          c='red', alpha=0.5, s=30, marker='x', label='Incorrect')
        
        # delta_bound = max(np.abs(deltas).max() * 1.2, 0.3)
        # self.ax_delta.set_xlim(-delta_bound, delta_bound)
        # self.ax_delta.set_ylim(-delta_bound, delta_bound)
        # self.ax_delta.set_aspect('equal')
        # self.ax_delta.grid(True, alpha=0.3)
        # self.ax_delta.axhline(y=0, color='k', linewidth=0.5)
        # self.ax_delta.axvline(x=0, color='k', linewidth=0.5)
        # self.ax_delta.legend(loc='upper right', fontsize=8)
        
        # Update stats
        n_correct = correct.sum()
        accuracy = n_correct / n_trials if n_trials > 0 else 0
        mean_delta_mag = np.mean(np.linalg.norm(deltas, axis=1))
        
        stats_str = (
            f"Trials: {n_trials} | "
            f"Correct: {n_correct} ({accuracy:.1%}) | "
            f"Mean |Δ|: {mean_delta_mag:.3f}"
        )
        self.stats_text.set_text(stats_str)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        if n_trials != self.last_trial_count:
            print(f"[{time.strftime('%H:%M:%S')}] {stats_str}")
            self.last_trial_count = n_trials
    
    def run(self):
        """Run the real-time visualization loop."""
        print("=" * 60)
        print("Real-time AEPsych Visualizer")
        print("=" * 60)
        print(f"Database: {self.db_path}")
        print(f"Poll interval: {self.poll_interval}s")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        plt.show(block=False)
        
        try:
            while True:
                self.update_plot()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nStopping visualizer...")
        finally:
            plt.ioff()
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Real-time AEPsych visualizer")
    parser.add_argument(
        '--db', type=str, default='databases/default.db',
        help='Path to AEPsych database'
    )
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Poll interval in seconds'
    )
    parser.add_argument(
        '--bounds', type=float, nargs=2, default=[-0.7, 0.7],
        help='Model space bounds'
    )
    args = parser.parse_args()
    
    visualizer = RealtimeVisualizer(
        db_path=args.db,
        poll_interval=args.interval,
        model_bounds=tuple(args.bounds)
    )
    visualizer.run()


if __name__ == "__main__":
    main()