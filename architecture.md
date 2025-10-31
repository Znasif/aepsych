System Architecture for Comprehensive Psychophysical Characterization
This document outlines the system architecture designed for the comprehensive and data-efficient characterization of human perceptual thresholds, as detailed in the referenced research. The system is built to address the "psychophysical curse of dimensionality" by combining intelligent, adaptive data collection with a robust validation framework.
The architecture is designed to support two distinct but interleaved types of testing:
Adaptive Trial Sampling: Efficiently gathers data for model fitting by selecting the most informative stimuli in real-time.
Validation Testing: Independently verifies the model's predictions using the traditional Method of Constant Stimuli (MOCS).
1. Core Methodological Components
1.1. Wishart Process Psychophysical Model (WPPM)
The WPPM is the central, semi-parametric model of the system. It is not used for real-time trial selection but is fit post-hoc to the adaptively sampled data.
Function: To model the internal noise that limits color discrimination across the entire stimulus space.
Core Assumption: The internal noise, characterized by a covariance matrix, varies smoothly across the stimulus space.
Output: A continuously varying field of covariance matrices that, in turn, defines the entire four-dimensional psychometric field. This allows for the prediction of discrimination performance for any pair of stimuli.
1.2. Adaptive Sampling Engine (AEPsych)
AEPsych is the computational engine responsible for the primary mode of data collection. It is a non-parametric approach that ensures the data collection is not biased by the assumptions of the final WPPM.
Function: To select the most informative stimulus pair for each trial to efficiently map discrimination thresholds.
Underlying Model: A probit-Bernoulli Gaussian Process (GP) model that assumes smooth variation in performance across the stimulus space.
Acquisition Function: Employs the Expected Absolute Volume Change (EAVC) to determine the next trial that will maximally reduce uncertainty about the 66.7% correct threshold level.
1.3. Validation Method (MOCS)
The Method of Constant Stimuli (MOCS) is a classic psychophysical method used to create an independent dataset for validating the final WPPM fit.
Function: To measure psychometric functions at 25 pre-selected reference points and chromatic directions.
Implementation: Validation trials are pre-generated and consist of 12 comparison stimulus levels for each of the 25 conditions.
2. System Implementation: A Dual-Computer Architecture
To ensure a smooth, real-time experimental flow for the participant while accommodating the computationally intensive nature of adaptive trial selection, the system is implemented using a decoupled, dual-computer architecture.
2.1. Rationale
The core challenge is that the AEPsych algorithm requires significant computation time (>1.5s) to update its model and select the next trial. A single-computer system would introduce long, variable delays between trials, harming participant engagement. The dual-computer setup separates the real-time stimulus presentation from the non-real-time computational workload.
2.2. Computer 1: Stimulus Presentation & Rendering
This machine is responsible for all real-time interaction with the participant.
Role: Renders the 3D visual scene, presents stimuli, and records participant responses.
Hardware: A capable graphics workstation (e.g., Alienware with NVIDIA GeForce RTX 3080).
Software:
Unity (v2022.3.24f1): The graphics engine used to construct and render the visual scene, including the 3D "blobby" stimuli.
Custom C# Scripts: To manage the trial sequence, apply gamma correction, and handle I/O.
Input/Output:
Input: Reads text files containing RGB values for the next trial from the shared network disk.
Output: Presents stimuli on a calibrated monitor (DELL U2723QE), receives participant responses via a gamepad (Logitech Gamepad F310), and writes response data to the shared disk.
2.3. Computer 2: Adaptive Trial Generation & Logic
This is the "brains" of the operation, performing all the heavy computation for the adaptive testing.
Role: Runs the AEPsych algorithm to determine the parameters for upcoming adaptive trials.
Hardware: A high-performance computing workstation (e.g., 12-core Intel i9 with NVIDIA GeForce RTX3070).
Software:
Python (3.11): The primary language for the experimental logic and model fitting.
JAX: A high-performance numerical computing library used to accelerate the AEPsych model updates and the final WPPM fitting.
Input/Output:
Input: Reads participant response files from the shared network disk.
Output: Writes text files containing the RGB values for the next adaptive trial to the shared network disk.
2.4. Inter-Computer Communication Protocol
Communication is managed asynchronously via a simple but effective file-based system.
Mechanism: A shared network disk accessible by both computers.
Protocol: A custom protocol based on the creation and polling of text files. The Stimulus PC polls for a "next trial" file, while the Logic PC polls for a "last response" file.
3. Workflow & Data Flow
3.1. Phase 1: Offline Calibration
Before the experiment, the display must be meticulously calibrated.
Hardware: SpectraScan PR-670 radiometer.
Software: MATLAB.
Process: The monitor's primaries and gamma functions are measured. Transformation matrices are derived to convert between the device-dependent RGB space and the perceptual "model space." An inverse gamma lookup table is created for color correction.
3.2. Phase 2: Online Experiment Execution & Trial Scheduling
This phase details how the two testing types (adaptive and validation) are managed in real-time using a fallback strategy.
Initiation: The Stimulus PC presents a trial (e.g., Trial #N).
Response: The participant responds. The Stimulus PC writes the response data to the shared disk.
Computation Trigger: The Logic PC detects the new response file and begins computing the next adaptive trial (Trial #N+1). It has a deadline of approximately 2.9 seconds to complete this task.
Real-Time Continuity: The Stimulus PC does not wait. It proceeds with the fixed-time inter-trial interval (ITI).
The Scheduling Decision: When it's time to present Trial #N+1, the Stimulus PC checks the shared disk:
If the Logic PC has finished: A new adaptive trial file exists. The Stimulus PC reads it and presents the AEPsych-determined stimulus.
If the Logic PC has NOT finished (Fallback): No new adaptive trial file exists. To avoid dead time, the Stimulus PC instead pulls the next pre-generated MOCS validation trial from a randomized queue and presents it.
Loop: This process repeats. The fallback mechanism ensures the experiment continues smoothly, effectively interleaving adaptive and validation trials based on computational availability.
3.3. Phase 3: Post-Hoc Model Fitting and Validation
After all trials are completed, the collected data is separated and analyzed.
Data for Fitting: The AEPsych-driven trials and any fallback trials are used to fit the WPPM using Python and JAX.
Data for Validation: The held-out MOCS validation trials are used to assess the accuracy of the fitted WPPM, providing an unbiased measure of the model's performance.

Detailed File and Directory Structure
This document specifies the file and directory structure required to implement the dual-computer testing architecture. The structure is designed to be modular, scalable, and maintainable, clearly separating source code, configuration, real-time data exchange, and final results.
1. Project Root Directory
This is the main directory for the project, containing all source code, configurations, and analysis scripts. It would be version-controlled (e.g., with Git).
code
Code
/project_root/
│
├── architecture.md             # The high-level architecture document.
├── file_structure.md           # This document.
├── README.md                   # Project setup and usage instructions.
├── requirements.txt            # Python dependencies (numpy, jax, etc.).
│
├── config/                     # Configuration files, separating parameters from code.
│   ├── experiment_params.json  # Global settings: trial timings, ITI, screen resolution, etc.
│   ├── display_calibration.mat # Output from MATLAB calibration: gamma tables, transformation matrices.
│   └── participants/
│       ├── P01_config.json     # Participant-specific settings (e.g., ID, notes).
│       └── P01_validation_trials.csv # Pre-generated MOCS validation trials for participant P01.
│
├── data/                       # All generated data, organized by participant. NOT version-controlled.
│   └── P01/                    # Data for participant P01.
│       ├── raw/                # Raw, unprocessed data, one file per session.
│       │   ├── P01_S01_log.csv
│       │   └── P01_S02_log.csv
│       ├── processed/          # Cleaned, aggregated data ready for analysis.
│       │   └── P01_compiled_data.csv
│       ├── models/             # Saved model fits.
│       │   └── P01_wppm_fit.pkl
│       └── figures/            # Generated plots and visualizations.
│           ├── P01_threshold_contours.png
│           └── P01_validation_scatter.png
│
├── notebooks/                  # Jupyter notebooks for analysis and visualization.
│   ├── 01_data_aggregation.ipynb # Script to compile raw session logs into a processed file.
│   ├── 02_fit_wppm_model.ipynb   # Notebook to load processed data and fit the WPPM.
│   └── 03_validation_analysis.ipynb # Notebook for validation, regression analysis, and plotting.
│
├── scripts/                    # Helper and one-off scripts.
│   └── generate_validation_trials.py # Script to create the pre-generated MOCS and Fallback trials.
│
└── src/                        # All source code.
    ├── stimulus_presentation/  # Unity project for the Stimulus PC.
    │   ├── Assets/
    │   │   ├── Scenes/
    │   │   │   └── main_scene.unity
    │   │   └── Scripts/
    │   │       ├── TrialController.cs  # Manages trial sequence, reads/writes to shared drive.
    │   │       ├── StimulusManager.cs  # Updates blobby object colors based on RGB values.
    │   │       └── GammaCorrection.cs  # Applies the pre-computed gamma correction.
    │   └── ... (other Unity project files)
    │
    └── adaptive_engine/        # Python code for the Logic PC.
        ├── main_loop.py        # Main script that runs the experiment logic.
        ├── aepsych_wrapper.py  # Module to interface with the AEPsych library.
        ├── wppm_fitter.py      # Module for post-hoc WPPM fitting.
        └── utils/
            ├── file_io.py      # Functions for reading/writing the communication JSON files.
            └── transformations.py # Color space transformation functions.
2. Shared Network Drive (Real-time Communication)
This directory is the critical, high-traffic communication hub between the two computers during an experiment. Its structure is ephemeral and session-based, designed for clarity and to prevent data collisions.
code
Code
/shared_drive/
│
└── P01/                        # Top-level folder for the current participant.
    └── S01/                    # Folder for the current session (e.g., Session 01).
        │
        ├── to_stimulus_pc/     # Directory for commands sent TO the Stimulus PC.
        │   │
        │   └── next_trial.json # The next trial to be displayed.
        │                       # (Created by Logic PC; Deleted by Stimulus PC after reading).
        │
        ├── from_stimulus_pc/   # Directory for data sent FROM the Stimulus PC.
        │   │
        │   └── response_{timestamp}.json # Contains the result of the last trial.
        │                                 # (Created by Stimulus PC; Deleted by Logic PC after reading).
        │
        └── SESSION_STATUS.txt  # A simple flag file to control the state.
                                # Content: "RUNNING", "PAUSED", "COMPLETED".
2.1. Contents of Communication Files
next_trial.json
This file contains all the information the Stimulus PC needs to render the next trial.
code
JSON
{
  "participant_id": "P01",
  "session_index": 1,
  "trial_index": 152,
  "trial_type": "ADAPTIVE", // or "VALIDATION", or "FALLBACK"
  "stimuli": [
    {"type": "reference",   "rgb": [0.51, 0.48, 0.49]},
    {"type": "reference",   "rgb": [0.51, 0.48, 0.49]},
    {"type": "comparison",  "rgb": [0.53, 0.47, 0.50]}
  ]
}
response_{timestamp}.json
This file contains the participant's response and is the trigger for the Logic PC to compute the next trial. The timestamp ensures a unique filename.
code
JSON
{
  "participant_id": "P01",
  "session_index": 1,
  "trial_index": 151,
  "trial_type_shown": "VALIDATION",
  "response_correct": true,
  "response_time_ms": 754,
  "stimuli_shown": {
    "reference_rgb": [0.50, 0.50, 0.50],
    "comparison_rgb": [0.50, 0.55, 0.50]
  }
}
3. Data Log File Structure (P01_S01_log.csv)
The raw data from each session is appended to a master CSV log file for that session. This creates a permanent, easily parsable record.
Columns:
timestamp, participant_id, session_index, trial_index, trial_type, ref_r, ref_g, ref_b, comp_r, comp_g, comp_b, response_correct, response_time_ms