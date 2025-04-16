# Meta Dataset for Fall Detection - MD4FD

This tool enables the creation of the Meta Dataset for Fall Detection (MD4FD) by extracting and preprocessing data from multiple public datasets, including Le2i, UR Fall, FALL-UP, and High-Quality Fall Simulation. These datasets were selected for their wide use and representativeness in the context of fall detection. The main goal is to standardize and enrich the available data by converting video sequences into normalized sequences of 2D body landmarks, annotated as either Fall (1) or Activity of Daily Living (0).

The tool allows to: 
 - Landmark Extraction: uses MediaPipe to extract 33 body landmarks from video frames. From these, only 11 key joints are retained: front face, shoulders, wrists, hips, knees, and ankles.
 - Normalization: landmark coordinates are normalized by image width and height to ensure consistency across different video resolutions.
 - Body Aspect Ratio (BAR): a custom feature that defines the overall orientation of the body is computed. It is the maximum body occupancy in the vertical direction over the body width in the horizontal direction.
 - Sequence Creation: video frames are extracted. The user can define settings for the extraction by choosing the number of frames to be concatenated into a single sequence and the overlapping factor to define from which frames the next sequence should start.
 - Annotation Interface: a GUI is provided to manually label each sequence as Fall or ADL.
 - Data Output: final data is saved in a structured JSON format which includes 22 normalized joint coordinates, 1 BAR value and 1 label (Fall or ADL).

This approach ensures data anonymization, increases dataset variability and facilitates the training of fall detection models that rely on the temporal dynamics of human motion.

# Repository Structure
The repository is organized as follows:
- Four folders, each named after one of the datasets being considered. In detail each folder contains:
   - The code used to extract the subsequences.
   - The corresponding extracted JSON file with the annotated subsequences.
- A folder named dataset_tool which contains utility modules for dataset processing:
   - landmark_extractor: for extracting body landmarks from video sequences.
   - dataset_creator: for generating annotated subsequences from raw data.
   - dataset_merger: for combining multiple datasets into a single unified JSON file.
   - dataset_normalizer: for applying final normalization across all samples.

# Cite this dataset

A. Longo, A. Bono, G. Guaragnella, T. D’Orazio, "A Transformer Based Architecture and a Meta Dataset for Fall Detection", *Submitted to Computer Methods and Programs in Biomedicine*, 2025.
