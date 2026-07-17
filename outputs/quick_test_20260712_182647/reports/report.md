# Audio QA Robustness Evaluation Report

## Experiment Overview

- **Experiment**: quick_test
- **Description**: Quick test: mock model, baseline+label_only, 5 samples from local fixture
- **Timestamp**: 2026-07-12T10:27:06.439584+00:00
- **Device**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Total wall time**: 18.6s
- **Models**: mock_model
- **Benchmarks**: mmau
- **Perturbations**: baseline, label_only

## Timing Breakdown

| Phase | Duration (s) | Duration (min) |
| --- | --- | --- |
| Total | 18.6 | 0.3 |
| Model loading | 0.0 | 0.0 |
| Benchmark loading | 7.0 | 0.1 |
| Perturbation | 0.0 | 0.0 |
| Inference | 0.0 | 0.0 |
| Overhead (I/O, logging) | 11.6 | 0.2 |

### Per-Task Timing

| Task | Duration (s) | Samples/s | Accuracy |
| --- | --- | --- | --- |
| mock_model__mmau__baseline | 11.4 | 87.58 | 0.3770 |
| mock_model__mmau__label_only | 0.1 | 18783.44 | 0.3770 |

## Overall Accuracy

| model | benchmark | perturbation | accuracy | total | correct | errors | delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mock_model | mmau | baseline | 0.377 | 1000 | 377 | 0 | 0.0 |
| mock_model | mmau | label_only | 0.377 | 1000 | 377 | 0 | 0.0 |

### Per-Task Throughput

| model | benchmark | perturbation | duration_s | samples/s |
| --- | --- | --- | --- | --- |
| mock_model | mmau | baseline | 11.419 | 87.577 |
| mock_model | mmau | label_only | 0.053 | 18783.437 |

## Robustness Gap (Accuracy drop from Baseline)

| model | benchmark | label_only |
| --- | --- | --- |
| mock_model | mmau | 0.0 |

## Per-Category Breakdown

### mock_model / mmau / baseline

| category_field | category_value | accuracy | count | correct |
| --- | --- | --- | --- | --- |
| category | Information Extraction | 0.3503 | 294 | 103 |
| category | Reasoning | 0.3881 | 706 | 274 |
| dataset | AudioSet | 0.6296 | 216 | 136 |
| dataset | Clotho | 0.2083 | 48 | 10 |
| dataset | fma_large | 0.5 | 6 | 3 |
| dataset | guitarset | 0.6327 | 49 | 31 |
| dataset | iemocap | 0.6486 | 37 | 24 |
| dataset | jamendo | 0.0 | 3 | 0 |
| dataset | meld | 0.2093 | 43 | 9 |
| dataset | musicbench | 0.2983 | 181 | 54 |
| dataset | musiccaps | 0.25 | 48 | 12 |
| dataset | musidb | 0.7143 | 7 | 5 |
| dataset | mustard | 0.3 | 20 | 6 |
| dataset | sdd | 0.3 | 40 | 12 |
| dataset | synthetic | 0.2617 | 149 | 39 |
| dataset | voxceleb | 0.2353 | 153 | 36 |
| difficulty | easy | 0.2791 | 258 | 72 |
| difficulty | hard | 0.2716 | 232 | 63 |
| difficulty | medium | 0.4745 | 510 | 242 |
| sub_category | Acoustic Scene Reasoning | 0.2083 | 48 | 10 |
| sub_category | Acoustic Source Inference | 0.8125 | 48 | 39 |
| sub_category | Ambient Sound Interpretation | 0.2292 | 48 | 11 |
| sub_category | Conversational Fact Retrieval | 0.65 | 20 | 13 |
| sub_category | Counting | 0.2 | 20 | 4 |
| sub_category | Dissonant Emotion Interpretation | 0.3 | 20 | 6 |
| sub_category | Eco-Acoustic Knowledge | 0.8936 | 47 | 42 |
| sub_category | Emotion Flip Detection | 0.15 | 20 | 3 |
| sub_category | Emotion State summarisation | 0.2 | 20 | 4 |
| sub_category | Emotional Tone Interpretation | 0.1818 | 33 | 6 |
| sub_category | Event-Based Knowledge Retrieval | 0.25 | 20 | 5 |
| sub_category | Event-Based Sound Reasoning | 0.8542 | 48 | 41 |
| sub_category | Harmony and Chord Progressions | 0.4545 | 33 | 15 |
| sub_category | Instrumentation | 0.2286 | 35 | 8 |
| sub_category | Key highlight Extraction | 0.6 | 20 | 12 |
| sub_category | Lyrical Reasoning | 0.5 | 10 | 5 |
| sub_category | Melodic Structure Interpretation | 0.3939 | 33 | 13 |
| sub_category | Multi Speaker Role Mapping | 0.35 | 20 | 7 |
| sub_category | Musical Genre Reasoning | 0.3235 | 34 | 11 |
| sub_category | Musical Texture Interpretation | 0.1471 | 34 | 5 |
| sub_category | Phonemic Stress Pattern Analysis | 0.2353 | 153 | 36 |
| sub_category | Phonological Sequence Decoding | 0.3 | 20 | 6 |
| sub_category | Rhythm and Tempo Understanding | 0.2826 | 46 | 13 |
| sub_category | Socio-cultural Interpretation | 0.35 | 20 | 7 |
| sub_category | Sound-Based Event Recognition | 0.0217 | 46 | 1 |
| sub_category | Temporal Event Reasoning | 0.4167 | 48 | 20 |
| sub_category | Temporal Reasoning | 0.6071 | 56 | 34 |
| task | music | 0.3503 | 334 | 117 |
| task | sound | 0.4925 | 333 | 164 |
| task | speech | 0.2883 | 333 | 96 |
### mock_model / mmau / label_only

| category_field | category_value | accuracy | count | correct |
| --- | --- | --- | --- | --- |
| category | Information Extraction | 0.3503 | 294 | 103 |
| category | Reasoning | 0.3881 | 706 | 274 |
| dataset | AudioSet | 0.6296 | 216 | 136 |
| dataset | Clotho | 0.2083 | 48 | 10 |
| dataset | fma_large | 0.5 | 6 | 3 |
| dataset | guitarset | 0.6327 | 49 | 31 |
| dataset | iemocap | 0.6486 | 37 | 24 |
| dataset | jamendo | 0.0 | 3 | 0 |
| dataset | meld | 0.2093 | 43 | 9 |
| dataset | musicbench | 0.2983 | 181 | 54 |
| dataset | musiccaps | 0.25 | 48 | 12 |
| dataset | musidb | 0.7143 | 7 | 5 |
| dataset | mustard | 0.3 | 20 | 6 |
| dataset | sdd | 0.3 | 40 | 12 |
| dataset | synthetic | 0.2617 | 149 | 39 |
| dataset | voxceleb | 0.2353 | 153 | 36 |
| difficulty | easy | 0.2791 | 258 | 72 |
| difficulty | hard | 0.2716 | 232 | 63 |
| difficulty | medium | 0.4745 | 510 | 242 |
| sub_category | Acoustic Scene Reasoning | 0.2083 | 48 | 10 |
| sub_category | Acoustic Source Inference | 0.8125 | 48 | 39 |
| sub_category | Ambient Sound Interpretation | 0.2292 | 48 | 11 |
| sub_category | Conversational Fact Retrieval | 0.65 | 20 | 13 |
| sub_category | Counting | 0.2 | 20 | 4 |
| sub_category | Dissonant Emotion Interpretation | 0.3 | 20 | 6 |
| sub_category | Eco-Acoustic Knowledge | 0.8936 | 47 | 42 |
| sub_category | Emotion Flip Detection | 0.15 | 20 | 3 |
| sub_category | Emotion State summarisation | 0.2 | 20 | 4 |
| sub_category | Emotional Tone Interpretation | 0.1818 | 33 | 6 |
| sub_category | Event-Based Knowledge Retrieval | 0.25 | 20 | 5 |
| sub_category | Event-Based Sound Reasoning | 0.8542 | 48 | 41 |
| sub_category | Harmony and Chord Progressions | 0.4545 | 33 | 15 |
| sub_category | Instrumentation | 0.2286 | 35 | 8 |
| sub_category | Key highlight Extraction | 0.6 | 20 | 12 |
| sub_category | Lyrical Reasoning | 0.5 | 10 | 5 |
| sub_category | Melodic Structure Interpretation | 0.3939 | 33 | 13 |
| sub_category | Multi Speaker Role Mapping | 0.35 | 20 | 7 |
| sub_category | Musical Genre Reasoning | 0.3235 | 34 | 11 |
| sub_category | Musical Texture Interpretation | 0.1471 | 34 | 5 |
| sub_category | Phonemic Stress Pattern Analysis | 0.2353 | 153 | 36 |
| sub_category | Phonological Sequence Decoding | 0.3 | 20 | 6 |
| sub_category | Rhythm and Tempo Understanding | 0.2826 | 46 | 13 |
| sub_category | Socio-cultural Interpretation | 0.35 | 20 | 7 |
| sub_category | Sound-Based Event Recognition | 0.0217 | 46 | 1 |
| sub_category | Temporal Event Reasoning | 0.4167 | 48 | 20 |
| sub_category | Temporal Reasoning | 0.6071 | 56 | 34 |
| task | music | 0.3503 | 334 | 117 |
| task | sound | 0.4925 | 333 | 164 |
| task | speech | 0.2883 | 333 | 96 |
