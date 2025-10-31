# EigenDrug Project - SNU Creative Integrated Design 1 (Fall 2025)

This project is part of the **Creative Integrated Design 1** course (Course Code: M1522.000200 001) in the **Department of Computer Science and Engineering, College of Engineering, Seoul National University** during the **Fall semester of 2025**.

The project is conducted in collaboration with **EigenDrug Inc.**, as a real-world industry assignment.

## Author

- **Name:** Jaejoon Kim  
- **Email:** [jjkim030309@gmail.com](mailto:jjkim030309@gmail.com)  
- Please contact via email for any inquiries or issues related to the project.

## Dependencies

- astral uv (https://docs.astral.sh/uv)
- AbBiBench (https://github.com/MSBMI-SAFE/AbBiBench)
- hmmer==3.4 (https://github.com/EddyRivasLab/hmmer)
- ANARCI (https://github.com/oxpig/ANARCI)

## installation

```
git clone https://github.com/J-joon/abbibench_toolkit && cd abbibench_toolkit
uv sync
```

## directory structure

```
./abbibench_toolkit
|--./data # dataset
|--./outputs # output results
```

## Usage
After running get_model_log_likelihood.py for a model, please make sure the resulting output csv file be under ./abbibench_toolkit/outputs

```
uv run compute_correlation —model-name diffab —dataset-name aayl49_ML
# example output: rho: 0.0012, p-value: 9.1073e-01
```


## Download Dataset

```
git lfs install
git clone https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset data
```
