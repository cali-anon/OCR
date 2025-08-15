# Objective-Driven Calibrated Recommendations - Supplementary Meterial

## Main Text _with Appendix_ (PDF) is [HERE]()
The PDF contains the **main text** and a detailed **appendix** with **proofs, additional experiments, and limitations**.

- **Main Text (start)** — [HERE]()
- **Limitations** — [HERE]()
- **Proofs** — [HERE]()
- **Additional Experiments** — [HERE]()

**## How to Run the Code

To run the experiments, you need to have Python installed with the required packages. You can install the dependencies using pip:

```bash
pip install numpy matplotlib pandas scikit-learn tqdm seaborn torch
```

### Synthetic Experiments

The scripts for the synthetic experiments are located in the `src` directory.

  * To run the experiment with varying training data sizes, execute:
    ```bash
    python src/main_train_data.py
    ```
  * To run the experiment with varying lambda values, execute:
    ```bash
    python src/main_lambda.py
    ```
  * To run the experiment with varying numbers of categories, execute:
    ```bash
    python src/main_category.py
    ```

### Real-world Experiments (KuaiSim)

The scripts for the real-world experiments using the KuaiSim simulator are in the `real_kuaisim` directory.

  * To run the main experiment with varying training data sizes, execute:
    ```bash
    python real_kuaisim/real_kuaisim5.py
    ```
  * To run the experiment evaluating the effect of feedback influence, execute:
    ```bash
    python real_kuaisim/real_kuaisim5_val_feedback_influence.py
    ```
  * To run the experiment evaluating the effect of the tanh threshold, execute:
    ```bash
    python real_kuaisim/real_kuaisim5_val_threshold.py
    ```**# OCR

## How to Run the Code

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Code

The project contains several main scripts for different experiments:

#### Synthetic Data Experiments (`src/` directory)
- **`main_category.py`**: Experiments with varying category numbers
- **`main_lambda.py`**: Experiments with different lambda values
- **`main_train_data.py`**: Experiments with varying training data sizes

Run any of these synthetic experiments scripts from the `src/` directory:
```bash
python main_category.py
python main_lambda.py
python main_train_data.py
```

#### Real Data Experiments (`real_kuaisim/` directory)
- **`real_kuaisim5.py`**: Main real data experiment
- **`real_kuaisim5_val_feedback_influence.py`**: Validation with feedback influence
- **`real_kuaisim5_val_threshold.py`**: Validation with threshold

Run from the `real_kuaisim/` directory:
```bash
python real_kuaisim5.py
```
