# Objective-Driven Calibrated Recommendations - Supplementary Material

## Main Text _with Appendix_ (PDF) is [HERE](https://github.com/cali-anon/OCR/raw/main/ocr_paper.pdf)
The PDF contains the **main text** and a detailed **appendix** with **additional setting details and additional experiment results**.

- **Main Text (start)** — [HERE](https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2Fcali-anon%2FOCR%2Fmain%2Focr_paper.pdf?v=20250815#page=1)
- **Appendix A1: Additional Setting Details** — [HERE](https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2Fcali-anon%2FOCR%2Fmain%2Focr_paper.pdf?v=20250815#page=10)
- **Appendix A2: Additional Experiment Results** — [HERE](https://mozilla.github.io/pdf.js/web/viewer.html?file=https%3A%2F%2Fraw.githubusercontent.com%2Fcali-anon%2FOCR%2Fmain%2Focr_paper.pdf?v=20250815#page=10)

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
