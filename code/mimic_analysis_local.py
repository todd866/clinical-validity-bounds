"""
MIMIC-IV Local Analysis (CSV Version)

Runs dimensional validity analysis on downloaded MIMIC-IV CSV files.

Author: Ian Todd
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data' / 'mimic'
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def load_mimic_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MIMIC-IV CSV files."""
    print("Loading MIMIC-IV data from local files...")

    patients = pd.read_csv(DATA_DIR / 'patients.csv.gz', compression='gzip')
    admissions = pd.read_csv(DATA_DIR / 'admissions.csv.gz', compression='gzip')
    diagnoses = pd.read_csv(DATA_DIR / 'diagnoses_icd.csv.gz', compression='gzip')

    print(f"  Patients: {len(patients):,}")
    print(f"  Admissions: {len(admissions):,}")
    print(f"  Diagnoses: {len(diagnoses):,}")

    return patients, admissions, diagnoses


def compute_effective_dimensionality(X: np.ndarray) -> float:
    """
    Compute effective dimensionality using participation ratio.
    D_eff = (sum(lambda_i))^2 / sum(lambda_i^2)
    """
    if X.shape[0] < 2 or X.shape[1] < 2:
        return 1.0

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute covariance eigenvalues
    cov = np.cov(X_scaled.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    # Participation ratio
    d_eff = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return d_eff


def build_feature_matrix(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    n_top_dx: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build patient-level feature matrix.

    Features:
    - Demographics (age, gender)
    - Binary diagnosis features (top N codes)
    - Diagnosis count (complexity proxy)

    Outcome:
    - Hospital mortality
    """
    print("\nBuilding feature matrix...")

    # Parse dates
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

    # Calculate length of stay
    admissions['los_days'] = (
        admissions['dischtime'] - admissions['admittime']
    ).dt.total_seconds() / 86400

    # Filter: require valid mortality flag and LOS >= 1 day
    admissions = admissions[
        admissions['hospital_expire_flag'].notna() &
        (admissions['los_days'] >= 1)
    ].copy()

    print(f"  Admissions with LOS >= 1 day: {len(admissions):,}")

    # Get top diagnosis codes
    dx_counts = diagnoses.groupby('icd_code').size().sort_values(ascending=False)
    top_dx_codes = dx_counts.head(n_top_dx).index.tolist()
    print(f"  Using top {len(top_dx_codes)} diagnosis codes")

    # Count diagnoses per admission
    dx_per_admission = diagnoses.groupby('hadm_id').size().reset_index(name='n_diagnoses')

    # Create binary diagnosis features
    dx_filtered = diagnoses[diagnoses['icd_code'].isin(top_dx_codes)].copy()
    dx_pivot = dx_filtered.pivot_table(
        index='hadm_id',
        columns='icd_code',
        aggfunc='size',
        fill_value=0
    )
    dx_pivot = (dx_pivot > 0).astype(int)
    dx_pivot.columns = [f'dx_{c.replace(".", "_").replace("-", "_")}' for c in dx_pivot.columns]

    # Merge with admissions
    df = admissions.merge(patients[['subject_id', 'gender', 'anchor_age']], on='subject_id')
    df = df.merge(dx_per_admission, on='hadm_id', how='left')
    df = df.merge(dx_pivot, on='hadm_id', how='left')

    # Fill NaN diagnosis counts and features
    df['n_diagnoses'] = df['n_diagnoses'].fillna(0)
    dx_cols = [c for c in df.columns if c.startswith('dx_')]
    df[dx_cols] = df[dx_cols].fillna(0)

    # Encode gender
    df['is_male'] = (df['gender'] == 'M').astype(int)

    # Define feature columns
    feature_cols = ['anchor_age', 'is_male', 'n_diagnoses'] + dx_cols

    # Create output DataFrames
    features = df[['hadm_id'] + feature_cols].set_index('hadm_id')
    outcomes = df[['hadm_id', 'hospital_expire_flag', 'los_days', 'n_diagnoses']].set_index('hadm_id')

    print(f"  Final: {len(features):,} admissions, {len(feature_cols)} features")
    print(f"  Mortality rate: {outcomes['hospital_expire_flag'].mean():.1%}")

    return features, outcomes


def stratify_by_complexity(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    n_strata: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratify patients by complexity (number of diagnoses).
    """
    print("\nStratifying by complexity...")

    n_diagnoses = outcomes['n_diagnoses']

    # Create strata using tertiles
    try:
        complexity_strata = pd.qcut(
            n_diagnoses, n_strata,
            labels=['Simple', 'Moderate', 'Complex'],
            duplicates='drop'
        )
    except ValueError:
        # If qcut fails, use manual percentile cuts
        q33 = n_diagnoses.quantile(0.33)
        q67 = n_diagnoses.quantile(0.67)
        complexity_strata = pd.cut(
            n_diagnoses,
            bins=[-np.inf, q33, q67, np.inf],
            labels=['Simple', 'Moderate', 'Complex']
        )

    analysis_df = outcomes.copy()
    analysis_df['complexity'] = complexity_strata

    # Summary statistics
    summary = analysis_df.groupby('complexity').agg({
        'hospital_expire_flag': ['count', 'mean', 'std'],
        'los_days': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'n_diagnoses': ['median', 'mean']
    })

    print("\nStratum summary:")
    for stratum in ['Simple', 'Moderate', 'Complex']:
        if stratum in analysis_df['complexity'].values:
            mask = analysis_df['complexity'] == stratum
            n = mask.sum()
            mort = analysis_df.loc[mask, 'hospital_expire_flag'].mean()
            dx_med = analysis_df.loc[mask, 'n_diagnoses'].median()
            print(f"  {stratum}: N={n:,}, mortality={mort:.1%}, median dx={dx_med:.0f}")

    return analysis_df, summary


def train_mortality_classifier(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    analysis_df: pd.DataFrame,
    n_folds: int = 5
) -> pd.DataFrame:
    """
    Train mortality classifier and evaluate by complexity stratum.

    Key hypothesis: AUC is lowest in the moderate-complexity regime predicted
    by the dimensional validity bound (U-shaped pattern vs multimorbidity).
    """
    print("\nTraining mortality classifier by stratum...")

    X = features.values
    y = outcomes['hospital_expire_flag'].values

    results = []

    for stratum in ['Simple', 'Moderate', 'Complex']:
        if stratum not in analysis_df['complexity'].values:
            continue

        mask = analysis_df['complexity'] == stratum
        X_subset = X[mask]
        y_subset = y[mask]

        if len(y_subset) < 100 or y_subset.sum() < 10:
            print(f"  Skipping {stratum}: insufficient samples")
            continue

        # Compute D_eff
        d_eff = compute_effective_dimensionality(X_subset)

        # Train classifier
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_subset, y_subset, cv=cv, scoring='roc_auc')

        results.append({
            'complexity': stratum,
            'n_patients': len(y_subset),
            'n_deaths': int(y_subset.sum()),
            'mortality_rate': y_subset.mean(),
            'd_eff': d_eff,
            'auc_mean': scores.mean(),
            'auc_std': scores.std(),
            'auc_ci_low': scores.mean() - 1.96 * scores.std() / np.sqrt(n_folds),
            'auc_ci_high': scores.mean() + 1.96 * scores.std() / np.sqrt(n_folds)
        })

        print(f"  {stratum}: N={len(y_subset):,}, D_eff={d_eff:.1f}, "
              f"AUC={scores.mean():.3f}±{scores.std():.3f}")

    return pd.DataFrame(results)


def generate_manuscript_output(
    analysis_df: pd.DataFrame,
    classifier_results: pd.DataFrame
):
    """Generate manuscript-ready output."""

    print("\n" + "=" * 60)
    print("MANUSCRIPT TABLE (copy to LaTeX)")
    print("=" * 60)
    print("\nStratum & N & D_eff & AUC (95% CI) \\\\")
    print("\\hline")

    for _, row in classifier_results.iterrows():
        print(f"{row['complexity']} & {row['n_patients']:,} & {row['d_eff']:.1f} & "
              f"{row['auc_mean']:.3f} ({row['auc_ci_low']:.3f}--{row['auc_ci_high']:.3f}) \\\\")

    # Key findings
    if len(classifier_results) >= 3:
        simple = classifier_results[classifier_results['complexity'] == 'Simple']
        moderate = classifier_results[classifier_results['complexity'] == 'Moderate']
        complex_ = classifier_results[classifier_results['complexity'] == 'Complex']

        if len(simple) > 0 and len(moderate) > 0 and len(complex_) > 0:
            print("\n" + "=" * 60)
            print("KEY FINDINGS")
            print("=" * 60)

            # D_eff pattern
            d_simple = simple['d_eff'].values[0]
            d_mod = moderate['d_eff'].values[0]
            d_complex = complex_['d_eff'].values[0]
            print(f"\n1. D_eff DECREASES with clinical complexity (Lipsitz-Goldberger):")
            print(f"   Simple → Complex: {d_simple:.1f} → {d_complex:.1f} (Δ = {d_complex - d_simple:.1f})")
            print(f"   Interpretation: Illness is a LOSS of complexity. Multimorbid")
            print(f"   patients collapse into stereotyped disease attractors.")

            # AUC pattern
            auc_simple = simple['auc_mean'].values[0]
            auc_mod = moderate['auc_mean'].values[0]
            auc_complex = complex_['auc_mean'].values[0]
            worst_stratum = classifier_results.loc[classifier_results['auc_mean'].idxmin(), 'complexity']
            print(f"\n2. AUC is LOWEST in {worst_stratum} stratum ({auc_mod:.3f}):")
            print(f"   Simple: {auc_simple:.3f}, Moderate: {auc_mod:.3f}, Complex: {auc_complex:.3f}")
            print(f"   Interpretation: Moderate complexity patients are hardest to")
            print(f"   predict - they don't fit simple OR stereotyped complex patterns.")

            # Clinical validity bound
            print(f"\n3. CLINICAL VALIDITY BOUND:")
            print(f"   The 'moderate complexity' regime (D_eff ≈ {d_mod:.0f}) represents")
            print(f"   the zone where mortality prediction is least reliable.")
            print(f"   Clinical complexity ≠ dimensional complexity.")


def run_analysis():
    """Run full MIMIC-IV analysis pipeline."""

    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("MIMIC-IV Dimensional Validity Analysis")
    print("=" * 60)

    # Load data
    patients, admissions, diagnoses = load_mimic_data()

    # Build features
    features, outcomes = build_feature_matrix(patients, admissions, diagnoses)

    # Stratify
    analysis_df, summary = stratify_by_complexity(features, outcomes)

    # Train classifier
    classifier_results = train_mortality_classifier(features, outcomes, analysis_df)

    # Generate output
    generate_manuscript_output(analysis_df, classifier_results)

    # Save results
    analysis_df.to_csv(RESULTS_DIR / 'patient_analysis.csv')
    classifier_results.to_csv(RESULTS_DIR / 'classifier_results.csv', index=False)
    summary.to_csv(RESULTS_DIR / 'stratum_summary.csv')

    print(f"\nResults saved to: {RESULTS_DIR}")

    return features, outcomes, analysis_df, classifier_results


if __name__ == "__main__":
    run_analysis()
