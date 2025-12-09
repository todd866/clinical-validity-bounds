"""
Simulations for: The Dimensional Validity Bound

Simulations:
1. Diagnostic cascade - false positive accumulation
2. Dimensional matching - observer-patient coupling stability
3. Classifier degradation - U-shaped AUC vs multimorbidity burden
4. Treatment heterogeneity - outcome variance within diagnostic categories

Author: Ian Todd
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PUBLICATION FIGURE SETTINGS
# =============================================================================

# Publication style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Color palette (Nature-style, matched across all figures)
COLORS = {
    'primary': '#2166AC',      # Blue
    'secondary': '#B2182B',    # Red
    'tertiary': '#4DAF4A',     # Green
    'quaternary': '#FF7F00',   # Orange
    'gray': '#666666',
    'light_gray': '#E8E8E8',
}

# =============================================================================
# 1. DIAGNOSTIC CASCADE SIMULATION
# =============================================================================

def diagnostic_cascade_probability(n_tests: int, fpr: float = 0.05) -> float:
    """
    Calculate probability of at least one false positive given n independent tests.

    P(at least 1 FP) = 1 - (1 - fpr)^n_tests
    """
    return 1 - (1 - fpr) ** n_tests


def simulate_diagnostic_cascade(
    n_patients: int = 10000,
    n_tests_range: Tuple[int, int] = (1, 50),
    fpr: float = 0.05,
    cascade_threshold: int = 1
) -> pd.DataFrame:
    """
    Simulate diagnostic cascades across patients with varying test counts.

    Returns DataFrame with:
    - n_tests: number of tests ordered
    - p_cascade_theoretical: theoretical probability
    - p_cascade_simulated: simulated probability
    - mean_false_positives: average FP count
    """
    results = []

    for n_tests in range(n_tests_range[0], n_tests_range[1] + 1):
        # Theoretical probability
        p_theoretical = diagnostic_cascade_probability(n_tests, fpr)

        # Simulate
        false_positives = np.random.binomial(n_tests, fpr, n_patients)
        p_simulated = np.mean(false_positives >= cascade_threshold)
        mean_fp = np.mean(false_positives)

        results.append({
            'n_tests': n_tests,
            'p_cascade_theoretical': p_theoretical,
            'p_cascade_simulated': p_simulated,
            'mean_false_positives': mean_fp
        })

    return pd.DataFrame(results)


def plot_diagnostic_cascade(df: pd.DataFrame, save_path: str = None):
    """Plot diagnostic cascade probability vs number of tests."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Probability of cascade
    ax1.plot(df['n_tests'], df['p_cascade_theoretical'], '-', color=COLORS['primary'],
             lw=2, label='Theoretical')
    ax1.plot(df['n_tests'], df['p_cascade_simulated'], '--', color=COLORS['secondary'],
             lw=2, label='Simulated')
    ax1.axhline(y=0.5, color=COLORS['gray'], linestyle=':', alpha=0.7)
    ax1.axhline(y=0.9, color=COLORS['gray'], linestyle=':', alpha=0.7)
    ax1.set_xlabel('Number of Tests Ordered')
    ax1.set_ylabel('P(≥1 False Positive)')
    ax1.set_title('Diagnostic Cascade Probability')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Right: Expected false positives
    ax2.plot(df['n_tests'], df['mean_false_positives'], '-', color=COLORS['tertiary'], lw=2)
    ax2.fill_between(df['n_tests'], 0, df['mean_false_positives'],
                     color=COLORS['tertiary'], alpha=0.3)
    ax2.set_xlabel('Number of Tests Ordered')
    ax2.set_ylabel('Expected False Positives')
    ax2.set_title('Expected Spurious Findings')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# =============================================================================
# 2. DIMENSIONAL MATCHING SIMULATION (Kuramoto-inspired)
# =============================================================================

def compute_effective_dimensionality(X: np.ndarray) -> float:
    """
    Compute effective dimensionality using participation ratio.
    D_eff = (sum(lambda_i))^2 / sum(lambda_i^2)
    """
    if X.shape[0] < 2:
        return 1.0

    # Standardize
    X_centered = X - X.mean(axis=0)

    # Compute covariance eigenvalues
    cov = np.cov(X_centered.T)
    if cov.ndim == 0:
        return 1.0

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros

    if len(eigenvalues) == 0:
        return 1.0

    # Participation ratio
    d_eff = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return d_eff


def simulate_patient_dynamics(
    n_dims: int = 100,
    n_timesteps: int = 500,
    coupling_strength: float = 0.1,
    noise_level: float = 0.5
) -> np.ndarray:
    """
    Simulate high-dimensional patient dynamics using coupled oscillators.

    Returns: (n_timesteps, n_dims) array of patient state trajectory
    """
    # Natural frequencies (heterogeneous)
    omega = np.random.normal(1.0, 0.3, n_dims)

    # Initial phases
    theta = np.random.uniform(0, 2*np.pi, n_dims)

    trajectory = np.zeros((n_timesteps, n_dims))
    dt = 0.1

    for t in range(n_timesteps):
        # Kuramoto-like coupling
        mean_sin = np.mean(np.sin(theta))
        mean_cos = np.mean(np.cos(theta))

        # Update each oscillator
        for i in range(n_dims):
            coupling = coupling_strength * (mean_sin * np.cos(theta[i]) -
                                           mean_cos * np.sin(theta[i]))
            noise = noise_level * np.random.randn()
            theta[i] += dt * (omega[i] + coupling + noise)

        # Store observable state (could be sin(theta) or other projection)
        trajectory[t] = np.sin(theta)

    return trajectory


def simulate_observer_coupling(
    patient_trajectory: np.ndarray,
    observer_dims: int,
    coupling_strength: float = 0.5
) -> Tuple[np.ndarray, float]:
    """
    Simulate an observer (doctor/algorithm) coupling to patient dynamics.

    Key physical insight: The observer must track the patient's state in real-time.
    With fewer dimensions, the observer's "view" of the patient becomes increasingly
    blurry - like trying to diagnose from a pixelated image.

    We measure "sync_error" as the proportion of patient state variance
    that the observer CANNOT capture. This directly corresponds to
    diagnostic information loss.

    Returns:
    - observer_estimate: observer's best estimate of patient state
    - sync_error: fraction of variance unexplained (0=perfect, 1=total loss)
    """
    n_timesteps, patient_dims = patient_trajectory.shape
    observed_dims = min(observer_dims, patient_dims)

    # Use PCA to find the optimal low-dimensional projection
    from sklearn.decomposition import PCA

    pca = PCA(n_components=observed_dims)
    projected = pca.fit_transform(patient_trajectory)

    # The key metric: what fraction of variance is captured?
    # This is exactly pca.explained_variance_ratio_.sum()
    variance_explained = pca.explained_variance_ratio_.sum()

    # Sync error = variance NOT explained
    sync_error = 1.0 - variance_explained

    # Observer's reconstruction (for visualization)
    observer_estimate = pca.inverse_transform(projected)

    return observer_estimate, sync_error


def dimensional_matching_experiment(
    patient_dims: int = 100,
    observer_dims_list: List[int] = [5, 10, 20, 30, 50, 70, 100],
    n_trials: int = 20
) -> pd.DataFrame:
    """
    Test how observer dimensionality affects diagnostic accuracy.

    Key result: Below r ≈ 0.3, the observer cannot reliably distinguish
    between different patient states (diagnostic categories).
    """
    results = []

    for trial in range(n_trials):
        # Generate patient trajectory with consistent parameters
        # Use trial as seed for reproducibility
        np.random.seed(trial * 100)

        patient_traj = simulate_patient_dynamics(
            n_dims=patient_dims,
            n_timesteps=500,
            coupling_strength=0.2,  # Moderate coupling creates structure
            noise_level=0.4  # Some noise makes it realistic
        )
        patient_deff = compute_effective_dimensionality(patient_traj)

        for obs_dims in observer_dims_list:
            _, sync_error = simulate_observer_coupling(patient_traj, obs_dims)

            # Stability = 1 - error (for ARI-based metric)
            # sync_error is already 1 - ARI, so stability = ARI
            stability = 1.0 - sync_error

            results.append({
                'trial': trial,
                'patient_dims': patient_dims,
                'patient_deff': patient_deff,
                'observer_dims': obs_dims,
                'dim_ratio': obs_dims / patient_dims,
                'sync_error': sync_error,
                'stability': stability
            })

    return pd.DataFrame(results)


def plot_dimensional_matching(df: pd.DataFrame, save_path: str = None):
    """Plot dimensional matching results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Aggregate by dim_ratio
    agg = df.groupby('dim_ratio').agg({
        'stability': ['mean', 'std'],
        'sync_error': ['mean', 'std']
    }).reset_index()
    agg.columns = ['dim_ratio', 'stability_mean', 'stability_std',
                   'error_mean', 'error_std']

    # Left: Stability vs dimensional ratio
    ax1.errorbar(agg['dim_ratio'], agg['stability_mean'],
                yerr=agg['stability_std'], fmt='o-', capsize=5, lw=2,
                color=COLORS['primary'], markerfacecolor=COLORS['primary'],
                markeredgecolor='white', markersize=8)
    ax1.axvline(x=0.3, color=COLORS['secondary'], linestyle='--', alpha=0.7,
                label='Critical threshold (r = 0.3)')
    ax1.set_xlabel('Dimensional Ratio (D_observer / D_patient)')
    ax1.set_ylabel('Coupling Stability')
    ax1.set_title('Dimensional Matching and Code Stability')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.grid(True, alpha=0.3)

    # Right: Sync error vs dimensional ratio
    ax2.errorbar(agg['dim_ratio'], agg['error_mean'],
                yerr=agg['error_std'], fmt='s-', capsize=5, lw=2,
                color=COLORS['quaternary'], markerfacecolor=COLORS['quaternary'],
                markeredgecolor='white', markersize=8)
    ax2.set_xlabel('Dimensional Ratio (D_observer / D_patient)')
    ax2.set_ylabel('Synchronization Error')
    ax2.set_title('Tracking Error vs Observer Complexity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# =============================================================================
# 3. CLASSIFIER DEGRADATION SIMULATION
# =============================================================================

def generate_synthetic_patients_by_stratum(
    n_per_stratum: int = 1000,
    n_features: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic patient data modeling the U-shaped AUC pattern.

    Key insight from MIMIC-IV analysis:
    - Low multimorbidity: High D_eff (~44), clean signals → good AUC
    - Moderate multimorbidity: Moderate D_eff (~37), zone of maximum entropy → worst AUC
    - High multimorbidity: Low D_eff (~30), manifold collapse → better AUC

    The moderate stratum has maximum entropy because patients have activated
    multiple pathological dimensions but haven't synchronized into stereotyped
    disease attractors.

    Returns:
    - X: features (n_patients, n_features)
    - y: binary outcome
    - complexity: patient complexity score (1=simple, 2=moderate, 3=complex)
    """
    np.random.seed(random_state)

    X_list, y_list, complexity_list = [], [], []

    def generate_stratum_with_target_deff(n_samples, n_features, target_deff, noise_level, signal_strength):
        """Generate data with approximately target D_eff."""
        # Create eigenvalue spectrum that gives target D_eff
        # D_eff = (sum λ)² / sum(λ²)
        # For exponential decay: λ_i = exp(-i/tau), D_eff ≈ tau * (1 - exp(-n/tau))
        # Solve for tau given target D_eff

        # Create covariance with controlled spectrum
        n_active = int(target_deff * 1.2)  # Active dimensions
        eigenvalues = np.zeros(n_features)

        # Exponential decay in eigenvalues
        tau = target_deff / 2
        for i in range(min(n_active, n_features)):
            eigenvalues[i] = np.exp(-i / tau)

        # Add small noise floor
        eigenvalues += 0.01

        # Normalize
        eigenvalues = eigenvalues / eigenvalues.sum() * n_features

        # Generate random orthogonal basis
        Q, _ = np.linalg.qr(np.random.randn(n_features, n_features))

        # Create covariance matrix
        cov = Q @ np.diag(eigenvalues) @ Q.T

        # Generate data from this distribution
        X = np.random.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

        # Add noise
        X += np.random.randn(n_samples, n_features) * noise_level

        # Generate outcome based on first few dimensions (signal)
        n_signal = int(target_deff / 3)
        weights = np.random.randn(n_signal)
        logits = X[:, :n_signal] @ weights * signal_strength
        probs = 1 / (1 + np.exp(-logits))
        y = (np.random.rand(n_samples) < probs).astype(int)

        return X, y

    # Stratum 1: LOW multimorbidity
    # High D_eff (~44), very clean signals → BEST AUC (healthy patients are easy)
    # High target_deff to ensure D_eff > Moderate after noise effects
    X_low, y_low = generate_stratum_with_target_deff(
        n_per_stratum, n_features,
        target_deff=50, noise_level=0.05, signal_strength=2.0
    )
    X_list.append(X_low)
    y_list.append(y_low)
    complexity_list.append(np.ones(n_per_stratum, dtype=int))

    # Stratum 2: MODERATE multimorbidity (Zone of Maximum Entropy)
    # Moderate D_eff (~37), weak signal → WORST AUC
    # Lower noise than before to ensure D_eff < Low (monotonic decrease)
    X_mod, y_mod = generate_stratum_with_target_deff(
        n_per_stratum, n_features,
        target_deff=37, noise_level=0.3, signal_strength=0.4
    )
    # Additional label noise for maximum entropy regime
    flip_mask = np.random.rand(n_per_stratum) < 0.15
    y_mod[flip_mask] = 1 - y_mod[flip_mask]
    X_list.append(X_mod)
    y_list.append(y_mod)
    complexity_list.append(np.ones(n_per_stratum, dtype=int) * 2)

    # Stratum 3: HIGH multimorbidity (Complex)
    # Low D_eff (~30), collapsed manifold → BETTER than moderate (but not best)
    # Lipsitz-Goldberger: sick patients are stereotyped, easier to predict
    X_high, y_high = generate_stratum_with_target_deff(
        n_per_stratum, n_features,
        target_deff=28, noise_level=0.15, signal_strength=1.2
    )
    # Small label noise
    flip_mask = np.random.rand(n_per_stratum) < 0.03
    y_high[flip_mask] = 1 - y_high[flip_mask]
    X_list.append(X_high)
    y_list.append(y_high)
    complexity_list.append(np.ones(n_per_stratum, dtype=int) * 3)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    complexity = np.concatenate(complexity_list)

    return X, y, complexity


def generate_synthetic_patients(
    n_patients: int = 1000,
    n_features: int = 50,
    n_informative: int = 10,
    complexity_groups: int = 3,
    noise_level: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper for backward compatibility. Calls the new stratified generator.
    """
    return generate_synthetic_patients_by_stratum(
        n_per_stratum=n_patients // complexity_groups,
        n_features=n_features,
        random_state=42
    )


def classifier_by_complexity(
    X: np.ndarray,
    y: np.ndarray,
    complexity: np.ndarray,
    n_folds: int = 5
) -> pd.DataFrame:
    """
    Test classifier performance stratified by patient complexity.
    """
    results = []

    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    for comp_level in np.unique(complexity):
        mask = complexity == comp_level
        X_subset = X[mask]
        y_subset = y[mask]

        if len(y_subset) < 50:
            continue

        # Cross-validated performance
        scores = cross_val_score(clf, X_subset, y_subset, cv=n_folds, scoring='roc_auc')

        # Compute D_eff for this subset
        d_eff = compute_effective_dimensionality(X_subset)

        results.append({
            'complexity': comp_level,
            'n_patients': len(y_subset),
            'd_eff': d_eff,
            'auc_mean': scores.mean(),
            'auc_std': scores.std()
        })

    return pd.DataFrame(results)


def plot_classifier_degradation(df: pd.DataFrame, save_path: str = None):
    """Plot classifier performance vs patient complexity."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(df['complexity'], df['auc_mean'],
               yerr=df['auc_std'], fmt='o-', capsize=5, lw=2, markersize=10,
               color=COLORS['primary'], markerfacecolor=COLORS['primary'],
               markeredgecolor='white')

    ax.set_xlabel('Multimorbidity Burden')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('U-Shaped Performance: Zone of Maximum Entropy')
    ax.set_xticks(df['complexity'].values)
    ax.set_xticklabels(['Low', 'Moderate', 'High'])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # Add D_eff annotations
    for _, row in df.iterrows():
        ax.annotate(f'D_eff={row["d_eff"]:.1f}',
                   (row['complexity'], row['auc_mean'] - 0.05),
                   ha='center', color=COLORS['gray'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# =============================================================================
# 4. TREATMENT HETEROGENEITY SIMULATION
# =============================================================================

def simulate_treatment_heterogeneity(
    n_patients: int = 1000,
    n_diagnoses: int = 5,
    hidden_dims: int = 20,
    treatment_effect_base: float = 0.5
) -> pd.DataFrame:
    """
    Simulate treatment response heterogeneity within diagnostic categories.

    Patients with same diagnosis but different hidden state have different outcomes.
    """
    results = []

    for dx in range(n_diagnoses):
        n_dx = n_patients // n_diagnoses

        # Hidden patient state (not captured by diagnosis)
        hidden_state = np.random.randn(n_dx, hidden_dims)

        # True treatment effect depends on hidden state
        # (diagnosis captures only ~30% of relevant variation)
        hidden_effect = hidden_state @ np.random.randn(hidden_dims) * 0.3

        # Observed treatment response
        treatment_response = treatment_effect_base + hidden_effect + np.random.randn(n_dx) * 0.2

        # Compute D_eff of hidden state
        d_eff = compute_effective_dimensionality(hidden_state)

        for i in range(n_dx):
            results.append({
                'diagnosis': f'ICD_{dx}',
                'hidden_d_eff': d_eff,
                'treatment_response': treatment_response[i],
                'patient_id': i + dx * n_dx
            })

    return pd.DataFrame(results)


def plot_treatment_heterogeneity(df: pd.DataFrame, save_path: str = None):
    """Plot treatment response variance within diagnostic categories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Violin plot of treatment response by diagnosis
    diagnoses = df['diagnosis'].unique()
    data_by_dx = [df[df['diagnosis'] == dx]['treatment_response'].values for dx in diagnoses]

    parts = ax1.violinplot(data_by_dx, positions=range(len(diagnoses)),
                          showmeans=True, showmedians=True)
    # Style the violin plot with publication colors
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['primary'])
        pc.set_edgecolor(COLORS['primary'])
        pc.set_alpha(0.7)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        if partname in parts:
            parts[partname].set_edgecolor(COLORS['gray'])
            parts[partname].set_linewidth(1.5)

    ax1.set_xticks(range(len(diagnoses)))
    ax1.set_xticklabels(diagnoses, rotation=45)
    ax1.set_xlabel('Diagnostic Category')
    ax1.set_ylabel('Treatment Response')
    ax1.set_title('Treatment Response Heterogeneity\nWithin Diagnostic Categories')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Variance by diagnosis
    variance_by_dx = df.groupby('diagnosis')['treatment_response'].var().reset_index()
    ax2.bar(range(len(variance_by_dx)), variance_by_dx['treatment_response'],
            color=COLORS['secondary'], edgecolor='white', linewidth=1)
    ax2.set_xticks(range(len(variance_by_dx)))
    ax2.set_xticklabels(variance_by_dx['diagnosis'], rotation=45)
    ax2.set_xlabel('Diagnostic Category')
    ax2.set_ylabel('Response Variance')
    ax2.set_title('Hidden Dimensionality Creates\nWithin-Category Variance')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# =============================================================================
# MAIN - Run all simulations
# =============================================================================

def run_all_simulations(output_dir: str = '../results'):
    """Run all simulations and save results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/../figures', exist_ok=True)

    print("=" * 60)
    print("Dimensional Validity Bound - Simulation Suite")
    print("=" * 60)

    # 1. Diagnostic Cascade
    print("\n1. Diagnostic Cascade Simulation...")
    cascade_df = simulate_diagnostic_cascade(n_patients=10000, n_tests_range=(1, 50))
    cascade_df.to_csv(f'{output_dir}/diagnostic_cascade.csv', index=False)
    plot_diagnostic_cascade(cascade_df, f'{output_dir}/../figures/fig1_diagnostic_cascade.png')
    print(f"   At 20 tests: P(cascade) = {cascade_df[cascade_df['n_tests']==20]['p_cascade_simulated'].values[0]:.2%}")

    # 2. Dimensional Matching
    print("\n2. Dimensional Matching Simulation...")
    matching_df = dimensional_matching_experiment(
        patient_dims=100,
        observer_dims_list=[5, 10, 20, 30, 50, 70, 100],
        n_trials=20
    )
    matching_df.to_csv(f'{output_dir}/dimensional_matching.csv', index=False)
    plot_dimensional_matching(matching_df, f'{output_dir}/../figures/fig2_dimensional_matching.png')

    # Summarize
    agg = matching_df.groupby('dim_ratio')['stability'].mean()
    print(f"   Stability at r=0.1: {agg.iloc[0]:.3f}")
    print(f"   Stability at r=1.0: {agg.iloc[-1]:.3f}")

    # 3. Classifier Degradation (U-shaped pattern)
    print("\n3. Classifier Degradation Simulation...")
    X, y, complexity = generate_synthetic_patients(n_patients=3000, n_features=60)
    classifier_df = classifier_by_complexity(X, y, complexity)
    classifier_df.to_csv(f'{output_dir}/classifier_degradation.csv', index=False)
    plot_classifier_degradation(classifier_df, f'{output_dir}/../figures/fig3_classifier_degradation.png')
    print(f"   Low multimorbidity AUC: {classifier_df[classifier_df['complexity']==1]['auc_mean'].values[0]:.3f}")
    print(f"   Moderate multimorbidity AUC: {classifier_df[classifier_df['complexity']==2]['auc_mean'].values[0]:.3f} (zone of max entropy)")
    print(f"   High multimorbidity AUC: {classifier_df[classifier_df['complexity']==3]['auc_mean'].values[0]:.3f}")

    # 4. Treatment Heterogeneity
    print("\n4. Treatment Heterogeneity Simulation...")
    treatment_df = simulate_treatment_heterogeneity(n_patients=2000, n_diagnoses=5)
    treatment_df.to_csv(f'{output_dir}/treatment_heterogeneity.csv', index=False)
    plot_treatment_heterogeneity(treatment_df, f'{output_dir}/../figures/fig4_treatment_heterogeneity.png')

    variance_by_dx = treatment_df.groupby('diagnosis')['treatment_response'].var()
    print(f"   Mean within-diagnosis variance: {variance_by_dx.mean():.3f}")

    print("\n" + "=" * 60)
    print("All simulations complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: {output_dir}/../figures")
    print("=" * 60)


if __name__ == "__main__":
    run_all_simulations()
